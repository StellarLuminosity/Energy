import argparse
import os
import sys
import time
import torch
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2ForCausalLM, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset
from config_loader import get_config_from_args

config = get_config_from_args()
from trainer import Trainer, DistillTrainer
from utils import (CSVLogger, prepare_dataset, format_time_elapsed, 
                  is_main_process, main_print, check_batch_shape, fix_seed,
                  inspect_mixed_precision, inspect_model,
                  set_modules_to_forward_prefetch, set_modules_to_backward_prefetch,
                  create_manifest, build_run_identity, get_directory, init_wandb_run, slurm_term_handler, 
                  ManifestManager, DistillDataset, get_round_path, cleanup_and_exit, 
                  exception_handler, load_loss_jsonls, top_k_percent_ids_sorted)
from ensemble import ModelEnsemble, EnsembleLoader
from checkpoint import Checkpointer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
from shard_weight import *
import atexit
from pathlib import Path
from datasets import Dataset, DatasetDict
import wandb
import signal, threading, functools

def train_single_round(start_round, round_num, dataset, output_path, logger, wandb_run, overall_start_time, rank, device, ensembleloader, checkpointer, args, manifest, lr_scheduler=None):
    """ Train a single round of the ensemble distillation process. """
    main_print(f"\n{'='*50}")
    round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
    main_print(f"{'='*50}")

    # -------------------------------------
    # Update wandb run name for this round
    # -------------------------------------
    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"{config.run_name}_round_{round_num}"
        wandb_run.log({"round": round_num})
    
    # -------------------------------------
    # Update manifest with current round
    # -------------------------------------
    if is_main_process():
        manifest.update('round', round_num, section='train')
        manifest.set_status('TRAINING')

    # ----------------------------------
    # Load and Shard Student Model
    # ----------------------------------
    if not config.ensemble_random_init:
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
    else:
        cfg = AutoConfig.from_pretrained(config.student_model_name)
        student_model = Qwen2ForCausalLM(cfg).to('cuda')

    # ----------------------------------
    # Create optimizer
    # ----------------------------------
    optim = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)
    
    # ----------------------------------
    # Checkpoint Loading
    # ----------------------------------
    lr_scheduler_loaded = None
    if round_num == start_round and config.resume_from_checkpoint:
        main_print(f"Loading checkpoint for round {round_num}")
        try:
            state = checkpointer.load(student_model, optim)
            if state.get("lr_scheduler_state"): 
                lr_scheduler_loaded = state["lr_scheduler_state"]
            global_step = int(state.get("global_step", state.get("step", 0))) 
            epoch = int(state.get("epoch", 0))
            start_epoch = epoch + 1
            main_print(f"Successfully loaded checkpoint state")
        except Exception as e:
            main_print(f"Warning: Could not load checkpoint state: {e}")
            main_print(f"Continuing with fresh optimizer/scheduler")
    else:
        # ----------------------------------
        # Mixed precision setup
        # ----------------------------------
        fsdp_kwargs = {}
        if args.mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,  # 16-bit precision for model parameters
                reduce_dtype=torch.float32,  # 32-bit precision for reduction operations
            )

        # ----------------------------------
        # Shard Model
        # ----------------------------------
        for layer in student_model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(student_model, **fsdp_kwargs)

        if args.explicit_prefetching:
            set_modules_to_forward_prefetch(student_model, num_to_forward_prefetch=2)
            set_modules_to_backward_prefetch(student_model, num_to_backward_prefetch=2)

        inspect_model(student_model)

        if args.mixed_precision: inspect_mixed_precision(student_model)

    # ----------------------------------
    # Load or Update Ensemble Models
    # ----------------------------------
    ensemble = None
    if hasattr(ensembleloader, 'current_ensemble'):
        ensemble = ensembleloader.current_ensemble
    elif round_num > 0:
        ensemble = ensembleloader.load_or_update_ensemble(None, device="cuda")
    
    # ----------------------------------
    # Create LR Scheduler
    # ----------------------------------
    train_dataloader, _ = prepare_dataset(
        dataset['train'],
        dataset['test'],
    )
    
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warmup_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if lr_scheduler_loaded and round_num == start_round and config.resume_from_checkpoint:
        lr_scheduler.load_state_dict(lr_scheduler_loaded)
        main_print("Loaded LR scheduler state from checkpoint")
        
    # ----------------------------------
    # Initialize trainer 
    # ----------------------------------
    trainer = DistillTrainer(
        student_model, 
        optim, 
        lr_scheduler,
        ensemble,
        logger=logger,
        round_num=round_num,
        checkpointer=checkpointer,
        overall_start_time=overall_start_time,  
        wandb_run=wandb_run if is_main_process() else None,
    )
    trainer.prepare_train()

    # ----------------------------------
    # SLURM Signal Handling
    # ----------------------------------
    handler = functools.partial(slurm_term_handler, trainer=trainer, output_path=output_path, manifest=manifest)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------
    for epoch_num in range(0, config.num_train_epochs):
        epoch_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        main_print(f"\n{'='*50}")
        main_print(f"--> Starting Epoch {epoch_num} at: {epoch_start_datetime}")
        main_print(f"{'='*50}")
        
        # ----------------------------------
        # Load dataset for this epoch
        # ----------------------------------
        train_dataloader, eval_dataloader = prepare_dataset(
            dataset['train'],
            dataset['test'],
        )
        
        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch_num)

        if hasattr(eval_dataloader, "sampler") and hasattr(eval_dataloader.sampler, "set_epoch"):
            eval_dataloader.sampler.set_epoch(epoch_num)   # <--- add this

        if is_main_process():
            check_batch_shape(train_dataloader)

        train_dl_iterator = iter(train_dataloader)

        # ----------------------------------
        # Training Loop
        # ---------------------------------- 
        # Toggle for quick tests
        count = 0
        for step_idx in tqdm(range(len(train_dataloader)), disable=rank != 0, file=sys.stdout, mininterval=1.0, ncols=100):
            if args.explicit_prefetching:
                trainer.model.unshard()
            batch = next(train_dl_iterator)
            trainer.step(batch, eval_dataloader, epoch_num)
            # if trainer.should_stop: 
            #     main_print("Early stopping triggered")
            #     break
            # Toggle for quick tests
            # if count == 10:
            #     break
            # count += 1

        dist.barrier()

    # ----------------------------------
    # Collect metrics
    # ----------------------------------
    round_metrics = {
        'round_num': round_num,
        'final_loss': trainer.current_loss if hasattr(trainer, 'current_loss') else None,
        'min_eval_loss': trainer.min_eval_loss if hasattr(trainer, 'min_eval_loss') else None,
        'total_steps': trainer.tr_step if hasattr(trainer, 'tr_step') else None,
    }

    # ----------------------------------
    # Save model for ensemble
    # ----------------------------------
    ensemble_dir = ensembleloader.save_model_for_ensemble(student_model, round_num)
    main_print(f"Saved ensemble model at: {ensemble_dir}")

    del student_model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ---------------------------------------
    # Update wandb run name and log metrics
    # ---------------------------------------
    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"round_{round_num}_epoch_{epoch_num}"
        wandb_run.log({"epoch": epoch_num, "round": round_num}) 

    if is_main_process():
        manifest.update({
            f'round_{round_num}_final_loss': round_metrics['final_loss'],
            f'round_{round_num}_min_eval_loss': round_metrics['min_eval_loss'],
            f'round_{round_num}_total_steps': round_metrics['total_steps'],
        }, section='outcomes')
    
    return ensemble_dir, round_metrics


def main(args):
    # ----------------------------------
    # Pipeline
    # ----------------------------------
    # Raw Dataset → Chat Template → Tokenization → Label Creation → Teacher Inference → Logit Caching → Distributed Sampling → Custom Collation → FSDP2 Model → Training Loop
    #  ↓              ↓                  ↓               ↓                 ↓                   ↓                  ↓                  ↓               ↓             ↓
    # Messages    Single String       Token IDs     Loss Labels       Top-K Logits        Cached Data           Batches            Padded         Sharded       Training
    # (JSON)      (Text)             (Tensors)       (-100/IDs)         (Tensors)           (Disk)               (GPU)              (GPU)          (GPU)         (GPU)

    # ----------------------------------
    # Directory Structure
    # ----------------------------------   
    # Run_ID Format: {timestamp}-{git_hash}-{hyperparameter_fingerprint}
    # Example: "20241221-143025-a1b2c3d4-e5f6g7h8"
    #
    # {config.base_output_dir}/
    # └── {output_path (run_id)}/                           # Unique run directory
    #     ├── CSV_metrics.csv                 # Training metrics log (CSVLogger)
    #     ├── manifest.txt                    # Run metadata and configuration
    #     ├── STATUS.RUNNING                  # Status sentinel (during training)
    #     ├── STATUS.DONE                     # Status sentinel (on completion)
    #     ├── STATUS.FAILED                   # Status sentinel (on failure)
    #     └── checkpoints/                    # Model checkpoints directory (Checkpointer)
    #         ├── 0/                          # Round 0 checkpoints
    #         │   ├── step_00001000_loss_2.3456/
    #         │   │   ├── model_state_dict.pt
    #         │   │   ├── optim_state_dict.pt
    #         │   │   └── training_state.pt
    #         │   └── step_00003000_loss_1.9876/ # Final checkpoint for round 0
    #         ├── 1/                          # Round 1 checkpoints
    #         │   ├── step_00001000_loss_1.8765/
    #         │   └── step_00002000_loss_1.7654/
    #         └── 2/                          # Round 2 checkpoints
    #             └── step_00001500_loss_1.6543/
    #     ├── round_0_model/                  # Final inference-ready model for round 0
    #     │   ├── model_state_dict.pt
    #     │   ├── config.json
    #     │   ├── tokenizer.json
    #     │   └── generation_config.json
    #     └── round_1_model/                  # Final inference-ready model for round 1
    #         ├── model_state_dict.pt
    #         └── ... (etc.)

    # ----------------------------------
    # DDP Setup
    # ----------------------------------

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    fix_seed(config.seed)

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_print(f"--> Starting training at: {overall_start_datetime}\n")

    # ----------------------------------
    # Exception Handling
    # ----------------------------------
    # from utils import setup_exception_handling
    # setup_exception_handling()

    # ----------------------------------
    # Run Configuration 
    # ----------------------------------
    fix_seed(config.seed)

    if config.resume_from_checkpoint:
        output_path = config.checkpointed_dir
        run_id = os.path.basename(output_path)
        wandb_id = None  # Will be loaded from manifest
        wandb_name = None
        slug = None
    else:
        if torch.distributed.get_rank() == 0:
            run_id, slug, wandb_name, wandb_id = build_run_identity()
            output_path = get_directory(run_id)
        else:
            run_id, slug, wandb_name, wandb_id = build_run_identity()
            output_path = os.path.join(config.base_output_dir, run_id)

    # ----------------------------------
    # Dataset Loading
    # ----------------------------------
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset() if config.synthetic_data else dataClass.get_teacher_logprobs()
    
    # ----------------------------------
    # ID Tracking 
    # ----------------------------------
    if getattr(config, 'enable_id_tracking', True):
        loss_log_path = os.path.join(config.logs_dir, 'loss_log_0.jsonl')
        if os.path.exists(loss_log_path) and os.path.getsize(loss_log_path) > 0:
            # Second+ run: filter by most difficult examples
            main_print(f"Found existing loss logs, filtering to top {config.difficulty_filter_percentage}% most difficult examples")
            by_id, all_rows = load_loss_jsonls(loss_log_path)
            if len(by_id) > 0:  # Only filter if we have loss data
                top_ids = top_k_percent_ids_sorted(by_id, config.difficulty_filter_percentage)
                dataset = dataset.filter(lambda ex: ex['id'] in top_ids, num_proc=32)
                main_print(f"Filtered dataset to {len(top_ids)} most difficult examples ({config.difficulty_filter_percentage}% of {len(by_id)})")
            else:
                main_print("No loss data found in logs, proceeding with full dataset")
        else:
            main_print("No loss logs found, proceeding with full dataset for first run")
            if is_main_process():
                os.makedirs(config.logs_dir, exist_ok=True)
    else:
        main_print("ID tracking disabled, proceeding with full dataset")
    
    # Ensure all processes are synchronized after dataset filtering and directory creation
    dist.barrier()

    # ----------------------------------
    # Create Checkpointer Instance
    # ----------------------------------
    checkpointer = Checkpointer(output_path) 

    # ----------------------------------
    # Logging and WandB config
    # ----------------------------------
    logger = None
    if is_main_process():
        logger = CSVLogger(output_path, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time)
        atexit.register(logger.flush)
    
    wandb_run = init_wandb_run() if is_main_process() else None

    # ----------------------------------
    # Manifest file
    # ----------------------------------
    if config.resume_from_checkpoint:
        # Load existing manifest
        manifest_path = os.path.join(output_path, "manifest.txt")
        manifest = ManifestManager(manifest_path)
        start_round = manifest.get('round', section='train', default=0)
        start_round = start_round + 1 if start_round > 0 else 0
    else:
        if is_main_process():
            manifest = create_manifest(output_path, start_time_str=overall_start_datetime, 
                                     wandb_run=wandb_run, wandb_id=wandb_id)
        else:
            manifest_path = os.path.join(output_path, "manifest.txt")
            manifest = ManifestManager(manifest_path)
        start_round = 0

    # ----------------------------------
    # Ensemble Loader
    # ----------------------------------
    ensembleloader = EnsembleLoader(output_path)

    # ----------------------------------
    # Metrics Output and Logging
    # ----------------------------------
    if is_main_process():   
        metadata_dict = {
            "Run id": run_id,
            "Slug": slug if slug else "N/A",
            "Wandb run id": wandb_run.id if wandb_run else None,
            "Teacher Model": config.teacher_model_name,
            "Student Model": config.student_model_name,
            "Dataset Name": config.dataset_name,
            "Dataset Type": config.dataset_type,
            "Alpha": config.alpha,
            "Learning rate": config.learning_rate,
            "Total Rounds": config.total_rounds,
            "Steps per Round": config.steps_per_round,
            "Eval batch size": config.eval_batch_size,
            "Start Time": overall_start_datetime,
            "Model Save Dir": output_path,
        }
    main_print("\n==== RUN CONFIGURATION ====")
    main_print(f"Run: {run_id}")
    main_print(f"Created logging directory: {output_path}")
    main_print(f"Models stored in: {output_path}")
    main_print(f"{run_id}")
    if is_main_process():
        for k, v in metadata_dict.items():
            main_print(f"{k}: {v}")
    main_print("===========================")

    if is_main_process() and not config.resume_from_checkpoint: logger.log(function="main", phase="none", round_num=0, metadata=metadata_dict)

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------
    ensemble_model = None
    lr_scheduler = None  # Will be loaded from checkpoint if resuming
    if config.resume_from_checkpoint:
        ensemble_model = ensembleloader.load_or_update_ensemble(None, device="cuda")
        
    for round_num in range(start_round, config.total_rounds):
        ensembleloader.current_ensemble = ensemble_model

        ensemble_dir, metrics = train_single_round(
            start_round = start_round,
            round_num=round_num,
            dataset=dataset,
            output_path=output_path,
            logger=logger,
            wandb_run=wandb_run,
            overall_start_time=overall_start_time,
            rank=rank,
            device=device,
            ensembleloader = ensembleloader,
            checkpointer=checkpointer,
            args=args,
            manifest=manifest,
            lr_scheduler=lr_scheduler if (round_num == start_round and config.resume_from_checkpoint) else None,
        )
        
        if is_main_process():
            main_print(f"Completed round {round_num}")
            main_print(f"Round metrics: {metrics}")
        
        if ensemble_model is None:
            ensemble_model = ModelEnsemble(
                [ensemble_dir],
                torch_dtype=torch.bfloat16,
                vocab_size=config.student_vocab_size,
            )
            ensemble_model = ensemble_model.to("cuda")
        else:
            ensemble_model.add_model(ensemble_dir)

    # ----------------------------------
    # Cleanup and Final Summary
    # ----------------------------------
    try:
        # Final barrier to ensure all processes are synchronized before cleanup
        dist.barrier()
        
        if is_main_process() and wandb_run is not None:
            wandb_run.finish()
            main_print("--> Finished wandb run")
            
        # Final summary
        total_time = time.time() - float(overall_start_time)
        if is_main_process():
            manifest.finalize(success=True, wall_time_sec=total_time)
            main_print("\n" + "="*60)
            main_print("TRAINING COMPLETED SUCCESSFULLY")
            main_print("="*60)
            main_print(f"Total rounds completed: {config.total_rounds}")
            main_print(f"Output directory: {output_path}")
            main_print(f"Total training time: {format_time_elapsed(total_time)}")
            main_print(f"Final manifest status: {manifest.get('status', default='UNKNOWN')}")
            main_print("="*60)
        
        # Clear CUDA cache before destroying process group
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Final barrier before cleanup
        dist.barrier()
        
    except Exception as e:
        if is_main_process():
            print(f"Warning during cleanup: {e}")
    
    finally:
        # Destroy process group with error handling
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            if is_main_process():
                print(f"Warning during process group cleanup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
