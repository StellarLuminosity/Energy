import argparse
import os
import pdb
import time
import wandb
import torch
import torch.distributed as dist
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from tqdm.auto import tqdm
from typing import Any

from distill_bench.core.config_loader import load_config
from distill_bench.core.trainer import Trainer
from distill_bench.core.utils import prepare_dataset, get_dataset, is_main_process, main_print, fix_seed
from distill_bench.core.checkpoint import SimpleCheckpointer
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.environment import save_environment, collect_environment


# Watch GPU usage in real-time
# ssh kn149  # Your node
# watch -n 1 nvidia-smi

# Track memory
# free_b, total_b = torch.cuda.mem_get_info()
# used_b = total_b - free_b
# print(f"GPU {torch.cuda.current_device()} memory: {used_b/1024**3:.2f} / {total_b/1024**3:.2f} GiB")

# device = torch.cuda.current_device()
# allocated = torch.cuda.memory_allocated(device) / 1024**3
# print(f"Allocated: {allocated:.2f} GB")
# reserved = torch.cuda.memory_reserved(device) / 1024**3
# print(f"Reserved: {reserved:.2f} GB")

# torch.set_printoptions(profile="full")
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-SFT")

# ==================================================
# Main Training Function
# ==================================================
def main(args):
    """
    Simplified single teacher-student distillation pipeline.
    
    Args:
        args: Argparse namespace with config and mixed_precision flag
    """
    
    # Load config
    config = load_config(args.config)
    
    # ----------------------------------
    # DDP Setup and Initialization
    # ----------------------------------
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    fix_seed(config.seed)

    # ----------------------------------
    # Timer and Logging Start
    # ----------------------------------
    overall_start_time = time.time()
    main_print(f"--> Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ----------------------------------
    # Output Directory
    # ----------------------------------
    output_path = config.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    # ----------------------------------
    # Environment Metadata
    # ----------------------------------
    if is_main_process():
        save_environment(output_path, filename="environment.json")
    
    # ----------------------------------
    # Energy Tracking Setup
    # ----------------------------------
    energy_tracker = None
    if is_main_process() and getattr(config, 'energy_enabled', False):
        energy_tracker = EnergyTracker(
            output_dir=output_path,
            experiment_name=getattr(config, 'experiment_name', 'kd_experiment'),
            nvml_poll_interval_ms=getattr(config, 'energy_nvml_poll_ms', 500),
            track_cpu=getattr(config, 'energy_track_cpu', True),
        )
        main_print("Energy tracking enabled")
    
    # ----------------------------------
    # Wandb Initialization
    # ----------------------------------
    if is_main_process():
        # Generate run name if not provided
        run_name = config.wandb_run_name
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            student_short = config.student_model_name.split('/')[-1]
            teacher_short = config.teacher_model_name.split('/')[-1]
            run_name = f"{student_short}_from_{teacher_short}_{timestamp}"
        
        # Collect hardware metadata
        env_metadata = collect_environment()
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "teacher_model": config.teacher_model_name,
                "student_model": config.student_model_name,
                "num_epochs": config.num_epochs,
                "num_training_steps": config.num_training_steps,
                "batch_size": config.batch_size,
                "eval_batch_size": config.eval_batch_size,
                "learning_rate": config.learning_rate,
                "max_grad_norm": config.max_grad_norm,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "alpha": config.alpha,
                "temperature": config.kl_temperature,
                "dataset": config.dataset_name,
                "seed": config.seed,
                "mixed_precision": args.mixed_precision,
            },
        )
        
        # Log hardware metadata to W&B
        wandb.config.update({
            "hardware/gpu_count": len(env_metadata.get('gpus', [])),
            "hardware/gpu_type": env_metadata['gpus'][0]['name'] if env_metadata.get('gpus') else 'None',
            "hardware/gpu_vram_gb": env_metadata['gpus'][0]['memory_total_mb'] / 1024 if env_metadata.get('gpus') else 0,
            "hardware/cpu": env_metadata['cpu']['brand'],
            "hardware/cpu_cores": env_metadata['cpu']['physical_cores'],
            "software/pytorch": env_metadata['software']['pytorch_version'],
            "software/cuda": env_metadata['software']['cuda_version'],
            "software/transformers": env_metadata['software']['transformers_version'],
        })
        
        main_print(f"Wandb initialized: {wandb.run.url}")
    
    # ----------------------------------
    # Dataset Loading
    # ----------------------------------
    main_print("Loading dataset...")
    dataset = get_dataset(config)
    train_dataloader, eval_dataloader = prepare_dataset(
        dataset['train'],
        dataset['test'],
        config,
    )
    
    # ----------------------------------
    # Load Student Model
    # ----------------------------------
    main_print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
    )
    
    # ----------------------------------
    # FSDP Setup for Student
    # ----------------------------------
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    
    # Move to device and wrap with FSDP
    # FSDP will handle efficient sharding during wrapping
    student_model = student_model.to(device)
    for layer in student_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(student_model, **fsdp_kwargs)
    
    # ----------------------------------
    # Optimizer and Scheduler
    # ----------------------------------
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    
    num_training_steps = len(train_dataloader) * config.num_epochs if config.num_training_steps == 0 else config.num_training_steps
    num_warmup_steps = config.num_warmup_steps
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # ----------------------------------
    # Checkpointer Setup
    # ----------------------------------
    checkpointer = SimpleCheckpointer(output_path)
    
    # ----------------------------------
    # Resume from Checkpoint (if applicable)
    # ----------------------------------
    start_epoch = 0
    global_step = 0
    if config.resume_from_checkpoint:
        checkpoint_data = checkpointer.load(student_model, optimizer, lr_scheduler)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            global_step = checkpoint_data.get('global_step', 0)
            main_print(f"Resumed from epoch {start_epoch-1}, step {global_step}")
    
    # ----------------------------------
    # Initialize Trainer
    # ----------------------------------
    trainer = Trainer(
        student_model=student_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
        config=config,
    )
    
    # Sync trainer state with loaded checkpoint
    trainer.global_step = global_step
    trainer.epoch = start_epoch
    
    # ----------------------------------
    # Training Loop
    # ----------------------------------
    main_print("\n" + "="*50)
    main_print("Starting Training")
    main_print("="*50)
    
    # Start energy tracking for training stage
    if energy_tracker:
        energy_tracker.start_stage("student_train")
    
    total_tokens_processed = 0
    
    for epoch in range(start_epoch, config.num_epochs):
        trainer.epoch = epoch
        main_print(f"\nEpoch {epoch}/{config.num_epochs-1}")
        
        # ----------------------------------
        # Epoch Setup
        # ----------------------------------
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Initialize tracking variables
        epoch_train_loss = 0.0
        num_train_steps = 0
        eval_count = 0
        
        # ----------------------------------
        # Training Iteration
        # ----------------------------------
        progress_bar = tqdm(train_dataloader, disable=rank != 0, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Debug mode: stop after max_steps
            if config.debug_mode and trainer.global_step >= config.debug_max_steps:
                main_print(f"[DEBUG MODE] Reached max steps ({config.debug_max_steps}), stopping training")
                break
            
            # Train step (handles gradient accumulation internally)
            loss = trainer.train_step(batch)
            epoch_train_loss += loss
            num_train_steps += 1
            
            # Count tokens for energy tracking (count all tokens in batch)
            if energy_tracker and "input_ids" in batch:
                # Count based on labels (excludes padding marked as -100)
                if "labels" in batch:
                    batch_tokens = (batch["labels"] != -100).sum().item()
                else:
                    # Fallback: count all input tokens
                    batch_tokens = batch["input_ids"].numel()
                total_tokens_processed += batch_tokens
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'step': trainer.global_step
                })
            
            # ------ Periodic Evaluation ------
            if trainer.global_step > 0 and trainer.global_step % config.eval_steps == 0:
                dist.barrier()  # Sync before eval
                
                # Temporarily pause training energy tracking for eval
                if energy_tracker and energy_tracker.current_stage == "student_train":
                    energy_tracker.end_stage(tokens_processed=total_tokens_processed)
                    energy_tracker.start_stage("evaluation")
                
                eval_loss, should_stop = trainer.eval_step(eval_dataloader)
                main_print(f"Step {trainer.global_step}: eval_loss = {eval_loss:.4f}")
                eval_count += 1
                
                # Resume training energy tracking
                if energy_tracker and energy_tracker.current_stage == "evaluation":
                    energy_tracker.end_stage()
                    energy_tracker.start_stage("student_train")
                
                dist.barrier()  # Sync after eval
                
                # Check for early stopping
                if should_stop:
                    main_print("Early stopping: training terminated")
                    break
            
            # ------ Periodic Checkpointing ------
            # Skip in debug mode to avoid NCCL timeout
            if not config.debug_mode and trainer.global_step > 0 and trainer.global_step % config.save_steps == 0:
                trainer.save_checkpoint(loss=None) 
                
        # Skip end-of-epoch processing in debug mode (already stopped)
        if config.debug_mode and trainer.global_step >= config.debug_max_steps:
            main_print("[DEBUG MODE] Pipeline test complete!")
            break
        
        # ----------------------------------
        # End of Epoch Summary
        # ----------------------------------
        # Compute average training loss
        avg_train_loss = epoch_train_loss / num_train_steps if num_train_steps > 0 else 0.0
        
        # Run final evaluation for the epoch
        eval_loss, should_stop = trainer.eval_step(eval_dataloader)
        
        main_print(f"Epoch {epoch} Summary:")
        main_print(f"  Average Train Loss: {avg_train_loss:.4f}")
        main_print(f"  Eval Loss: {eval_loss:.4f}")
        
        if should_stop:
            main_print("Early stopping: training terminated")
            break

        # ----------------------------------
        # Save Epoch Checkpoint
        # ----------------------------------
        trainer.save_checkpoint(loss=eval_loss)
    
    # End energy tracking for training
    if energy_tracker and energy_tracker.current_stage:
        energy_tracker.end_stage(tokens_processed=total_tokens_processed)
    
    # ----------------------------------
    # Final Model Save
    # ----------------------------------
    # Synchronize before final save
    if dist.is_initialized():
        dist.barrier()
    
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    model_state_dict, _ = get_state_dict(student_model, optimizers=optimizer, options=options)
    
    if is_main_process():
        final_model_path = os.path.join(output_path, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # Save just the model state dict for inference
        torch.save(model_state_dict, os.path.join(final_model_path, "model.pt"))
        main_print(f"\nSaved final model to {final_model_path}")
        
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                config.student_model_name,
                torch_dtype=torch.bfloat16,
            )
            hf_model.load_state_dict(model_state_dict)
            hf_model.save_pretrained(os.path.join(final_model_path, "hf_format"))
            main_print(f"Also saved in HuggingFace format: {final_model_path}/hf_format")
        except Exception as e:
            main_print(f"Note: Could not save HuggingFace format ({e}), but .pt file is valid")

    if dist.is_initialized():
        dist.barrier()
    
    # ----------------------------------
    # Cleanup and Finalization
    # ----------------------------------
    total_time = time.time() - overall_start_time
    main_print(f"\nTraining completed in {total_time/3600:.2f} hours")
    main_print(f"Total tokens processed: {total_tokens_processed:,}")
    
    # Save energy summary
    if energy_tracker:
        summary_file = energy_tracker.save_summary({
            "total_time_hours": total_time / 3600,
            "total_tokens": total_tokens_processed,
        })
        main_print(f"Energy summary: {summary_file}")
        
        # Log to wandb
        if is_main_process():
            wandb.log(energy_tracker.get_wandb_metrics(prefix="energy"))
    
    # Finish wandb logging
    if is_main_process():
        wandb.finish()
    
    # Clean up distributed processes
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================================================
# Script Entry Point
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Teacher-Student Distillation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                        help="Use mixed precision training")
    args = parser.parse_args()
    main(args)
