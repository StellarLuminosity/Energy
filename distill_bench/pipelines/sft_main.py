"""
SFT Training Pipeline for Data Distillation.

Trains student model on teacher-generated synthetic data using standard
supervised fine-tuning (cross-entropy loss only, no KD).
"""

import argparse
import os
import time
import torch
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.utils import prepare_dataset, is_main_process, main_print, fix_seed
from distill_bench.data.synthetic_generation import load_or_generate_synthetic_dataset

def compute_sft_loss(model, batch, device):
    """Compute standard cross-entropy loss for SFT."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        reduction='mean'
    )
    
    return loss


def train_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch, config, use_wandb):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        loss = compute_sft_loss(model, batch, device)
        
        # Backward and optimize
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/step": num_steps,
                }, step=num_steps)
            
            num_steps += 1
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Debug mode early exit
        if config.debug_mode and num_steps >= config.debug_max_steps:
            main_print(f"[DEBUG MODE] Stopping at {num_steps} steps")
            break
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return avg_loss, num_steps


def eval_model(model, eval_loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            loss = compute_sft_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main(args):
    """Main SFT training pipeline."""
    # Load config
    config = load_config(args.config)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seed(config.seed)
    
    start_time = time.time()
    main_print(f"Starting SFT training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main_print(f"Config: {args.config}")
    
    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Energy tracking
    energy_tracker = None
    if config.energy_enabled:
        energy_tracker = EnergyTracker(
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
        )
        energy_tracker.start()
    
    # W&B
    use_wandb = config.wandb_enabled
    if use_wandb:
        run_name = config.wandb_run_name or f"sft_{config.student_model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.to_dict(),
        )
    
    # Load or generate synthetic dataset
    main_print("Loading/generating synthetic dataset...")
    if energy_tracker:
        energy_tracker.start_stage("data_generation")
    
    synthetic_dataset = load_or_generate_synthetic_dataset(config, energy_tracker)
    
    if energy_tracker:
        energy_tracker.end_stage()
    
    main_print(f"Synthetic dataset: {len(synthetic_dataset['train'])} train, {len(synthetic_dataset['test'])} eval")
    
    # Prepare dataloaders
    train_loader, eval_loader = prepare_dataset(
        synthetic_dataset['train'],
        synthetic_dataset['test'],
    )
    
    # Load student model
    main_print(f"Loading student model: {config.student_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    num_training_steps = len(train_loader) * config.num_epochs if config.num_training_steps == 0 else config.num_training_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    main_print("\n" + "="*50)
    main_print("Starting Training")
    main_print("="*50)
    
    if energy_tracker:
        energy_tracker.start_stage("student_train")
    
    for epoch in range(config.num_epochs):
        main_print(f"\nEpoch {epoch}/{config.num_epochs-1}")
        
        # Train
        train_loss, num_steps = train_epoch(
            model, train_loader, optimizer, lr_scheduler,
            device, epoch, config, use_wandb
        )
        
        # Evaluate
        eval_loss = eval_model(model, eval_loader, device)
        
        main_print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/epoch": epoch,
            }, step=num_steps)
        
        if config.debug_mode:
            break
    
    if energy_tracker:
        # Approximate tokens processed (batch_size * seq_len * steps)
        tokens_processed = config.batch_size * 1024 * num_steps * config.gradient_accumulation_steps
        energy_tracker.end_stage(tokens_processed=tokens_processed)
    
    # Save final model
    main_print("\nSaving final model...")
    final_model_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    main_print(f"Saved to: {final_model_path}")
    
    # Finalize
    total_time = time.time() - start_time
    main_print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    if energy_tracker:
        energy_tracker.stop()
        energy_tracker.save_summary()
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training on Synthetic Data")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    main(args)
