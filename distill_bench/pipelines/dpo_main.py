"""
Preference (DPO) distillation pipeline.

Runs a DPO training loop on preference pairs, with optional on-the-fly
generation of pairs via a judge model and energy tracking.
"""

import argparse
import os
import time
from datetime import datetime

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from distill_bench.core.config_loader import load_config
from distill_bench.core.dpo_trainer import DPOTrainer
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.utils import fix_seed, main_print
from distill_bench.data.preference_dataset import (
    create_dpo_dataloaders,
    load_or_build_preference_dataset,
    prepare_dpo_dataset,
)

def main(args):
    """Run the DPO pipeline."""
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    start_time = time.time()
    main_print(f"Starting DPO run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main_print(f"Config: {args.config}")

    energy_tracker = None
    if config.energy_enabled:
        energy_tracker = EnergyTracker(
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
            nvml_poll_interval_ms=config.energy_nvml_poll_ms,
            track_cpu=config.energy_track_cpu,
            country_iso_code=config.energy_country_iso,
        )

    use_wandb = config.wandb_enabled
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Preference dataset (load or generate)
    preference_dataset = load_or_build_preference_dataset(config, tokenizer, energy_tracker)
    tokenized_dataset = prepare_dpo_dataset(preference_dataset, tokenizer, config)
    train_loader, eval_loader = create_dpo_dataloaders(tokenized_dataset, tokenizer, config)

    # Models
    main_print(f"Loading policy model: {config.policy_model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.policy_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print(f"Loading reference model: {config.reference_model_name}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        config.reference_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
    num_training_steps = (
        len(train_loader) * config.num_epochs if config.num_training_steps == 0 else config.num_training_steps
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if use_wandb:
        run_name = config.wandb_run_name or f"dpo_{config.policy_model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.to_dict(),
        )

    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        beta=config.dpo_beta,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        use_wandb=use_wandb,
    )

    tokens_seen = 0
    if energy_tracker:
        energy_tracker.start_stage("dpo_training")

    for epoch in range(config.num_epochs):
        main_print(f"\nEpoch {epoch}/{config.num_epochs - 1}")
        train_loss, tokens_epoch = trainer.train_epoch(train_loader, device, energy_tracker)
        tokens_seen += tokens_epoch

        eval_loss = trainer.eval_epoch(eval_loader, device)
        main_print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        if config.debug_mode:
            main_print("[DEBUG MODE] Stopping after first epoch")
            break

    if energy_tracker:
        energy_tracker.end_stage(tokens_processed=tokens_seen)

    final_path = os.path.join(config.output_dir, "final_policy")
    os.makedirs(final_path, exist_ok=True)
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    main_print(f"Saved final policy to {final_path}")

    total_time = time.time() - start_time
    main_print(f"DPO run finished in {total_time/3600:.2f} hours; tokens processed: {tokens_seen:,}")

    if energy_tracker:
        summary_file = energy_tracker.save_summary()
        main_print(f"Energy summary: {summary_file}")
        if use_wandb:
            wandb.log(energy_tracker.get_wandb_metrics(prefix="energy"))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preference distillation with DPO")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    main(args)

