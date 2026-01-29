import argparse
import sys
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from distill_bench.core.config_loader import load_config
from distill_bench.core.utils import prepare_dataset, get_dataset, fix_seed
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.environment import save_environment


def load_model(model_path=None, model_name=None, student_model_name=None, device="cuda"):
    """Load model from checkpoint (.pt or HF directory) or HuggingFace hub.

    Args:
        model_path: Path to .pt file or HF format directory
        model_name: HuggingFace model name
        student_model_name: Student model name for initializing structure (required for checkpoints)
        device: Device to load model to
    """
    if model_path:
        if student_model_name is None:
            raise ValueError("student_model_name required when loading from checkpoint")

        # Initialize model structure first
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.bfloat16,
        )
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        student_model.gradient_checkpointing_enable()
        if hasattr(student_model.config, "use_cache"):
            student_model.config.use_cache = False

        if os.path.isdir(model_path):
            # Treat directory as HF-format checkpoint
            print(f"Detected directory checkpoint; loading HF format from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        elif os.path.isfile(model_path):
            print(f"Detected single file checkpoint format")
            print(f"Loading from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                # Training checkpoint format
                model.load_state_dict(checkpoint["model_state_dict"])
                if "epoch" in checkpoint:
                    print(f"Checkpoint info: Epoch {checkpoint['epoch']}, " f"Step {checkpoint.get('global_step', 'N/A')}")
            else:
                # Direct state dict (final model format)
                model.load_state_dict(checkpoint)
                print("Loaded final model state dict")
        else:
            raise ValueError(f"Path {model_path} is neither a file nor a directory!")

    elif model_name:
        # Load from HuggingFace
        print(f"Loading from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

    return model.to(device)


def compute_ce_loss(model, dataloader, device):
    """Compute loss on the test dataset."""
    model.eval()

    total_ce_loss = 0.0
    total_tokens = 0
    num_batches = 0

    print("\nEvaluating model...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", file=sys.stdout)):
            input_ids = batch["input_ids"].type(torch.LongTensor).to(device)
            attention_mask = batch["attention_mask"].type(torch.LongTensor).to(device)
            labels = batch["labels"].type(torch.LongTensor).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for next-token prediction
            vocab_size = logits.size(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Create mask for valid tokens (ignore -100)
            ignore_index = -100
            valid_mask = shift_labels != ignore_index
            valid_count = valid_mask.sum().item()

            if valid_count > 0:
                # Compute cross-entropy loss
                ce_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction="sum")

                total_ce_loss += ce_loss.item()
                total_tokens += valid_count
                num_batches += 1

            # Periodic cleanup
            del outputs, logits, shift_logits, shift_labels
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

    # Compute averages
    if total_tokens > 0:
        avg_ce_loss = total_ce_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    else:
        avg_ce_loss = float("inf")
        perplexity = float("inf")

    return avg_ce_loss, perplexity, num_batches


def eval_main(args):
    """Main evaluation function."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load config
    config = load_config(args.config)

    # Setup output directory for energy logs
    eval_output_dir = os.path.join(config.output_dir, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Setup energy tracking
    energy_tracker = None
    if getattr(config, "energy_enabled", False):
        energy_tracker = EnergyTracker(
            run_dir=config.get("output.run_dir"),
            experiment_name=f"{config.experiment_name}_eval",
            config=config,
        )
        print("Energy tracking enabled for evaluation")

    # Load dataset
    print("Loading test dataset...")
    dataset = get_dataset(dataset_path=config.dataset_teacher_logprobs)

    # Create dataloader
    _, eval_dataloader = prepare_dataset(
        dataset["train"],
        dataset["test"],
        config,
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        student_model_name=config.student_model_name if hasattr(config, "student_model_name") else None,
        device=device,
    )

    # Start energy tracking for evaluation
    if energy_tracker:
        energy_tracker.start_stage("eval_core")

    # Evaluate
    avg_ce_loss, perplexity, num_batches = compute_ce_loss(model, eval_dataloader, device)

    # End energy tracking
    if energy_tracker:
        # Count total tokens evaluated
        total_tokens = 0
        for batch in eval_dataloader:
            if "labels" in batch:
                total_tokens += (batch["labels"] != -100).sum().item()

        energy_tracker.end_stage(tokens_processed=total_tokens)
        summary_file = energy_tracker.save_summary()
        print(f"Energy summary saved to: {summary_file}")

    # Print results
    if args.model_path:
        print(f"Model: {args.model_path}")
    else:
        print(f"Model: {args.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Batches processed: {num_batches}")
    print(f"Cross-Entropy Loss: {avg_ce_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=" * 70)

    return avg_ce_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distributed checkpoint (directory):
  python kd_eval.py --config configs/experiments/kd_7b_to_1b.yaml --model_path /path/to/model_log/singular/checkpoints/checkpoint_epoch0_step5000
  
  # Final model (.pt file):
  python kd_eval.py --config configs/experiments/kd_7b_to_1b.yaml --model_path /path/to/model_log/singular/final_model/model.pt
  
  # Student baseline:
  python kd_eval.py --config configs/experiments/kd_7b_to_1b.yaml --model_name allenai/OLMo-2-0425-1B-SFT
        """,
    )

    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to checkpoint (directory or .pt file). "
        "Supports: (1) Distributed checkpoint dir (checkpoint_epoch0_step5000/), "
        "(2) Single .pt file (model.pt)",
    )
    model_group.add_argument("--model_name", type=str, help="HuggingFace model name (e.g., allenai/OLMo-2-1B)")

    args = parser.parse_args()
    eval_main(args)
