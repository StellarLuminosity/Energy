import argparse
import sys
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
import torch.distributed.checkpoint as dcp

from distill_bench.core.config_loader import load_config
from distill_bench.core.checkpoint import AppState
from distill_bench.core.utils import prepare_dataset, get_dataset, fix_seed

# Load config
config = load_config('configs/experiments/kd_7b_to_1b.yaml')


def load_distributed_checkpoint(checkpoint_dir, model):
    """Load a distributed checkpoint (directory format) into a model.
    
    This handles checkpoints saved with dcp.save() which are stored as
    directories with multiple shard files.
    
    For evaluation, we load the model weights directly without using AppState.
    """
    
    print(f"Loading distributed checkpoint from: {checkpoint_dir}")
    
    # Create a matching optimizer for loading (must match training optimizer type)
    # Training uses AdamW, so we must use AdamW here too
    dummy_optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    app_state = AppState(model=model, optimizer=dummy_optim, lr_scheduler=None)
    
    # Load using dcp.load with AppState
    state_dict = {"app": app_state}
    dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
    
    # Clean up optimizer (we only needed it for loading)
    del dummy_optim
    
    print(f"✓ Loaded checkpoint - Epoch: {app_state.epoch}, Step: {app_state.global_step}")
    return app_state.epoch, app_state.global_step, app_state.loss


def load_model(model_path=None, model_name=None, student_model_name=None, device='cuda'):
    """Load model from checkpoint, distributed checkpoint, or HuggingFace.
    
    Supports three formats:
    1. Single .pt file: final_model/model.pt
    2. Distributed checkpoint directory: checkpoint_epoch0_step5000/
    3. HuggingFace model name: allenai/OLMo-2-0425-1B-SFT
    
    Args:
        model_path: Path to checkpoint file or directory
        model_name: HuggingFace model name
        student_model_name: Student model name for initializing structure (required for checkpoints)
        device: Device to load model to
    """
    if model_path:
        if student_model_name is None:
            raise ValueError("student_model_name required when loading from checkpoint")
        
        # Initialize model structure first
        model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.bfloat16,
        )
        
        # Check if path is a directory (distributed checkpoint) or file (single .pt)
        if os.path.isdir(model_path):
            # Distributed checkpoint (directory with shards)
            print(f"Detected distributed checkpoint format")
            epoch, step, loss = load_distributed_checkpoint(model_path, model)
            print(f"Checkpoint info: Epoch {epoch}, Step {step}, Loss {loss:.4f}")
            
        elif os.path.isfile(model_path):
            # Old format: Single .pt file
            print(f"Detected single file checkpoint format")
            print(f"Loading from: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Training checkpoint format
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    print(f"Checkpoint info: Epoch {checkpoint['epoch']}, "
                          f"Step {checkpoint.get('global_step', 'N/A')}")
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
                ce_loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=ignore_index,
                    reduction='sum'
                )
                
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
        avg_ce_loss = float('inf')
        perplexity = float('inf')
    
    return avg_ce_loss, perplexity, num_batches


def eval_main(args):
    """Main evaluation function."""
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load dataset
    print("Loading test dataset...")
    dataset = get_dataset(config)
    
    # Create dataloader
    _, eval_dataloader = prepare_dataset(
        dataset['train'],
        dataset['test'],
        config,
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        student_model_name=config.student_model_name if hasattr(config, 'student_model_name') else None,
        device=device
    )
    
    # Evaluate
    avg_ce_loss, perplexity, num_batches = compute_ce_loss(
        model, eval_dataloader, device
    )
    
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
    print("="*70)
    
    return avg_ce_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distributed checkpoint (directory):
  DISTILL_CONFIG=configs/experiments/kd_7b_to_1b.yaml python kd_eval.py --model_path /scratch/klambert/model_log/singular/checkpoints/checkpoint_epoch0_step5000
  
  # Final model (.pt file):
  DISTILL_CONFIG=configs/experiments/kd_7b_to_1b.yaml python kd_eval.py --model_path /scratch/klambert/model_log/singular/final_model/model.pt
  
  # Student baseline:
  DISTILL_CONFIG=configs/experiments/kd_7b_to_1b.yaml python kd_eval.py --model_name allenai/OLMo-2-0425-1B-SFT
        """
    )
    
    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path", 
        type=str, 
        help="Path to checkpoint (directory or .pt file). "
             "Supports: (1) Distributed checkpoint dir (checkpoint_epoch0_step5000/), "
             "(2) Single .pt file (model.pt)"
    )
    model_group.add_argument(
        "--model_name", 
        type=str, 
        help="HuggingFace model name (e.g., allenai/OLMo-2-1B)"
    )
    
    args = parser.parse_args()
    eval_main(args)



