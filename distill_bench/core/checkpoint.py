"""
Single-process checkpointing for distillation training.
"""
import os
import glob
import shutil
import torch

from distill_bench.core.utils import main_print, is_main_process


class SimpleCheckpointer:
    """Lightweight checkpoint manager using torch.save/torch.load."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, lr_scheduler, epoch: int, global_step: int, loss: float):
        """Save model/optimizer/lr_scheduler state to a single file."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{global_step}.pt"
        )
        
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
        }
        torch.save(payload, checkpoint_path)
        
        if is_main_process():
            main_print(f"✓ Saved checkpoint to {checkpoint_path}")
            self._cleanup_old_checkpoints(keep_last=3)
    
    def load(self, model, optimizer, lr_scheduler):
        """Load the latest checkpoint if it exists (returns metadata or None)."""
        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt")),
            key=os.path.getmtime,
        )
        if not checkpoints:
            main_print("No checkpoints found.")
            return None
        
        latest = checkpoints[-1]
        main_print(f"Loading checkpoint from {latest}")
        payload = torch.load(latest, map_location="cpu")
        
        model.load_state_dict(payload["model_state_dict"])
        if optimizer and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if lr_scheduler and payload.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
        
        return {
            "epoch": payload.get("epoch", 0),
            "global_step": payload.get("global_step", 0),
            "loss": payload.get("loss", 0.0),
        }
    
    def _cleanup_old_checkpoints(self, keep_last: int = 7):
        """Remove older checkpoints, keeping the most recent ones."""
        checkpoints = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt")),
            key=os.path.getmtime,
        )
        if len(checkpoints) <= keep_last:
            return
        
        to_remove = checkpoints[:-keep_last]
        for ckpt in to_remove:
            try:
                os.remove(ckpt)
                main_print(f"Removed old checkpoint: {ckpt}")
            except OSError:
                pass
