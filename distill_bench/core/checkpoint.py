"""
Simplified checkpointing for distillation training.
"""
import os
import glob
import shutil
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from distill_bench.core.utils import main_print, is_main_process

# How it works:
# Each rank saves only its shard to separate files in the directory.
# Creates a directory structure like:
#   checkpoint_epoch0_step100/
#   ├── __0_0.distcp          # Rank 0's shard
#   ├── __1_0.distcp          # Rank 1's shard
#   ├── __2_0.distcp          # Rank 2's shard
#   ├── __3_0.distcp          # Rank 3's shard
#   └── .metadata             # Metadata about the checkpoint

# ==================================================
# AppState Wrapper
# ==================================================
class AppState(Stateful):
    """
    Wrapper for checkpointing the Application State. This object is compliant
    with the Stateful protocol, so DCP will automatically call state_dict/load_state_dict
    as needed in the dcp.save/load APIs.
    
    This wrapper handles calling distributed state dict methods on the model,
    optimizer, and lr_scheduler.
    """
    
    def __init__(self, model, optimizer=None, lr_scheduler=None, epoch=0, global_step=0, loss=0.0):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = epoch
        self.global_step = global_step
        self.loss = loss
    
    def state_dict(self):
        """
        Get state dict for model and optimizer using FSDP-aware methods.
        This automatically manages FSDP FQNs and sets the default state dict type to FSDP.SHARDED_STATE_DICT.
        """
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        
        state = {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss": self.loss,
        }
        
        # Add lr_scheduler state if available
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load state dict into model and optimizer using FSDP-aware methods.
        """
        # Set state dicts on the model and optimizer
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )
        
        # Load metadata
        self.epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict.get("global_step", 0)
        self.loss = state_dict.get("loss", 0.0)
        
        # Load lr_scheduler state if available
        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


# ==================================================
# Checkpointer Class
# ==================================================
class SimpleCheckpointer:
    """Simple checkpoint manager for saving and loading model states."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    # ----------------------------------
    # Save Checkpoint
    # ----------------------------------
    def save(self, model, optimizer, lr_scheduler, epoch, global_step, loss):
        """Save a checkpoint using distributed checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch{epoch}_step{global_step}"
        )
        
        # Create AppState wrapper with all components
        app_state = AppState(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            global_step=global_step,
            loss=loss
        )
        
        # All ranks participate in this operation
        state_dict = {"app": app_state}
        dcp.save(state_dict, checkpoint_id=checkpoint_path)
        
        # Synchronize after save to ensure all ranks finished
        if dist.is_initialized():
            dist.barrier()

        if is_main_process():
            main_print(f"✓ Saved checkpoint to {checkpoint_path}")
            # Keep only the last 3 checkpoints
            self._cleanup_old_checkpoints(keep_last=3)
        
        # Synchronize after cleanup
        if dist.is_initialized():
            dist.barrier()

    
    # ----------------------------------
    # Load Checkpoint
    # ----------------------------------
    def load(self, model, optimizer, lr_scheduler):
        """Load the latest checkpoint if it exists."""
        # Find all checkpoint directories (not .pt files anymore)
        checkpoints = [
            d for d in glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*"))
            if os.path.isdir(d)
        ]
        
        if not checkpoints:
            main_print("No checkpoints found.")
            return None
        
        # Sort by modification time and get the latest
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        main_print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Create AppState wrapper with model, optimizer, and lr_scheduler
        app_state = AppState(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        
        # Use dcp.load for distributed checkpoint loading
        # All ranks participate in this operation
        state_dict = {"app": app_state}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=latest_checkpoint
        )
        
        # Synchronize after load to ensure all ranks have loaded
        if dist.is_initialized():
            dist.barrier()
        
        # Return metadata for training resumption
        return {
            'epoch': app_state.epoch,
            'global_step': app_state.global_step,
            'loss': app_state.loss,
        }
    
    # ----------------------------------
    # Cleanup Old Checkpoints
    # ----------------------------------
    def _cleanup_old_checkpoints(self, keep_last=7):
        """Remove old checkpoints, keeping only the most recent ones."""
        # Find all checkpoint directories
        checkpoints = [
            d for d in glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*"))
            if os.path.isdir(d)
        ]
        
        if len(checkpoints) <= keep_last:
            return
        
        # Identify the oldest checkpoint by creation time (when it was saved)
        oldest_checkpoint = min(checkpoints, key=os.path.getctime)
        
        # Remove the entire checkpoint directory
        shutil.rmtree(oldest_checkpoint)
        main_print(f"Removed oldest checkpoint: {oldest_checkpoint}")
