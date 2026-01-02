import torch
import wandb
import torch.nn.functional as F
import torch.distributed as dist
from tqdm.auto import tqdm
import pdb
import sys

from distill_bench.core.utils import is_main_process, main_print

# ==================================================
# Helper Functions
# ==================================================
def _gather(x: torch.Tensor) -> torch.Tensor:
    """Safely gather tensors across all processes with proper device and shape handling."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    
    device = torch.cuda.current_device()
    x = x.to(device)
    
    original_shape = x.shape
    x_flat = x.flatten()
    output_tensors = [torch.zeros_like(x_flat) for _ in range(dist.get_world_size())]
    
    try:
        dist.all_gather(output_tensors, x_flat)
        output_tensors = [t.reshape(original_shape) for t in output_tensors]
        return torch.cat(output_tensors, dim=0)
    except Exception as e:
        print(f"Warning: _gather failed with error {e}, returning original tensor")
        return x


# ==================================================
# Trainer Class
# ==================================================
class Trainer:
    """Trainer for teacher-student distillation using cached teacher logits."""
    
    def __init__(
        self,
        student_model,
        optimizer,
        lr_scheduler,
        config,
        checkpointer=None,
    ):
        self.student_model = student_model
        self.model = student_model  # For compatibility with checkpoint saving
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.checkpointer = checkpointer
        self.global_step = 0
        self.epoch = 0
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.min_eval_loss = float('inf')
        self.current_eval_loss = 0.0
        
        # Early stopping: track last 2 eval losses
        self.recent_eval_losses = []
        
        # Wandb logging
        self.use_wandb = WANDB_AVAILABLE and is_main_process()
        
        # Gradient accumulation
        self.gad = 0  # gradient accumulated steps counter
        self.gas = getattr(config, 'gradient_accumulation_steps', 1)
    
    # ----------------------------------
    # Loss Computation
    # ----------------------------------
    def compute_loss(self, batch):
        """
        Compute hybrid distillation loss combining:
        1. Cross-entropy loss on true labels
        2. KL divergence between student and cached teacher distributions
        
        Returns: (total_loss, valid_count, ce_loss, kl_loss)
        All losses use reduction='sum' for proper gradient accumulation.
        """
        # Move batch to GPU
        device = torch.cuda.current_device()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # ------ Student Forward Pass ------
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits
        del student_outputs
        
        # ------ Prepare Logits for Next-Token Prediction ------
        # Shift for next-token prediction
        vocab_size = student_logits.size(-1)
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Free unshifted logits to save memory
        del student_logits
        
        # Flatten
        shift_student_logits = shift_student_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_student_logits.device)
        
        # Create mask for valid positions (not padding)
        ignore_index = -100
        mask = shift_labels != ignore_index
        valid_count = mask.sum()
        
        # ------ Cross-Entropy Loss ------
        ce_loss = F.cross_entropy(
            shift_student_logits, 
            shift_labels, 
            ignore_index=ignore_index, 
            reduction='sum'
        )
        
        # ------ KL Divergence Loss (Sparse) ------
        kl_loss = torch.tensor(0.0, device=device)
        
        if mask.sum() > 0 and 'logprob_values' in batch and 'logprob_indices' in batch:
            # Get cached teacher logits (sparse)
            teacher_logprob_values = batch['logprob_values'].to(device)  # [B, T, K]
            teacher_logprob_indices = batch['logprob_indices'].to(device)  # [B, T, K]
            
            # Shift teacher logits to align with next-token prediction
            shift_teacher_logprob_values = teacher_logprob_values[..., :-1, :].contiguous()
            shift_teacher_logprob_indices = teacher_logprob_indices[..., :-1, :].contiguous()
            
            # Flatten: [B, T-1, K] -> [(B*(T-1)), K]
            shift_teacher_logprob_values = shift_teacher_logprob_values.view(-1, shift_teacher_logprob_values.size(-1))
            shift_teacher_logprob_indices = shift_teacher_logprob_indices.view(-1, shift_teacher_logprob_indices.size(-1))
            
            # Only compute KL on valid (non-masked) positions
            masked_student_logits = shift_student_logits[mask]  # [valid_tokens, V]
            masked_teacher_values = shift_teacher_logprob_values[mask]  # [valid_tokens, K]
            masked_teacher_indices = shift_teacher_logprob_indices[mask]  # [valid_tokens, K]
            
            # Compute student log probabilities with temperature
            student_log_probs = F.log_softmax(masked_student_logits / self.config.kl_temperature, dim=-1)
            
            # Gather student log probs at teacher's cached indices
            student_selected_log_probs = student_log_probs.gather(dim=-1, index=masked_teacher_indices)
            
            # Compute KL divergence (both distributions use same temperature)
            # KL(teacher || student) = sum(teacher_prob * (teacher_logprob - student_logprob))
            kl_loss = F.kl_div(
                student_selected_log_probs, 
                masked_teacher_values, 
                log_target=True, 
                reduction='sum'
            )
            
            kl_loss = kl_loss * (self.config.kl_temperature ** 2)
        
        # ------ Combine Losses ------
        total_loss = self.config.alpha * ce_loss + (1 - self.config.alpha) * kl_loss
        
        return total_loss, valid_count, ce_loss, kl_loss
    
    # ----------------------------------
    # Training Step
    # ----------------------------------
    def train_step(self, batch):
        """Single training step with gradient accumulation support."""
        self.model.train()
        
        # ------ Prepare Batch ------
        # Ensure batch tensors are LongTensor
        batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
        batch["attention_mask"] = batch["attention_mask"].type(torch.LongTensor)
        batch["labels"] = batch["labels"].type(torch.LongTensor)
        
        self.gad += 1
        
        # ------ Initialization and Cleanup ------
        # First batch warning
        if self.global_step == 0 and self.rank == 0:
            main_print("First batch (FSDP initialization + CUDA compilation)...")

        # Periodic memory cleanup
        if self.global_step % 100 == 0:
            dist.barrier()
            torch.cuda.empty_cache()
            dist.barrier()
        
        # ------ Compute Loss ------
        tr_step_loss, valid_count, ce_loss, kl_loss = self.compute_loss(batch)
        
        # ------ Gradient Accumulation ------
        is_accumulating = (self.global_step + 1) % self.gas != 0
        grad_norm = None
        
        if is_accumulating:
            # No need to sync while accumulating gradients
            if hasattr(self.model, 'set_requires_gradient_sync'):
                self.model.set_requires_gradient_sync(False)
            
            # Normalize and backward
            normalized_loss = tr_step_loss / self.gas
            normalized_loss.backward()
            
            if hasattr(self.model, 'set_requires_gradient_sync'):
                self.model.set_requires_gradient_sync(True)
        else:
            # Final accumulation step - sync gradients
            if hasattr(self.model, 'set_requires_gradient_sync'):
                self.model.set_requires_gradient_sync(True)
            
            # Normalize and backward
            normalized_loss = tr_step_loss / self.gas
            normalized_loss.backward()
            
            # Gradient clipping and optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        # ------ Gather Metrics Across GPUs ------
        loss_sum = _gather(tr_step_loss.reshape(1)).sum().item()
        valid_sum = _gather(valid_count.float().reshape(1)).sum().item()
        ce_loss_sum = _gather(ce_loss.reshape(1)).sum().item() if ce_loss is not None else 0.0
        kl_loss_sum = _gather(kl_loss.reshape(1)).sum().item() if kl_loss is not None else 0.0
        
        # Compute average loss per token
        avg_loss = loss_sum / valid_sum if valid_sum > 0 else 0.0
        avg_ce_loss = ce_loss_sum / valid_sum if valid_sum > 0 else 0.0
        avg_kl_loss = kl_loss_sum / valid_sum if valid_sum > 0 else 0.0
        
        # ------ Logging ------
        # Log to wandb (only on main process and when not accumulating)
        if self.use_wandb and not is_accumulating:
            log_dict = {
                "train/loss": avg_loss,
                "train/ce_loss": avg_ce_loss,
                "train/kl_loss": avg_kl_loss,
                "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                "train/step": self.global_step,
            }
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            
            wandb.log(log_dict, step=self.global_step)
        
        self.global_step += 1
        
        return avg_loss
    
    # ----------------------------------
    # Evaluation Step
    # ----------------------------------
    def eval_step(self, eval_dataloader):
        """Evaluate the model."""
        main_print("Evaluating...")
        self.model.eval()
        
        # ------ Initialize Accumulators ------
        total_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        total_ce_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        total_kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
        total_valid_tokens = torch.tensor(0, device=torch.cuda.current_device())
        
        # ------ Evaluation Loop ------
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, 
                            desc="Evaluating", 
                            disable=self.rank != 0,
                            file=sys.stdout):
                
                # Ensure batch tensors are LongTensor
                batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
                batch["attention_mask"] = batch["attention_mask"].type(torch.LongTensor)
                batch["labels"] = batch["labels"].type(torch.LongTensor)
                
                # Compute loss
                tr_step_loss, valid_count, ce_loss, kl_loss = self.compute_loss(batch)
                
                # Accumulate metrics
                total_loss += tr_step_loss
                total_valid_tokens += valid_count
                if ce_loss is not None:
                    total_ce_loss += ce_loss
                if kl_loss is not None:
                    total_kl_loss += kl_loss
        
        # ------ Gather Metrics Across GPUs ------
        gathered_loss = _gather(total_loss.reshape(1)).sum().item()
        gathered_tokens = _gather(total_valid_tokens.reshape(1)).sum().item()
        gathered_ce_loss = _gather(total_ce_loss.reshape(1)).sum().item()
        gathered_kl_loss = _gather(total_kl_loss.reshape(1)).sum().item()
        
        # Compute average loss per token
        avg_loss = gathered_loss / gathered_tokens if gathered_tokens > 0 else 0.0
        avg_ce_loss = gathered_ce_loss / gathered_tokens if gathered_tokens > 0 else 0.0
        avg_kl_loss = gathered_kl_loss / gathered_tokens if gathered_tokens > 0 else 0.0
        
        main_print(f"Step: {self.global_step}, Eval Loss: {avg_loss:.4f}")
        
        # ------ Update Tracking ------
        self.min_eval_loss = min(avg_loss, self.min_eval_loss)
        self.current_eval_loss = avg_loss
        
        # ------ Early Stopping Check ------
        # Track last 3 eval losses to detect overfitting
        self.recent_eval_losses.append(avg_loss)
        if len(self.recent_eval_losses) > 3:
            self.recent_eval_losses.pop(0)  # Keep only last 3
        
        # Check if current loss is worse than both of the previous 2 (overfitting detected)
        should_stop = False
        if len(self.recent_eval_losses) >= 3:
            # Current loss is the last one, previous 2 are the ones before
            current = self.recent_eval_losses[-1]
            prev_two = self.recent_eval_losses[-3:-1]  # Get the 2 values before current
            if current > prev_two[0] and current > prev_two[1]:
                should_stop = True
                main_print(f"Early stopping triggered: eval loss {current:.4f} > previous two values {prev_two[0]:.4f}, {prev_two[1]:.4f}")
        
        # ------ Logging ------
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "eval/loss": avg_loss,
                "eval/ce_loss": avg_ce_loss,
                "eval/kl_loss": avg_kl_loss,
                "eval/min_loss": self.min_eval_loss,
                "eval/step": self.global_step,
                "eval/epoch": self.epoch,
            }, step=self.global_step)
        
        # ------ Cleanup ------
        self.model.train()
        
        # Free memory
        del total_loss, total_ce_loss, total_kl_loss, total_valid_tokens
        torch.cuda.empty_cache()
        
        return avg_loss, should_stop
    
    # ----------------------------------
    # Checkpoint Saving
    # ----------------------------------
    def save_checkpoint(self, loss: float = None):
        """Save checkpoint via checkpointer."""
        
        if self.checkpointer is not None:
            self.checkpointer.save(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                epoch=self.epoch,
                global_step=self.global_step,
                loss=loss if loss is not None else 0.0
            )
