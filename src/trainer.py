import torch
import torch.nn.functional as F
import numpy as np
import pdb
from transformers import TrainerCallback
from trl import SFTTrainer
import config
from abc import ABC, abstractmethod
import csv 
import sys
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from utils import main_print, is_main_process, AsyncLossLogger
from tqdm.auto import tqdm
from datetime import datetime
import math
from checkpoint import Checkpointer

# ---------------------- Callbacks ----------------------

class LoggingCallback(TrainerCallback):
    def __init__(self, logger, round_num, overall_start_time):
        self.logger = logger
        self.round_num = round_num
        self.overall_start_time = overall_start_time

    def on_prediction_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get("loss", None)
        if loss is not None:
            self.logger.log(
            function="on_prediction_step_end",
            round_num=self.round_num,
            epoch_num=getattr(self, "epoch", None),
            phase="eval",
            role="student",
            step=state.global_step,
            eval_loss=loss.mean().item(),
            alpha=config.alpha,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        control = super().on_log(args, state, control, logs=logs, **kwargs)

        self.logger.log(
            function="on_log",
            round_num=self.round_num,
            epoch_num=getattr(self, "epoch", None),
            phase="train",
            role="student",
            step=state.global_step,
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            grad_norm=logs.get("grad_norm"),
            alpha=config.alpha,
        )

        return control

# ---------------------- Trainer ----------------------

def _gather(x: torch.Tensor) -> torch.Tensor:
    """Safely gather tensors across all processes with proper device and shape handling."""
    # previous implementation:
    # output_tensors = [x.clone() for _ in range(dist.get_world_size())]
    # dist.all_gather(output_tensors, x)
    # return torch.cat(output_tensors, dim=0)
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



# async_loss_logger.py
import os, json, time, threading, queue, tempfile
from typing import Any, Dict, Iterable

def _to_scalar(x: Any):
    # Works for Python numbers, PyTorch tensors, numpy scalars
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().float().item()
            raise ValueError("Tensor must be scalar")
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, (np.generic,)):
            return float(x)
    except Exception:
        pass
    # Python int/float/bool
    if isinstance(x, (int, float, bool)):
        return float(x)
    # Last resort
    return float(x)



class Trainer(ABC):
    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        logger=None,
        checkpointer=None,
        round_num=0,
        overall_start_time=None,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.checkpointer = checkpointer
        self.round_num = round_num
        self.overall_start_time = overall_start_time
        self.wandb_run = wandb_run
        self.processed_id = 0
        self.gad = 0    # gradient accumulated incremented before fwd pass.
        self.gas = config.gradient_accumulation_steps
        self.tr_step = 0    # Need to update when read ckpt
        self.rank = dist.get_rank()
        self.min_eval_loss = 1e12
        self.current_eval_loss = 1e12
        # Early stopping state
        self.best_loss = float('inf')
        self.wait = 0
        self.should_stop = False
        self.epoch = 0
        # Initialize callback for prediction step logging
        self.callback = LoggingCallback(logger, round_num, overall_start_time) if logger else None
        
        # Initialize loss logger only if ID tracking is enabled
        self.loss_logger = None
        if getattr(config, 'enable_id_tracking', True):
            log_path = os.path.join(config.logs_dir, f"loss_log_{dist.get_rank()}.jsonl")
            self.loss_logger = AsyncLossLogger(log_path=log_path, flush_interval_s=1.0, snapshot_interval_s=60.0)


    def prepare_train(self):
        self.model.train()
        self.model.config.use_cache = False  # avoid cache warnings in training
        # Lightweight init log
        if self.logger is not None and self.rank == 0:
            self.logger.log(
                function="prepare_train",
                round_num=self.round_num,
                epoch_num=getattr(self, "epoch", 0),
                phase="init",
                role="student",
                step=self.tr_step,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                alpha=config.alpha,
            )

    def compute_loss(self, batch):
        '''
        reduce loss is sum
        this ensures that we weight all tokens in the dataset equally,
        rather than weighting each overall example equally when
        using high amounts of gradient accumulation.
        this can result in > 5 point improvements in AlpacaEval
        see https://github.com/huggingface/transformers/issues/24725 for
        more discussion and details.
        https://unsloth.ai/blog/gradient

        Return the valid count (num of valid tokens), which is the ratio between mean and sum for each batch.
        '''
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        labels = batch.pop('labels')
        outputs = self.model(**batch)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        ignore_index = getattr(config, "ignore_index", -100)
        valid_mask = shift_labels.ne(ignore_index)
        valid_count = valid_mask.sum()
        return loss, None, None, valid_count

    def step(self, train_batch, eval_dl, epoch):
        # Store the current epoch
        self.epoch = epoch
        
        train_loss = self.train_step(train_batch, epoch)

        test_loss = None
        if self.tr_step % config.logging_steps == 0:
            dist.barrier() 
            test_loss = self.eval_step(eval_dl, epoch)
            dist.barrier() 

        if self.wandb_run is not None and is_main_process():
            log_dict = {
                "train/loss": train_loss,
                "train/epoch": epoch,
                "train/round": self.round_num,
                "train/step": self.tr_step,
                "train/learning_rate": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
            }
            
            if test_loss is not None:
                log_dict.update({
                    "eval/loss": test_loss,
                    "eval/epoch": epoch,
                    "eval/round": self.round_num,
                    "eval/step": self.tr_step,
                })
            
            self.wandb_run.log(log_dict, step=self.tr_step)
        
        if self.tr_step % config.ckpt_save_steps == 0 and self.tr_step > 0 and is_main_process(): self.save_checkpoint(test_loss if test_loss is not None else (train_loss if train_loss is not None else 0.0))
        dist.barrier()

        self.tr_step += 1
        return train_loss, test_loss


    def log_id_loss(self, tr_step_loss, next_token_loss, kl_loss, valid_count, ids):
        if self.loss_logger is not None:
            self.loss_logger.update_and_write_many(ids, tr_step_loss, next_token_loss, kl_loss, valid_count)

    
    def train_step(self, batch, epoch):
        self.model.train()
        # batch["input_ids"] = torch.tensor(batch["input_ids"])
        # batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        # batch["labels"] = torch.tensor(batch["labels"])
            
        batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
        batch["attention_mask"] = batch["attention_mask"].type(torch.LongTensor)
        batch["labels"] = batch["labels"].type(torch.LongTensor)
        
        self.gad += 1
        self.processed_id += 1

        grad_norm = None
        if self.tr_step % 100 == 0:
            dist.barrier() 
            torch.cuda.empty_cache()
            dist.barrier() 
        # Compute loss and backpropagate (supporting grad accumulation)
        if (self.tr_step + 1) % self.gas != self.gas - 1:
            # no need to sync while accumulating gradients
            # if self.tr_step >= 15:
                # breakpoint()
            self.model.set_requires_gradient_sync(False)  # with (grad = False):
            tr_step_loss, next_token_loss, kl_loss, valid_count = self.compute_loss(batch)

            self.log_id_loss(tr_step_loss, next_token_loss, kl_loss, valid_count, batch['id'])
            if isinstance(tr_step_loss, list):
                tr_step_loss = torch.stack(tr_step_loss).sum()
                next_token_loss = torch.stack(next_token_loss).sum()
                kl_loss = torch.stack(kl_loss).sum()
                valid_count = torch.stack(valid_count).sum()
            (tr_step_loss / self.gas).backward()
            self.model.set_requires_gradient_sync(True)
        else:
            # next forward / backward pass will be synced
            self.model.set_requires_gradient_sync(True)
            dist.barrier()
            tr_step_loss, next_token_loss, kl_loss, valid_count = self.compute_loss(batch)

            self.log_id_loss(tr_step_loss, next_token_loss, kl_loss, valid_count, batch['id'])
            if isinstance(tr_step_loss, list):
                tr_step_loss = torch.stack(tr_step_loss).sum()
                next_token_loss = torch.stack(next_token_loss).sum()
                kl_loss = torch.stack(kl_loss).sum()
                valid_count = torch.stack(valid_count).sum()

            # average all lossses (training)
            (tr_step_loss / self.gas).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.max_grad_norm)
            grad_norm = float(grad_norm)
            self.optim.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            self.optim.zero_grad()

        loss_sum = _gather(tr_step_loss.reshape(1)).mean().item()
        nt_sum = _gather((next_token_loss if next_token_loss is not None else torch.tensor(0.0, device=tr_step_loss.device)).reshape(1)).mean().item()
        kl_sum = _gather((kl_loss if kl_loss is not None else torch.tensor(0.0, device=tr_step_loss.device)).reshape(1)).mean().item()
        valid_sum = _gather(valid_count.float().reshape(1)).mean().item()

        # Periodic CSV logging
        if (
            self.logger is not None
            and self.rank == 0
            and (self.tr_step % getattr(config, "logging_steps", 1) == 0)
        ):
            self.logger.log(
                function="train_step",
                round_num=self.round_num,
                epoch_num=getattr(self, "epoch", None),
                phase="train",
                role="student",
                step=self.tr_step,
                train_loss=float(loss_sum) if loss_sum is not None else None,
                train_next_token_loss=float(nt_sum) if nt_sum is not None else None,
                train_kl_loss=float(kl_sum) if kl_sum is not None else None,
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                alpha=config.alpha,
            )

        return loss_sum
    
    def eval_step(self, eval_dl, epoch: int) -> float:
        main_print("Evaluating")
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        nxt_token_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        kl_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        valid_total = torch.tensor(0).to(torch.cuda.current_device())
        
        counter = 0

        for _, batch in enumerate(tqdm(eval_dl,
                                  disable=self.rank != 0,
                                  file=sys.stdout,
                                  mininterval=1.0,
                                  ncols=100)):
            with torch.no_grad():
                # batch["input_ids"] = batch["input_ids"].clone().detach()
                # batch["attention_mask"] = batch["attention_mask"].clone().detach()
                # batch["labels"] = batch["labels"].clone().detach()
                
                batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
                batch["labels"] = batch["labels"].type(torch.LongTensor)
                
                tr_step_loss, next_token_loss, kl_loss, valid_count = self.compute_loss(batch)
                if isinstance(tr_step_loss, list):
                    tr_step_loss = torch.stack(tr_step_loss).sum()
                    next_token_loss = torch.stack(next_token_loss).sum()
                    kl_loss = torch.stack(kl_loss).sum()
                    valid_count = torch.stack(valid_count).sum()
                eval_loss += tr_step_loss
                if next_token_loss is not None:
                    nxt_token_loss += next_token_loss
                if kl_loss is not None:
                    kl_loss += kl_loss
                valid_total += valid_count

                # Call prediction step callback for batch-level logging
                if self.callback and self.rank == 0:
                    # Create a simple state object with global_step
                    class SimpleState:
                        def __init__(self, global_step):
                            self.global_step = global_step
                    
                    state = SimpleState(self.tr_step)
                    # Pass the individual batch loss for logging
                    batch_loss = tr_step_loss / valid_count if valid_count > 0 else tr_step_loss
                    self.callback.on_prediction_step_end(
                        args=None, 
                        state=state, 
                        control=None, 
                        loss=batch_loss
                    )
                # Use only for quick tests
                # if counter == 10:
                #     break
                # counter += 1
        
        # So you don't see eval loss of a few million
        gathered_eval_loss = _gather(eval_loss.reshape(1)).sum().item()
        gathered_nxt_token_loss = _gather(nxt_token_loss.reshape(1)).sum().item()
        gathered_kl_loss = _gather(kl_loss.reshape(1)).sum().item()
        gathered_valid_total = _gather(valid_total.reshape(1)).sum().item()
        # Take the average of eval_loss on both cards, 
        mean_eval_loss = gathered_eval_loss / gathered_valid_total
        mean_nk_loss = gathered_nxt_token_loss / gathered_valid_total if gathered_valid_total > 0 else None
        mean_kl_loss = gathered_kl_loss / gathered_valid_total if gathered_valid_total > 0 else None

        self.metric = mean_eval_loss

        main_print(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")

        if is_main_process():
            if self.logger is not None:
                self.logger.log(
                    function="eval_step",
                    round_num=self.round_num,
                    epoch_num=epoch,
                    phase="eval",
                    role="student",
                    step=self.tr_step,
                    eval_loss=float(mean_eval_loss) if mean_eval_loss is not None else None,
                    eval_next_token_loss=float(mean_nk_loss) if mean_nk_loss is not None else None,
                    eval_kl_loss=float(mean_kl_loss) if mean_kl_loss is not None else None,
                    learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                    alpha=config.alpha,
                )
                try:
                    self.logger.flush()
                except Exception:
                    pass

        self.min_eval_loss = min(mean_eval_loss, self.min_eval_loss)
        self.current_eval_loss = mean_eval_loss

        # -------- Early stopping --------
        if self.best_loss - mean_eval_loss < config.early_stop_min_delta:
            self.best_loss = mean_eval_loss
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= config.early_stop_patience:
            main_print(f"Early stopping triggered: no improvement for {self.wait} evaluations.")
            self.should_stop = True
        
        self.model.train()
        
        # Cleanup evaluation tensors
        del eval_loss, nxt_token_loss, kl_loss, valid_total
        torch.cuda.empty_cache()
        
        return mean_eval_loss

    def save_checkpoint(self, loss: float):
        """Save model+optim via DCP and rotate per-round."""

        round_num = int(getattr(self, "round_num", 0))
        step = int(getattr(self, "global_step", getattr(self, "tr_step", 0)))

        training_state = {
            "epoch": getattr(self, "epoch", 0),
            "round_num": round_num,
            "global_step": step,
            "loss": loss,
            "rng": torch.random.get_rng_state().tolist() if torch.random else None,
        }

        self.checkpointer.save(self.model, self.optim, round_num, step, loss, training_state)

# ---------------------- DistillTrainer ----------------------

class DistillTrainer(Trainer):
    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        ensemble_model,
        logger=None,
        round_num=0,
        checkpointer=None,
        overall_start_time=None,
        wandb_run=None,
    ):
        super().__init__(
            model,
            optim,
            lr_scheduler,
            logger,
            checkpointer,
            round_num,
            overall_start_time,
            wandb_run,
        )
        self.ensemble_model = ensemble_model

    def compute_loss(self, batch):
        '''
        Compute loss with both next token perdiction and kl div with teacher logits.
        '''
        breakpoint()
        # ----------------------------
        # Compute Ensemble Predictions
        # ----------------------------
        ensemble_logits = None
        if self.ensemble_model is not None:
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                ensemble_logits = ensemble_outputs.logits

        # ----------------------------
        # Next token prediction and loss (sum)
        # ----------------------------
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        labels = batch.pop('labels')
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits

        if self.ensemble_model is not None:
            num_models = len(self.ensemble_model.models)
            logits += ensemble_logits
            logits /= num_models + 1
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(shift_logits.device)       

        ignore_index = getattr(config, "ignore_index", -100)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)
        alpha = config.alpha if not config.synthetic_data else 1

        next_token_loss = []   # list of 0-D tensors (sum loss per sequence)
        valid_count = []       # list of 0-D tensors (token count per sequence)
        hybrid_loss = []
        kl_loss = []

        for i in range(shift_logits.size(0)):
            # sum over valid tokens in sequence i; invalid ones (-100) are ignored by loss_fct
            seq_loss = loss_fct(shift_logits[i], shift_labels[i])          # scalar tensor
            next_token_loss.append(seq_loss)

            seq_valid = shift_labels[i].ne(ignore_index).sum()             # scalar tensor
            valid_count.append(seq_valid)

            if (labels != -100).sum == 0:
                print(labels)
            if not config.synthetic_data and alpha > 0:
                kl_loss.append(self.compute_kl_loss(
                    logits[i],
                    mask=labels[i] != -100,
                    logprob_values=[batch['logprob_values'][i]],
                    logprob_indices=[batch['logprob_indices'][i]],
                ))
            else:
                kl_loss.append(torch.tensor(0.0, device=logits.device))
        
        for i in range(shift_logits.size(0)):
            # hybrid_loss.append((1 - alpha) * kl_loss[i] + alpha * next_token_loss[i])
            # You’re computing losses per sequence, then mixing them into hybrid_loss = (1 - alpha) * KL + alpha * CE.
            # Each sequence can have a different number of valid tokens (after masking -100). If you don’t normalize, longer sequences contribute much larger magnitudes (especially KL which was summed), skewing the KL/CE balance and the gradient scale.
            # Normalize KL by valid token count to avoid scale explosion
            # Normalizing inside compute_loss ensures every sequence’s KL (and optionally CE) is on the same per-token scale before mixing with alpha. That keeps the ratio stable and independent of sequence length
            normalized_kl = kl_loss[i] / valid_count[i].clamp(min=1)
            hybrid_loss.append((1 - alpha) * normalized_kl + alpha * next_token_loss[i])

        return hybrid_loss, next_token_loss, kl_loss, valid_count

    def compute_kl_loss(self, student_logits, mask, logprob_values, logprob_indices):
        # -----------------------
        # Compute KL Loss
        # -----------------------
        student_probs = F.log_softmax(student_logits / config.kl_temperature, dim=-1)
        student_masked_probs = student_probs[mask]
        
        device = student_logits.device
        
        teacher_values_list = []
        teacher_indices_list = []
        
        for i in range(len(logprob_values)):
            values = torch.tensor(logprob_values[i], device=device, dtype=torch.float32)
            indices = torch.tensor(logprob_indices[i], device=device, dtype=torch.int64)
            teacher_values_list.append(values)
            teacher_indices_list.append(indices)
        
        teacher_logprob_values = torch.cat(teacher_values_list, dim=0)
        teacher_logprob_indices = torch.cat(teacher_indices_list, dim=0)
        
        student_selected_probs = student_masked_probs.gather(dim=-1, index=teacher_logprob_indices)
        kl_loss = F.kl_div(student_selected_probs, teacher_logprob_values, log_target=True, reduction="none").sum()
        
        del teacher_values_list, teacher_indices_list
        del teacher_logprob_values, teacher_logprob_indices
        del student_selected_probs, student_masked_probs
        
        return kl_loss

