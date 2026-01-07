"""
DPO trainer for preference-based distillation.

Implements the standard DPO objective:
    L = -log sigma(β * ((π_chosen - π_rejected) - (π_ref_chosen - π_ref_rejected)))
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distill_bench.core.utils import main_print


class DPOTrainer:
    """Lightweight single-GPU DPO trainer."""

    def __init__(
        self,
        policy_model,
        reference_model,
        optimizer,
        lr_scheduler,
        beta: float,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        use_wandb: bool = False,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.beta = beta
        self.gas = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.global_step = 0
        self.min_eval_loss = float("inf")
        self.recent_eval_losses = []

    @staticmethod
    def _sequence_logprob(
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute summed log-prob of response tokens for a batch."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        target_ids = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        seq_len = token_log_probs.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        response_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)) & attention_mask[:, 1:].bool()

        return (token_log_probs * response_mask).sum(dim=1)

    def _dpo_loss(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss and return metrics."""
        chosen_ids = batch["chosen_input_ids"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        chosen_attention = batch["chosen_attention_mask"].to(device)
        rejected_attention = batch["rejected_attention_mask"].to(device)
        chosen_prompt_len = batch["chosen_prompt_len"].to(device)
        rejected_prompt_len = batch["rejected_prompt_len"].to(device)

        policy_chosen_logp = self._sequence_logprob(self.policy_model, chosen_ids, chosen_attention, chosen_prompt_len)
        policy_rejected_logp = self._sequence_logprob(self.policy_model, rejected_ids, rejected_attention, rejected_prompt_len)

        with torch.no_grad():
            ref_chosen_logp = self._sequence_logprob(self.reference_model, chosen_ids, chosen_attention, chosen_prompt_len)
            ref_rejected_logp = self._sequence_logprob(
                self.reference_model, rejected_ids, rejected_attention, rejected_prompt_len
            )

        policy_log_ratio = policy_chosen_logp - policy_rejected_logp
        ref_log_ratio = ref_chosen_logp - ref_rejected_logp

        losses = -F.logsigmoid(self.beta * (policy_log_ratio - ref_log_ratio))
        loss = losses.mean()

        with torch.no_grad():
            margin = (policy_log_ratio - ref_log_ratio).detach()
            accuracy = (margin > 0).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "policy_log_ratio": policy_log_ratio.mean().item(),
            "ref_log_ratio": ref_log_ratio.mean().item(),
            "accuracy": accuracy,
        }
        return loss, metrics

    def _update_early_stop(self, eval_loss: float) -> bool:
        """Track eval losses and decide whether to stop early."""
        self.min_eval_loss = min(self.min_eval_loss, eval_loss)
        self.recent_eval_losses.append(eval_loss)
        if len(self.recent_eval_losses) > 3:
            self.recent_eval_losses.pop(0)

        if len(self.recent_eval_losses) >= 3:
            current = self.recent_eval_losses[-1]
            prev_two = self.recent_eval_losses[-3:-1]
            if current > prev_two[0] and current > prev_two[1]:
                main_print(
                    f"Early stopping triggered: eval loss {current:.4f} > previous two values {prev_two[0]:.4f}, {prev_two[1]:.4f}"
                )
                return True
        return False

    def train_epoch(
        self,
        dataloader: DataLoader,
        device: torch.device,
        eval_dataloader: Optional[DataLoader] = None,
        eval_steps: Optional[int] = None,
        epoch: Optional[int] = None,
        energy_tracker: Optional[object] = None,
    ) -> Tuple[float, int, bool]:
        """Train for one epoch with periodic evaluation and early stopping."""
        self.policy_model.train()
        total_loss = 0.0
        steps = 0
        tokens_processed = 0
        should_stop = False

        for step, batch in enumerate(dataloader):
            loss, metrics = self._dpo_loss(batch, device)
            loss = loss / self.gas
            loss.backward()

            tokens_this_step = (batch["chosen_attention_mask"].sum() + batch["rejected_attention_mask"].sum()).item()
            tokens_processed += tokens_this_step
            if energy_tracker:
                energy_tracker.add_tokens(tokens_this_step)

            update_step = (step + 1) % self.gas == 0
            if update_step:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/accuracy": metrics["accuracy"],
                            "train/policy_log_ratio": metrics["policy_log_ratio"],
                            "train/ref_log_ratio": metrics["ref_log_ratio"],
                            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "train/step": self.global_step,
                        },
                        step=self.global_step,
                    )

                # Periodic evaluation
                do_eval = (
                    eval_dataloader is not None
                    and eval_steps is not None
                    and eval_steps > 0
                    and self.global_step % eval_steps == 0
                )
                if do_eval:
                    eval_loss = self.eval_epoch(eval_dataloader, device)
                    should_stop = self._update_early_stop(eval_loss)

                    if self.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "eval/loss": eval_loss,
                                "eval/min_loss": self.min_eval_loss,
                                "eval/epoch": epoch if epoch is not None else -1,
                            },
                            step=self.global_step,
                        )
                    # Resume training mode after evaluation
                    self.policy_model.train()

                    if should_stop:
                        break

            total_loss += metrics["loss"]
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        return avg_loss, tokens_processed, should_stop

    def eval_epoch(self, dataloader: DataLoader, device: torch.device) -> float:
        """Evaluate the policy model against the reference."""
        self.policy_model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, _ = self._dpo_loss(batch, device)
                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        main_print(f"Eval DPO loss: {avg_loss:.4f}")
        return avg_loss
