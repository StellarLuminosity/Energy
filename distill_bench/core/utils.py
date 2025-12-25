"""
Simplified utilities for distillation training.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import load_from_disk
import random
import numpy as np

from configs.simple_config import config


# ==================================================
# Random Seed Utilities
# ==================================================
def fix_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================
# Distributed Training Utilities
# ==================================================
def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def main_print(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)

# ==================================================
# Custom Collator for Padding
# ==================================================
class CustomPadCollator:
    def __init__(self, max_length, pad_token_id: int = -100, pad_label_id: int = -100, pad_attention_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id
        self.pad_attention_id = pad_attention_id

    def __call__(self, batch):
        batch_padded = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Track other keys
        other_keys = [k for k in batch[0].keys() if k not in batch_padded]

        for item in batch:
            length = len(item["input_ids"])
            pad_len = self.max_length - length

            # Convert to tensor if needed, otherwise clone
            input_ids = item["input_ids"] if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"])
            attention_mask = item["attention_mask"] if isinstance(item["attention_mask"], torch.Tensor) else torch.tensor(item["attention_mask"])
            labels = item["labels"] if isinstance(item["labels"], torch.Tensor) else torch.tensor(item["labels"])

            batch_padded["input_ids"].append(torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            ]))

            batch_padded["attention_mask"].append(torch.cat([
                attention_mask,
                torch.full((pad_len,), self.pad_attention_id, dtype=attention_mask.dtype)
            ]))

            batch_padded["labels"].append(torch.cat([
                labels,
                torch.full((pad_len,), self.pad_label_id, dtype=labels.dtype)
            ]))

        # Stack padded fields
        for k in ["input_ids", "attention_mask", "labels"]:
            batch_padded[k] = torch.stack(batch_padded[k])

        # Add other keys without padding
        for k in other_keys:
            values = [item[k] for item in batch]
            
            # Special handling for logprob_values and logprob_indices (shape [T, K] per sample)
            if k in ["logprob_values", "logprob_indices"]:
                tensor_list = []
                for i, val in enumerate(values):
                    # Convert list to tensor
                    if isinstance(val, list):
                        # Use bfloat16 for logprob_values, int64 for indices (required by gather())
                        val_tensor = torch.tensor(val, dtype=torch.bfloat16 if k == "logprob_values" else torch.int64)
                    else:
                        val_tensor = val if isinstance(val, torch.Tensor) else torch.tensor(val)
                    
                    # Pad to max_length if needed (pad along sequence dimension)
                    seq_len = val_tensor.shape[0]
                    if seq_len < self.max_length:
                        pad_len = self.max_length - seq_len
                        if k == "logprob_values":
                            # Pad with -inf or 0 for logprobs
                            pad_value = torch.full((pad_len, val_tensor.shape[1]), -10000.0, dtype=val_tensor.dtype)
                        else:  # logprob_indices
                            # Pad with 0 for indices
                            pad_value = torch.zeros((pad_len, val_tensor.shape[1]), dtype=val_tensor.dtype)
                        val_tensor = torch.cat([val_tensor, pad_value], dim=0)
                    elif seq_len > self.max_length:
                        # Truncate if longer
                        val_tensor = val_tensor[:self.max_length]
                    
                    tensor_list.append(val_tensor)
                
                batch_padded[k] = torch.stack(tensor_list)
            else:
                try:
                    batch_padded[k] = torch.stack(values)
                except:
                    batch_padded[k] = values  # Leave as list if not stackable

        return batch_padded


# ==================================================
# Dataset Loading
# ==================================================
def get_dataset():
    """Load dataset."""
    return datasets.load_from_disk(config.dataset_path)


def prepare_dataset(train_ds, eval_ds):
    """Prepare DataLoaders with DistributedSampler."""
    custom_collator = CustomPadCollator(1024, pad_token_id=100277) # pad_token_id for OLmo2

    if dist.is_initialized():
        # Distributed training mode
        train_sampler = DistributedSampler(
            train_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=True,
            seed=config.seed,
            drop_last=True, 
        )
        test_sampler = DistributedSampler(
            eval_ds,
            dist.get_world_size(),
            dist.get_rank(),
            shuffle=False,
            drop_last=True, 
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=custom_collator,
            num_workers=0,
            persistent_workers=False,
            drop_last=True
        )   
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=config.eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            collate_fn=custom_collator,
            num_workers=0,
            persistent_workers=False,
            drop_last=True  
        )

        dist.barrier()
    else:
        # Standalone evaluation mode
        train_dataloader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=custom_collator,
            num_workers=0,
            drop_last=True
        )   
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=custom_collator,
            num_workers=0,
            drop_last=False  # Don't drop last batch in eval
        )

    return train_dataloader, eval_dataloader

