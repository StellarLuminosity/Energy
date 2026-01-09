import os
import sys
import torch
import glob
import shutil
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.utils import get_dataset, prepare_dataset


def get_teacher_logprobs(config):
    """Load or generate teacher logits dataset."""
    if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
        print("--> Teacher logprobs not found")
        sys.exit(1)

    print("--> Loading Teacher Logits")
    dataset = load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))
    return dataset


def build_teacher_logprobs_dataset(config):
    """Build and save the final `DatasetDict(train=..., test=...)` from chunks."""
    data_dict = {}
    for split in ["train", "test"]:
        split_dirs = sorted(glob.glob(os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}*")))
        chunk_paths = []
        for split_dir in split_dirs:
            chunk_paths += sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1]),
            )
        print(f"--> Concatenating {len(chunk_paths)} chunks for {split}")
        split_dataset = concatenate_datasets([load_from_disk(p) for p in chunk_paths])
        data_dict[split] = split_dataset

    final_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
    DatasetDict(data_dict).save_to_disk(final_path)


def cache_teacher_logprobs(config, energy_tracker):
    """Cache teacher model logprobs for knowledge distillation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_tokens = 0

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    teacher_model.eval()
    teacher_model.config.use_cache = False
    teacher_model.requires_grad_(False)
    print(f"✓ Teacher model loaded on {device}")

    dataset_path = Path(config.dataset_path)
    dataset = get_dataset(dataset_path=dataset_path)
    train_dataloader, test_dataloader = prepare_dataset(dataset["train"], dataset["test"], config)

    print("\n--> Generating Teacher Logits")
    for split, dataloader in [("train", train_dataloader), ("test", test_dataloader)]:
        save_dir = os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}")
        os.makedirs(save_dir, exist_ok=True)

        save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logprob_values": [], "logprob_indices": [], "id": []}
        chunk_id, samples_in_chunk = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[{split}] Caching teacher logprobs"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]  # remain on CPU

                logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits  # [batch, seq_len, vocab_size]
                logprobs = F.log_softmax(logits / config.temperature, dim=-1)
                values, indices = torch.topk(logprobs, k=100, dim=-1)  # [B, T, K]
                values = values.to("cpu")  # BF16
                indices = indices.to(torch.int32).to("cpu")  # INT32

                for b in range(values.size(0)):
                    if not (labels[b] != -100).any():
                        continue

                    save_ds["id"].append(batch["id"][b])
                    save_ds["input_ids"].append(batch["input_ids"][b].tolist())
                    save_ds["attention_mask"].append(batch["attention_mask"][b].tolist())
                    save_ds["labels"].append(batch["labels"][b].tolist())
                    save_ds["logprob_values"].append(values[b].tolist())  # shape [T, K]
                    save_ds["logprob_indices"].append(indices[b].tolist())  # shape [T, K]

                    samples_in_chunk += 1
                    # Count non-masked tokens for energy accounting
                    tok = (batch["labels"][b] != -100).sum().item()
                    total_tokens += tok
                    energy_tracker.add_tokens(tok)

                # Save in chunks
                if samples_in_chunk >= 3200:
                    chunk_path = os.path.join(save_dir, f"chunk_{chunk_id}")
                    if os.path.exists(chunk_path):
                        shutil.rmtree(chunk_path)
                    Dataset.from_dict(save_ds).save_to_disk(chunk_path)

                    save_ds = {k: [] for k in save_ds}  # Reset buffer
                    chunk_id += 1
                    samples_in_chunk = 0

        # Save final chunk if any
        if save_ds["input_ids"]:
            chunk_path = os.path.join(save_dir, f"chunk_{chunk_id}")
            if os.path.exists(chunk_path):
                shutil.rmtree(chunk_path)
            Dataset.from_dict(save_ds).save_to_disk(chunk_path)

    print("✓ Logits saved. Now building full dataset...")
    build_teacher_logprobs_dataset(config)
    print("✓ All teacher logprobs cached.")

    return total_tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache teacher logprobs for KD")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Override output.run_dir for this run")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_dir:
        config.override_run_dir(args.run_dir)

    run_dir = Path(getattr(config, "run_dir", None) or config.get("output.run_dir", None))
    run_dir.mkdir(parents=True, exist_ok=True)
    energy_tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="kd_logit_caching", config=config)
    energy_tracker.start_stage("logit_caching")

    total_tokens = cache_teacher_logprobs(config, energy_tracker=energy_tracker)

    energy_tracker.end_stage(tokens_processed=total_tokens)
    energy_tracker.save_summary()
