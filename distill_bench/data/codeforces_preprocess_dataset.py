import os
import random
import shutil
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker


def build_messages(sample):
    """Construct a minimal chat-style message list from a Codeforces prompt."""
    prompt_text = sample.get("prompt", "")
    return [{"role": "user", "content": prompt_text}]


def find_response_start(input_ids, response_template_ids):
    """Locate the index immediately after the assistant template tokens."""
    for i in range(len(input_ids) - len(response_template_ids) + 1):
        if input_ids[i : i + len(response_template_ids)] == response_template_ids:
            return i + len(response_template_ids)
    return -1


def inspect_dataset(split_dataset, tokenizer, num_examples: int = 3):
    """Lightweight inspection of the processed dataset."""
    if not isinstance(split_dataset, datasets.DatasetDict):
        print("Dataset is not split; skipping inspection.")
        return

    train_len = len(split_dataset["train"])
    test_len = len(split_dataset["test"])
    print(f"\n=== DATASET INSPECTION ===")
    print(f"Columns: {split_dataset['train'].column_names}")
    print(f"Train size: {train_len}, Test size: {test_len}")

    for split_name in ["train", "test"]:
        ds = split_dataset[split_name]
        if len(ds) == 0:
            continue
        print(f"\nSample decoded examples from '{split_name}':")
        for i in range(min(num_examples, len(ds))):
            rec = ds[i]
            input_ids = rec["input_ids"]
            attn = rec.get("attention_mask", None)
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if attn is not None and isinstance(attn, torch.Tensor):
                attn = attn.tolist()
            trimmed_ids = [tid for tid, mask in zip(input_ids, attn)] if attn else input_ids
            decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            print(f"Example {i} token ids (truncated 30): {trimmed_ids[:30]}")
            print(f"Example {i} text: {decoded[:500]}")


def main(config, energy_tracker: EnergyTracker = None, stage_name: str = "codeforces_preprocess_dataset"):
    started_here = False
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    response_template_ids = tokenizer("<|assistant|>\n", add_special_tokens=False)["input_ids"]

    dataset_name = config.dataset_name
    dataset_subset = getattr(config, "dataset_subset", None)
    dataset_split = getattr(config, "dataset_split", None)

    print("\n=== LOADING DATASET ===")
    print(f"Dataset: {dataset_name} | Subset: {dataset_subset} | Split: {dataset_split or 'train'}")

    if dataset_split:
        raw_dataset = datasets.load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        raw_dataset = datasets.load_dataset(dataset_name, dataset_subset)

    if isinstance(raw_dataset, datasets.DatasetDict):
        if dataset_split and dataset_split in raw_dataset:
            dataset = raw_dataset[dataset_split]
        elif "train" in raw_dataset:
            dataset = raw_dataset["train"]
        else:
            first_split = list(raw_dataset.keys())[0]
            dataset = raw_dataset[first_split]
            print(f"Split '{dataset_split}' not found; defaulting to '{first_split}'.")
    else:
        dataset = raw_dataset

    print(f"Original dataset size: {len(dataset)}")
    dataset = dataset.shuffle(seed=config.seed)
    sample_size = min(8000, len(dataset))
    dataset = dataset.select(range(sample_size))
    print(f"After subsampling: {len(dataset)} examples")

    def format_chat_data(sample):
        messages = build_messages(sample)
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sample["chat_text"] = chat_text
        return sample

    dataset = dataset.map(format_chat_data, desc="Formatting chat", num_proc=8)

    def tokenize_and_label(sample):
        tokenized = tokenizer(
            sample["chat_text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_sequence_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        response_start = find_response_start(input_ids.tolist(), response_template_ids)
        labels = torch.full_like(input_ids, fill_value=-100)
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels
        sample["response_start"] = response_start
        return sample

    dataset = dataset.map(tokenize_and_label, desc="Tokenizing", num_proc=8)

    def has_response_start(sample):
        return sample["response_start"] >= 0

    dataset = dataset.filter(has_response_start, desc="Filtering missing assistant marker")

    split_dataset = dataset.train_test_split(test_size=0.05, seed=config.seed)

    split_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    save_path = config.dataset_path
    print(f"\n=== SAVING DATASET TO {save_path} ===")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    cleaned = split_dataset.remove_columns([col for col in split_dataset["train"].column_names if col not in {"input_ids", "attention_mask", "labels", "id", "response_start"}])
    cleaned.save_to_disk(save_path)
    print(f"Saved train={len(cleaned['train'])}, test={len(cleaned['test'])}")

    inspect_dataset(cleaned, tokenizer)

    if energy_tracker and started_here:
        total_examples = len(cleaned["train"]) + len(cleaned["test"])
        energy_tracker.add_tokens(total_examples)
        energy_tracker.end_stage(tokens_processed=total_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Codeforces dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="codeforces_preprocess_dataset", config=cfg)

    main(cfg, energy_tracker=tracker, stage_name="codeforces_preprocess_dataset")
    tracker.save_summary()
