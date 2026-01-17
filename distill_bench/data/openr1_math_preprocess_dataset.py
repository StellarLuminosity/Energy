import os
import re
import random
import shutil
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

THINK_START = "<think>"
THINK_END = "</think>"


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks from a string, if present."""
    if not isinstance(text, str):
        return text

    # Simple loop so we handle multiple think blocks if they ever appear
    while True:
        start = text.find(THINK_START)
        if start == -1:
            break
        end = text.find(THINK_END, start + len(THINK_START))
        if end == -1:
            # No closing tag; be conservative and just break
            break
        text = text[:start] + text[end + len(THINK_END) :]
    return text.strip()


def create_response_labels(sample, tokenizer):
    """Mask everything but the assistant response."""
    if not isinstance(sample["input_ids"], torch.Tensor):
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)
    if not isinstance(sample["attention_mask"], torch.Tensor):
        sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)

    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    labels = input_ids.clone()
    labels.fill_(-100)

    response_ids = tokenizer("<|assistant|>\n", add_special_tokens=False)["input_ids"]
    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            start_pos = i + len(response_ids)
            break

    last_valid = attn.nonzero(as_tuple=True)[0].max().item()
    end_pos = last_valid + 1

    if start_pos >= 0:
        labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    labels = labels.masked_fill(attn == 0, -100)
    return labels


def contains_complete_response_template(sample, tokenizer):
    response_template_ids = tokenizer("<|assistant|>\n", add_special_tokens=False)["input_ids"]
    input_ids = sample["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    for i in range(len(input_ids) - len(response_template_ids) + 1):
        if input_ids[i : i + len(response_template_ids)] == response_template_ids:
            return True
    return False


def build_messages(sample):
    """Create chat messages; prefer existing messages if present."""
    if "messages" in sample and sample["messages"]:
        # OpenR1 already has messages; strip <think> blocks from assistant content
        msgs = []
        for m in sample["messages"]:
            m = dict(m)  # avoid mutating underlying HF object
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                m["content"] = strip_think_block(m["content"])
            msgs.append(m)
        return msgs

    # Fallback: build messages from problem / solution / answer
    problem = sample.get("problem") or sample.get("content") or ""
    answer = sample.get("answer")
    solution = sample.get("solution")

    assistant_content_parts = []
    if solution:
        assistant_content_parts.append(solution)
    if answer:
        assistant_content_parts.append(f"Answer: {answer}")
    assistant_content = "\n".join(assistant_content_parts) if assistant_content_parts else ""

    # Strip think if it was embedded in solution text
    assistant_content = strip_think_block(assistant_content) if assistant_content else ""

    messages = [{"role": "user", "content": problem}]
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


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


def main(config, energy_tracker: EnergyTracker = None, stage_name: str = "openr1_math_preprocess_dataset"):
    started_here = False
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    dataset_name = config.dataset_name
    dataset_subset = getattr(config, "dataset_subset", None)
    dataset_split = getattr(config, "dataset_split", "default")

    print("\n=== LOADING DATASET ===")
    print(f"Dataset: {dataset_name} | Subset: {dataset_subset} | Split: {dataset_split}")

    if dataset_split:
        raw_dataset = datasets.load_dataset(dataset_name, dataset_subset, split=dataset_split)
    else:
        raw_dataset = datasets.load_dataset(dataset_name, dataset_subset)

    if isinstance(raw_dataset, datasets.DatasetDict):
        if dataset_split in raw_dataset:
            dataset = raw_dataset[dataset_split]
        elif "train" in raw_dataset:
            dataset = raw_dataset["train"]
            print(f"Split '{dataset_split}' not found; defaulting to 'train'.")
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
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
        sample["messages"] = messages
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
        sample["input_ids"] = tokenized["input_ids"].squeeze(0)
        sample["attention_mask"] = tokenized["attention_mask"].squeeze(0)
        sample["labels"] = create_response_labels(sample, tokenizer)
        return sample

    dataset = dataset.map(tokenize_and_label, desc="Tokenizing", num_proc=8)

    before_filter = len(dataset)
    dataset = dataset.filter(lambda x: contains_complete_response_template(x, tokenizer), desc="Filtering", num_proc=8)
    after_filter = len(dataset)
    print(f"Kept {after_filter}/{before_filter} examples after assistant-marker filtering")
    if after_filter == 0:
        raise ValueError("No examples contain the assistant marker; check chat template and dataset format.")

    split_dataset = dataset.train_test_split(test_size=0.05, seed=config.seed)
    split_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    save_path = config.dataset_path
    print(f"\n=== SAVING DATASET TO {save_path} ===")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    allowed_cols = {"input_ids", "attention_mask", "labels"}
    cleaned = split_dataset.remove_columns([col for col in split_dataset["train"].column_names if col not in allowed_cols])
    cleaned.save_to_disk(save_path)
    print(f"Saved train={len(cleaned['train'])}, test={len(cleaned['test'])}")

    inspect_dataset(cleaned, tokenizer)

    if energy_tracker and started_here:
        total_examples = len(cleaned["train"]) + len(cleaned["test"])
        energy_tracker.add_tokens(total_examples)
        energy_tracker.end_stage(tokens_processed=total_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OpenR1 Math dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="openr1_math_preprocess_dataset", config=cfg)

    main(cfg, energy_tracker=tracker, stage_name="openr1_math_preprocess_dataset")
    tracker.save_summary()
