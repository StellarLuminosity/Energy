"""
Inspect the preprocessed Codeforces dataset saved by codeforces_preprocess_dataset.py.
Loads from disk, reports metadata, and decodes sample records.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable

import datasets
from transformers import AutoTokenizer

from distill_bench.core.config_loader import load_config


def _to_int_list(values: Any) -> list[int]:
    """Convert tensors/arrays/sequences to a plain Python list of ints."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _decode_tokens(input_ids: Any, attention_mask: Any, tokenizer) -> str:
    """Decode tokens, trimming padding based on the attention mask if available."""
    token_ids = _to_int_list(input_ids)
    attn = _to_int_list(attention_mask) if attention_mask is not None else None
    if attn:
        token_ids = [tid for tid, mask in zip(token_ids, attn) if mask == 1]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _format_seq(name: str, seq: list[int], head: int = 20, tail: int = 10) -> str:
    """Render a head/tail preview of a token sequence."""
    n = len(seq)
    if n <= head + tail:
        preview = seq
    else:
        preview = seq[:head] + ["..."] + seq[-tail:]
    return f"{name} (len={n}): {preview}"


def _describe_dataset(ds: datasets.Dataset | datasets.DatasetDict) -> None:
    print(f"Dataset type: {type(ds)}")
    if isinstance(ds, datasets.DatasetDict):
        keys = list(ds.keys())
        print(f"Dataset keys (splits): {keys}")
        for split_name, split in ds.items():
            print(f"{split_name}: size={len(split)}, columns={split.column_names}")
    else:
        print(f"Dataset size: {len(ds)}, columns={ds.column_names}")


def _count_non_one_attention(split: datasets.Dataset) -> int:
    """Count rows where attention_mask contains any value other than 1."""
    count = 0
    for attn in split["attention_mask"]:
        seq = _to_int_list(attn)
        if any(val != 1 for val in seq):
            count += 1
    return count


def _print_samples(
    split: datasets.Dataset,
    split_name: str,
    tokenizer,
    num_examples: int,
) -> None:
    total = len(split)
    print(f"\n--- {split_name} samples (showing up to {num_examples} of {total}) ---")
    if total == 0:
        print("No examples in this split.")
        return

    for idx in range(min(num_examples, total)):
        rec = split[idx]
        input_ids = _to_int_list(rec["input_ids"])
        attn = _to_int_list(rec.get("attention_mask"))
        labels = _to_int_list(rec.get("labels"))

        # First, show the tokenized fields.
        print(f"\n[{split_name} idx={idx}] tokenized fields:")
        print(f"  {_format_seq('input_ids', input_ids)}")
        print(f"  {_format_seq('attention_mask', attn)}")
        print(f"  {_format_seq('labels', labels)}")

        # Then, show detokenized text.
        decoded_input = _decode_tokens(input_ids, attn, tokenizer)
        label_tokens = [lid for lid in labels if lid != -100]
        decoded_labels = tokenizer.decode(label_tokens, skip_special_tokens=True) if label_tokens else ""
        print(f"[{split_name} idx={idx}] decoded input: {decoded_input[:500]}")
        print(f"[{split_name} idx={idx}] decoded labels: {decoded_labels[:500]}")


def _to_python_format(ds: datasets.Dataset | datasets.DatasetDict):
    """Ensure we get Python lists instead of framework tensors."""
    if isinstance(ds, datasets.DatasetDict):
        return datasets.DatasetDict({name: split.with_format("python") for name, split in ds.items()})
    return ds.with_format("python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect preprocessed Codeforces dataset on disk")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/sft_32b_to_1b.yaml",
        help="Experiment config to infer dataset_path and tokenizer_name",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for the preprocessed dataset path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional override for the tokenizer used to decode samples",
    )
    parser.add_argument("--num-examples", type=int, default=3, help="Number of samples to show per split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = args.dataset_path or cfg.dataset_path
    tokenizer_name = args.tokenizer or cfg.tokenizer_name

    if not dataset_path:
        raise ValueError("Dataset path is missing. Provide --dataset-path or set dataset_path in the config.")

    print(f"Loading preprocessed dataset from: {dataset_path}")
    ds = datasets.load_from_disk(dataset_path)
    _describe_dataset(ds)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds_python = _to_python_format(ds)

    if isinstance(ds_python, datasets.DatasetDict):
        for split_name in ["train", "test"]:
            if split_name in ds_python:
                split = ds_python[split_name]
                non_one = _count_non_one_attention(split)
                print(f"{split_name}: {non_one} examples have attention_mask values other than 1 (of {len(split)})")
                _print_samples(split, split_name, tokenizer, args.num_examples)
    else:
        non_one = _count_non_one_attention(ds_python)
        print(f"dataset: {non_one} examples have attention_mask values other than 1 (of {len(ds_python)})")
        _print_samples(ds_python, "dataset", tokenizer, args.num_examples)


if __name__ == "__main__":
    main()
