"""
Teacher-based Synthetic Data Generation for Data Distillation.

Generates synthetic training data by having a teacher model produce responses
to prompts from a base dataset. Tracks energy consumption during generation.
"""

import os
import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Optional
from torch.nn.utils.rnn import pad_sequence

from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.config_loader import Config, load_config


def generate_synthetic_dataset(
    config: Config,
    energy_tracker: Optional[EnergyTracker] = None,
    stage_name: str = "synthetic dataset teacher generation (sft)",
) -> datasets.DatasetDict:
    """
    Generate synthetic dataset using teacher model.
    Args:
        config: Configuration object
        energy_tracker: Optional energy tracker for measuring generation

    Returns:
        DatasetDict with synthetic training data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_config = config.get("synthetic_data.generation", {})
    temperature = gen_config.get("temperature", 0.7)
    top_p = gen_config.get("top_p", 0.9)
    max_new_tokens = gen_config.get("max_new_tokens", 512)
    decoding_strategy = gen_config.get("decoding_strategy", "sampling")
    generation_batch_size = gen_config.get("batch_size", 1)

    max_seq_len = getattr(config, "max_sequence_length", None) or config.get("data.max_sequence_length", 2048)
    synthetic_path = config.get("synthetic_data.synthetic_dataset_path")
    dataset_path = config.dataset_path or config.get("data.dataset_path")
    dataset_name = config.dataset_name or config.get("data.dataset_name")

    if not dataset_path:
        raise ValueError("dataset_path is not set. Run tulu_preprocess_dataset.py first.")

    print(f"Using dataset_choice='{dataset_name}' from preprocessed dataset at: {dataset_path}")

    # Start energy tracking for teacher generation (single stage)
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)

    # Load tokenizer and teacher model
    print(f"Loading teacher model: {config.teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    teacher_model.eval()

    print(f"Loading preprocessed prompt dataset from: {dataset_path}")
    prompt_dataset = datasets.load_from_disk(dataset_path)
    if isinstance(prompt_dataset, datasets.DatasetDict):
        prompt_dataset = prompt_dataset["train"]

    max_gen_examples = getattr(config, "max_gen_examples", None)

    print(f"Generating {len(max_gen_examples)} synthetic examples...")

    # Storage for synthetic data
    synthetic_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    total_tokens_generated = 0
    successful_generations = 0
    filtering_config = config.get("synthetic_data.filtering", {})
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    base_generation_kwargs = {
        "pad_token_id": pad_token_id,
        "do_sample": decoding_strategy == "sampling",
    }
    if decoding_strategy == "sampling":
        base_generation_kwargs["temperature"] = temperature
        base_generation_kwargs["top_p"] = top_p

    batch_prompts = []

    def flush_batch():
        nonlocal batch_prompts, total_tokens_generated, successful_generations
        if not batch_prompts:
            return

        prompt_lengths = [p["prompt_ids"].shape[0] for p in batch_prompts]
        max_prompt_length = max(prompt_lengths)
        max_new_tokens_for_batch = min(max_new_tokens, max_seq_len - max_prompt_length)
        if max_new_tokens_for_batch <= 0:
            batch_prompts = []
            return

        batch_input_ids = pad_sequence(
            [p["prompt_ids"] for p in batch_prompts],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        batch_attention_mask = pad_sequence(
            [p["prompt_attention_mask"] for p in batch_prompts],
            batch_first=True,
            padding_value=0,
        ).to(device)

        batch_outputs = teacher_model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens_for_batch,
            **base_generation_kwargs,
        ).cpu()

        for prompt_info, output, prompt_length in zip(batch_prompts, batch_outputs, prompt_lengths):
            generated_tokens = output[prompt_length:]

            synthetic_labels = torch.full_like(output, fill_value=-100)
            synthetic_labels[prompt_length:] = output[prompt_length:]

            if filtering_config.get("enabled", True):
                min_length = filtering_config.get("min_length", 10)
                max_length = filtering_config.get("max_length", max_seq_len)
                response_length = len(generated_tokens)
                total_length = len(output)
                if response_length < min_length:
                    print(f"Response length shorter than min length - skipping idx {prompt_info['idx']}")
                    continue
                if total_length > max_length:
                    print(f"Total length (prompt + response) is greater than max length - skipping idx {prompt_info['idx']}")
                    continue

            output_attention_mask = torch.ones_like(output)

            synthetic_data["input_ids"].append(output.tolist())
            synthetic_data["attention_mask"].append(output_attention_mask.tolist())
            synthetic_data["labels"].append(synthetic_labels.tolist())

            total_tokens_generated += len(generated_tokens)
            successful_generations += 1

        batch_prompts = []
        if device.type == "cuda" and successful_generations and successful_generations % 500 == 0:
            torch.cuda.empty_cache()

    # Generate responses
    with torch.inference_mode():
        processed_examples = 0
        for idx, example in enumerate(tqdm(prompt_dataset, desc="Generating synthetic data")):
            if max_gen_examples is not None and processed_examples >= max_gen_examples:
                print(f"[EARLY STOP] Reached synthetic generation limit ({max_gen_examples})")
                break
            processed_examples += 1
            try:
                input_ids = torch.tensor(example["input_ids"])
                attention_mask = torch.tensor(example["attention_mask"])
                existing_labels = torch.tensor(example["labels"])

                response_tokens = (existing_labels != -100).nonzero(as_tuple=True)[0]
                if len(response_tokens) == 0:
                    continue

                response_start = response_tokens[0].item()
                if response_start == 0:
                    continue

                prompt_ids = input_ids[:response_start]
                prompt_attention_mask = attention_mask[:response_start]
                prompt_length = prompt_ids.shape[0]

                if prompt_length >= max_seq_len - 10:
                    continue

                batch_prompts.append(
                    {
                        "idx": idx,
                        "prompt_ids": prompt_ids,
                        "prompt_attention_mask": prompt_attention_mask,
                    }
                )

                if len(batch_prompts) >= generation_batch_size:
                    flush_batch()

            except Exception as e:
                print(f"Warning: Failed to generate for example {idx}: {e}")
                continue

        flush_batch()

    # End energy tracking
    if energy_tracker:
        energy_tracker.end_stage(tokens_processed=total_tokens_generated)

    print(f"Successfully generated {successful_generations} examples")
    print(f"Total tokens generated: {total_tokens_generated:,}")

    # Clean up teacher model
    del teacher_model
    torch.cuda.empty_cache()

    # Create dataset
    synthetic_dataset = datasets.Dataset.from_dict(synthetic_data)
    synthetic_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Split into train/eval
    split_dataset = synthetic_dataset.train_test_split(
        test_size=0.05,
        seed=config.seed,
    )

    # Save if path specified
    if synthetic_path:
        os.makedirs(synthetic_path, exist_ok=True)
        split_dataset.save_to_disk(synthetic_path)
        print(f"Saved synthetic dataset to: {synthetic_path}")

    return split_dataset


def load_synthetic_dataset(config: Config) -> datasets.DatasetDict:
    """Load existing synthetic dataset"""
    synthetic_path = config.get("synthetic_data.synthetic_dataset_path")

    try:
        print(f"Loading existing synthetic dataset from: {synthetic_path}")
        return datasets.load_from_disk(synthetic_path)
    except Exception as e:
        print(f"Synthetic dataset not found at {synthetic_path}. Generate dataset before running.")


def run_basic_checks(split_dataset: datasets.DatasetDict, tokenizer: AutoTokenizer, num_examples: int = 3) -> None:
    """Lightweight sanity checks on the saved synthetic dataset."""
    if not isinstance(split_dataset, datasets.DatasetDict):
        print("Dataset is not split; skipping split checks.")
        return

    train_len = len(split_dataset["train"])
    test_len = len(split_dataset["test"])
    total_len = train_len + test_len
    print(f"[CHECK] Columns: {split_dataset['train'].column_names}")
    print(f"[CHECK] Split sizes -> train: {train_len}, test: {test_len}, total: {total_len}")

    sample_ds = split_dataset["train"]
    if len(sample_ds) == 0:
        print("[CHECK] No samples available to decode.")
        return

    print(f"[CHECK] Showing up to {num_examples} decoded samples from train split:")
    for i in range(min(num_examples, len(sample_ds))):
        rec = sample_ds[i]
        input_ids = rec["input_ids"]
        attn = rec.get("attention_mask", None)

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if attn is not None and isinstance(attn, torch.Tensor):
            attn = attn.tolist()

        if attn is not None:
            trimmed_ids = [tid for tid, mask in zip(input_ids, attn) if mask == 1]
        else:
            trimmed_ids = input_ids

        decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        print(f"[CHECK] Sample {i} token ids (truncated 50): {trimmed_ids[:50]}")
        print(f"[CHECK] Sample {i} text: {decoded}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset with energy tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="synthetic_generation", config=cfg)

    ds = generate_synthetic_dataset(cfg, energy_tracker=tracker)
    tracker.save_summary()
    try:
        run_basic_checks(ds, AutoTokenizer.from_pretrained(cfg.tokenizer_name))
    except Exception as e:
        print(f"[CHECK] Skipping dataset checks due to error: {e}")
