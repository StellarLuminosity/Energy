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

    max_seq_len = config.get("data.max_sequence_length", 1024)
    num_samples = config.get("synthetic_data.num_samples", 50000)
    synthetic_path = config.get("synthetic_data.synthetic_dataset_path")
    dataset_path = config.dataset_path or config.get("data.dataset_path")

    if not dataset_path:
        raise ValueError("dataset_path is not set. Run tulu_preprocess_dataset.py first.")

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

    if len(prompt_dataset) > num_samples:
        prompt_dataset = prompt_dataset.shuffle(seed=config.seed).select(range(num_samples))

    print(f"Generating {len(prompt_dataset)} synthetic examples...")

    # Storage for synthetic data
    synthetic_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    total_tokens_generated = 0
    successful_generations = 0

    # Generate responses
    with torch.no_grad():
        for idx, example in enumerate(tqdm(prompt_dataset, desc="Generating synthetic data")):
            try:
                input_ids = torch.tensor(example["input_ids"], device=device)
                attention_mask = torch.tensor(example["attention_mask"], device=device)
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

                # Generate response
                generation_kwargs = {
                    "max_new_tokens": min(max_new_tokens, max_seq_len - prompt_length),
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "do_sample": decoding_strategy == "sampling",
                }

                if decoding_strategy == "sampling":
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["top_p"] = top_p

                outputs = teacher_model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    attention_mask=prompt_attention_mask.unsqueeze(0),
                    **generation_kwargs,
                )

                # Full generated sequence (prompt + response)
                full_sequence = outputs[0]
                generated_tokens = full_sequence[prompt_length:]

                # Create labels: mask prompt (-100), keep response
                synthetic_labels = torch.full_like(full_sequence, fill_value=-100)
                synthetic_labels[prompt_length:] = full_sequence[prompt_length:]

                # Apply filtering on response length and max length
                filtering_config = config.get("synthetic_data.filtering", {})
                if filtering_config.get("enabled", True):
                    min_length = filtering_config.get("min_length", 10)
                    max_length = filtering_config.get("max_length", max_seq_len)
                    response_length = len(generated_tokens)
                    total_length = len(full_sequence)
                    if response_length < min_length:
                        print(f"Response length shorter than min length - skipping idx {idx}")
                        continue
                    if total_length > max_length:
                        print(f"Total length (prompt + response) is greater than max length - skipping idx {idx}")
                        continue

                # Create attention mask (all 1s for valid sequence)
                output_attention_mask = torch.ones_like(full_sequence)

                # Store
                synthetic_data["input_ids"].append(full_sequence.cpu().tolist())
                synthetic_data["attention_mask"].append(output_attention_mask.cpu().tolist())
                synthetic_data["labels"].append(synthetic_labels.cpu().tolist())

                total_tokens_generated += len(generated_tokens)
                successful_generations += 1

                # Periodic cleanup
                if idx % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Warning: Failed to generate for example {idx}: {e}")
                continue

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset with energy tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="synthetic_generation", config=cfg)

    generate_synthetic_dataset(cfg, energy_tracker=tracker)
    tracker.save_summary()
