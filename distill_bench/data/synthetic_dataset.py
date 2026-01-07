"""Generates synthetic dataset labels using the teacher model"""

import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import os
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker


def main(config, energy_tracker: EnergyTracker = None, stage_name: str = "synthetic_dataset"):
    """Generate synthetic dataset using teacher model."""
    started_here = False  # whether we passed the energy tracker from an external process or this is a separate run
    total_tokens = 0
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True

    dataset = datasets.load_from_disk(config.dataset_path)
    dataloader = DataLoader(dataset["train"].select(range(100)), batch_size=2)

    ds = {"input_ids": [], "attention_mask": [], "labels": []}

    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.teacher_device,
    )
    teacher.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
    teacher.eval()

    print("\n=== GENERATING TEACHER LOGITS ===")
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(config.teacher_device)
        attention_mask = batch["attention_mask"].to(config.teacher_device)

        # Append the original inputs directly
        ds["input_ids"].extend(input_ids.cpu().unbind())
        ds["attention_mask"].extend(attention_mask.cpu().unbind())

        with torch.no_grad():
            generation_output = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.5,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_sequences = generation_output.sequences
            gen_only = generated_sequences[:, input_ids.shape[1] :]

            # Create labels tensor with correct padding
            # The labels should be the same length as the input_ids + gen_only
            # We pad the original input_ids part with -100
            max_len = generated_sequences.shape[1]
            labels = torch.full((input_ids.size(0), max_len), fill_value=-100, dtype=torch.long, device=gen_only.device)

            # Fill the generated part with the actual tokens
            labels[:, input_ids.size(1) :] = gen_only

            ds["labels"].extend(labels.cpu().unbind())
            if energy_tracker:
                tok = (labels != -100).sum().item()
                total_tokens += tok
                energy_tracker.add_tokens(tok)

            del generation_output, generated_sequences, gen_only, labels, input_ids, attention_mask
            torch.cuda.empty_cache()

    dset = datasets.Dataset.from_dict(ds)
    dset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.makedirs(config.synthetic_dataset_path, exist_ok=True)
    dset.save_to_disk(config.synthetic_dataset_path)

    if energy_tracker and started_here:
        energy_tracker.end_stage(tokens_processed=total_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset using teacher")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(
        getattr(config, "run_dir", None) or config.get("output.run_dir", None) or getattr(config, "output_dir", "logs")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="synthetic_dataset", config=config)

    main(config, energy_tracker=tracker, stage_name="synthetic_dataset")
    tracker.save_summary()
