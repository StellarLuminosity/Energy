"""
Preference dataset utilities for DPO training using teacher-generated pairs only.

For each prompt:
1) Teacher generates two responses with sampling.
2) Teacher log-probs determine chosen vs rejected.
3) Dataset is saved to disk for reuse.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from distill_bench.core.config_loader import Config
from distill_bench.core.energy_logger import EnergyTracker
from distill_bench.core.utils import main_print
from distill_bench.core.config_loader import load_config


def _build_prompt_text(example: Dict, tokenizer: AutoTokenizer) -> Optional[str]:
    """Return chat-formatted prompt text from either `messages` or `prompt`."""
    if "chat_text" in example and example["chat_text"]:
        return example["chat_text"]

    if example.get("messages"):
        user_messages = [m for m in example["messages"] if m.get("role") == "user"]
        if not user_messages:
            return None
        return tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt = example.get("prompt")
    if prompt:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return None


def _has_prompt_fields(ds: datasets.Dataset) -> bool:
    """Return True if dataset includes a prompt text column."""
    columns = set(ds.column_names)
    return bool({"messages", "prompt", "chat_text"} & columns)


def _load_prompt_dataset(config: Config) -> datasets.Dataset:
    """
    Load prompts for processed dataset.
    """
    errors = []

    dataset_name = config.get("dpo.prompt_dataset_name", None) or config.dataset_name
    if dataset_name:
        try:
            ds = datasets.load_dataset(dataset_name, split="train")
            if _has_prompt_fields(ds):
                return ds
            errors.append(
                f"prompt dataset '{dataset_name}' is missing prompt text columns (`messages`, `prompt`, or `chat_text`)"
            )
        except Exception as e:
            errors.append(f"failed to load prompt dataset {dataset_name}: {e}")

    prompt_dataset_path = config.get("dpo.prompt_dataset_path", None) or config.dataset_path
    if prompt_dataset_path:
        try:
            ds = datasets.load_from_disk(prompt_dataset_path)
            ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
            if _has_prompt_fields(ds):
                return ds
            errors.append("dataset on disk is missing prompt text columns (`messages`, `prompt`, or `chat_text`)")
        except Exception as e:
            errors.append(f"failed to load prompt dataset from disk: {e}")

    error_str = "; ".join(errors) if errors else "no prompt dataset configured"
    raise ValueError(f"Unable to load prompt dataset for preference generation: {error_str}")


def generate_preference_dataset(
    config: Config,
    tokenizer: AutoTokenizer,
    energy_tracker: Optional[EnergyTracker] = None,
    stage_name: str = "teacher_generation",
) -> datasets.DatasetDict:
    """
    Generate preference pairs using the teacher model (generation + labeling).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = config.get("data.max_sequence_length", 1024)
    max_new_tokens = config.get("dpo.judge_labeling.max_new_tokens", 256)
    top_p = config.get("dpo.judge_labeling.top_p", 0.9)
    temperature = config.temperature
    prompt_limit = config.get("dpo.max_prompts", 8000)
    started_here = False

    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True

    main_print(f"Loading teacher model for preference generation: {config.teacher_model_name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    teacher_model.eval()

    print(f"Loading prompt dataset...")
    prompt_ds = _load_prompt_dataset(config)
    # Shuffle with a fixed seed for determinism before subsetting.
    prompt_seed = getattr(config, "seed", 42)
    prompt_ds = prompt_ds.shuffle(seed=prompt_seed)
    if prompt_limit and len(prompt_ds) > prompt_limit:
        prompt_ds = prompt_ds.select(range(prompt_limit))

    pairs: List[Dict] = []
    total_tokens = 0
    counter = 0

    batch_size = config.batch_size
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    buffer_prompt_texts: List[str] = []
    buffer_examples: List[Dict] = []
    buffer_prompt_ids: List[int] = []

    def _flush_batch():
        nonlocal counter, total_tokens, pairs
        if not buffer_prompt_texts:
            return

        # Tokenize as a batch
        batch_inputs = tokenizer(
            buffer_prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        prompt_lens = batch_inputs["attention_mask"].sum(dim=1).tolist()
        padded_prompt_len = batch_inputs["input_ids"].shape[1]

        # Generate 2 responses per prompt in one call and keep per-step scores for logprobs
        with torch.inference_mode():
            gen = teacher_model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=2,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_id,
            )

            sequences = gen.sequences  # (B*2, padded_prompt_len + gen_len)
            transition_scores = teacher_model.compute_transition_scores(
                sequences, gen.scores, normalize_logits=True
            )  # (B*2, gen_len)

        # For each original prompt i, its 2 generations are at indices 2*i and 2*i+1
        B = len(buffer_prompt_texts)
        for i in range(B):
            idx0 = 2 * i
            idx1 = 2 * i + 1

            # Decode only generated tokens (everything after padded prompt length)
            resp0 = tokenizer.decode(sequences[idx0, padded_prompt_len:], skip_special_tokens=True).strip()
            resp1 = tokenizer.decode(sequences[idx1, padded_prompt_len:], skip_special_tokens=True).strip()

            logp0 = float(transition_scores[idx0].sum().item())
            logp1 = float(transition_scores[idx1].sum().item())

            # Token accounting: count only generated tokens for energy bookkeeping
            gen_len = transition_scores.size(1)
            total_tokens += 2 * gen_len
            if energy_tracker:
                energy_tracker.add_tokens(2 * gen_len)

            if logp0 >= logp1:
                chosen, rejected = resp0, resp1
                logp_chosen, logp_rejected = logp0, logp1
            else:
                chosen, rejected = resp1, resp0
                logp_chosen, logp_rejected = logp1, logp0

            pairs.append(
                {
                    "prompt": buffer_examples[i].get("prompt", ""),
                    "chosen": chosen,
                    "rejected": rejected,
                    "teacher_logp_chosen": logp_chosen,
                    "teacher_logp_rejected": logp_rejected,
                    "meta": {
                        "prompt_id": buffer_prompt_ids[i],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "prompt_len": int(prompt_lens[i]),
                    },
                }
            )

            counter += 1
            if counter >= 10:
                print("Reached early stop after 10 generated pairs")
                return

        # Clear buffers
        buffer_prompt_texts.clear()
        buffer_examples.clear()
        buffer_prompt_ids.clear()

    for idx, example in enumerate(tqdm(prompt_ds, desc="Generating preference pairs")):
        prompt_text = _build_prompt_text(example, tokenizer)
        if prompt_text is None:
            continue

        # Quick length check before batching: skip near-max prompts
        # Note: tokenize without padding to estimate length accurately
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).input_ids
        prompt_len = len(prompt_ids)
        if prompt_len >= max_length - 4:
            continue

        buffer_prompt_texts.append(prompt_text)
        buffer_examples.append(example)
        buffer_prompt_ids.append(idx)

        if len(buffer_prompt_texts) >= batch_size:
            _flush_batch()
            if counter >= 10:
                break

    # Flush leftovers
    if counter < 10:
        _flush_batch()

    main_print(f"Generated {len(pairs)} preference pairs from {len(prompt_ds)} prompts")
    teacher_model.to("cpu")
    torch.cuda.empty_cache()

    dataset = datasets.Dataset.from_list(pairs)
    dataset = dataset.train_test_split(test_size=0.05, seed=config.seed)

    save_path = config.preference_dataset_path
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_path)
    main_print(f"Saved generated preference dataset to: {save_path}")

    if energy_tracker and started_here:
        energy_tracker.end_stage(tokens_processed=total_tokens)

    return dataset


def load_or_build_preference_dataset(
    config: Config,
    tokenizer: AutoTokenizer,
    energy_tracker: Optional[EnergyTracker] = None,
    stage_name: str = "teacher preference dataset label generation (dpo)",
) -> datasets.DatasetDict:
    """
    Always build (or reuse cached) preference dataset from teacher generations.
    """
    ds_path = config.preference_dataset_path or (Path(config.output_dir) / "preference_dataset")
    if Path(ds_path).exists():
        main_print(f"Loading preference dataset from: {ds_path}")
        return datasets.load_from_disk(ds_path)

    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
    dataset = generate_preference_dataset(config, tokenizer, energy_tracker, stage_name=stage_name)
    if energy_tracker and energy_tracker.current_stage:
        energy_tracker.end_stage()
    return dataset


def _tokenize_pair(
    prompt: str,
    response: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Tuple[List[int], int]:
    """Tokenize prompt/response pair and return token ids + prompt length."""
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_only,
        truncation=True,
        max_length=max_length,
    ).input_ids
    prompt_len = len(prompt_ids)

    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
    ).input_ids

    prompt_len = min(prompt_len, len(full_ids))
    return full_ids, prompt_len


def prepare_dpo_dataset(
    dataset: datasets.DatasetDict,
    tokenizer: AutoTokenizer,
    config: Config,
) -> datasets.DatasetDict:
    """Tokenize chosen/rejected pairs with prompt-length bookkeeping."""
    max_length = config.get("data.max_sequence_length", 1024)

    def _process(example: Dict) -> Dict:
        chosen_ids, chosen_prompt_len = _tokenize_pair(example["prompt"], example["chosen"], tokenizer, max_length)
        rejected_ids, rejected_prompt_len = _tokenize_pair(example["prompt"], example["rejected"], tokenizer, max_length)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_prompt_len": chosen_prompt_len,
            "rejected_input_ids": rejected_ids,
            "rejected_prompt_len": rejected_prompt_len,
        }

    processed = dataset.map(_process, remove_columns=dataset["train"].column_names)
    return processed


def _pad_sequences(seqs: List[List[int]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences."""
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
    return input_ids, attention_mask


def dpo_collate_fn(pad_token_id: int):
    """Collate function that pads chosen/rejected sequences separately."""

    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        chosen_seqs = [item["chosen_input_ids"] for item in batch]
        rejected_seqs = [item["rejected_input_ids"] for item in batch]

        chosen_input_ids, chosen_attention = _pad_sequences(chosen_seqs, pad_token_id)
        rejected_input_ids, rejected_attention = _pad_sequences(rejected_seqs, pad_token_id)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention,
            "chosen_prompt_len": torch.tensor([item["chosen_prompt_len"] for item in batch], dtype=torch.long),
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention,
            "rejected_prompt_len": torch.tensor([item["rejected_prompt_len"] for item in batch], dtype=torch.long),
        }

    return _collate


def create_dpo_dataloaders(
    dataset: datasets.DatasetDict,
    tokenizer: AutoTokenizer,
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/eval dataloaders for DPO."""
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collate = dpo_collate_fn(pad_token_id)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        dataset["test"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    return train_loader, eval_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate preference dataset with energy tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Override output.run_dir for this run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_dir:
        cfg.override_run_dir(args.run_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    # Initialize energy tracker for standalone use (stage name = filename)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="dpo_preference_dataset", config=cfg)

    dataset = generate_preference_dataset(cfg, tokenizer, tracker, stage_name="preference_dataset")
    tracker.save_summary()
