"""
Preference dataset utilities for DPO training.

Supports two modes:
1) Load an existing preference dataset from disk (with `prompt`, `chosen`, `rejected`).
2) Generate preference pairs on-the-fly with a judge model by sampling two responses
   per prompt and selecting the higher-likelihood response as `chosen`.
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


def _build_prompt_text(example: Dict, tokenizer: AutoTokenizer) -> Optional[str]:
    """Return chat-formatted prompt text from either `messages` or `prompt`."""
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


def _sequence_logprob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> float:
    """Compute summed log-probability of the response tokens (post prompt)."""
    with torch.no_grad():
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
        response_mask = (positions >= (prompt_length - 1)) & attention_mask[:, 1:].bool()

        return (token_log_probs * response_mask).sum(dim=1).item()


def generate_preference_dataset(
    config: Config,
    tokenizer: AutoTokenizer,
    energy_tracker: Optional[EnergyTracker] = None,
) -> datasets.DatasetDict:
    """
    Generate preference pairs using a judge model.

    For each prompt:
    - Sample two responses from the judge.
    - Compute judge log-prob for each response.
    - Label the higher-likelihood response as chosen, the other as rejected.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = config.get("data.max_sequence_length", 1024)
    max_new_tokens = config.get("dpo.judge_labeling.max_new_tokens", 256)
    top_p = config.get("dpo.judge_labeling.top_p", 0.9)
    temperature = config.dpo_judge_temperature
    prompt_limit = config.get("dpo.max_prompts", 2000)

    main_print(f"Loading judge model: {config.judge_model_name}")
    judge_model = AutoModelForCausalLM.from_pretrained(
        config.judge_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    judge_model.eval()

    prompt_dataset_name = config.dpo_preference_dataset or config.dataset_name
    main_print(f"Loading prompt dataset: {prompt_dataset_name}")
    prompt_ds = datasets.load_dataset(prompt_dataset_name, split="train")
    if prompt_limit and len(prompt_ds) > prompt_limit:
        prompt_ds = prompt_ds.select(range(prompt_limit), seed=config.seed)

    pairs: List[Dict] = []

    for idx, example in enumerate(tqdm(prompt_ds, desc="Generating preference pairs")):
        prompt_text = _build_prompt_text(example, tokenizer)
        if prompt_text is None:
            continue

        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        if prompt_len >= max_length - 4:
            continue

        responses: List[str] = []
        logps: List[float] = []
        seeds = [config.seed + 2 * idx, config.seed + 2 * idx + 1]

        for resp_seed in seeds:
            torch.manual_seed(resp_seed)
            outputs = judge_model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            full_sequence = outputs[0]
            response_tokens = full_sequence[prompt_len:]
            responses.append(tokenizer.decode(response_tokens, skip_special_tokens=True).strip())

            attention_mask = torch.ones_like(full_sequence).unsqueeze(0).to(device)
            input_ids = full_sequence.unsqueeze(0).to(device)
            logps.append(_sequence_logprob(judge_model, input_ids, attention_mask, prompt_len))

        if len(responses) != 2 or len(logps) != 2:
            continue

        if logps[0] >= logps[1]:
            chosen, rejected = responses[0], responses[1]
            logp_chosen, logp_rejected = logps[0], logps[1]
            seeds_pair = (seeds[0], seeds[1])
        else:
            chosen, rejected = responses[1], responses[0]
            logp_chosen, logp_rejected = logps[1], logps[0]
            seeds_pair = (seeds[1], seeds[0])

        pairs.append(
            {
                "prompt": example.get("prompt", ""),
                "chosen": chosen,
                "rejected": rejected,
                "teacher_logp_chosen": logp_chosen,
                "teacher_logp_rejected": logp_rejected,
                "meta": {
                    "prompt_id": idx,
                    "seed_chosen": seeds_pair[0],
                    "seed_rejected": seeds_pair[1],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )

    judge_model.to("cpu")
    torch.cuda.empty_cache()

    dataset = datasets.Dataset.from_list(pairs)
    dataset = dataset.train_test_split(test_size=0.05, seed=config.seed)

    if config.dpo_preference_dataset_path:
        save_path = Path(config.dpo_preference_dataset_path)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(save_path)
        main_print(f"Saved generated preference dataset to: {save_path}")

    return dataset


def load_or_build_preference_dataset(
    config: Config,
    tokenizer: AutoTokenizer,
    energy_tracker: Optional[EnergyTracker] = None,
) -> datasets.DatasetDict:
    """
    Load a preference dataset from disk or generate one with the judge.

    Expects dataset columns: prompt, chosen, rejected.
    """
    ds_path = config.dpo_preference_dataset_path
    if ds_path and Path(ds_path).exists():
        main_print(f"Loading preference dataset from: {ds_path}")
        return datasets.load_from_disk(ds_path)

    if config.dpo_judge_enabled:
        if energy_tracker:
            energy_tracker.start_stage("judge_labeling")
        dataset = generate_preference_dataset(config, tokenizer, energy_tracker)
        if energy_tracker:
            energy_tracker.end_stage()
        return dataset

    raise FileNotFoundError(
        "Preference dataset not found on disk and judge_labeling.disabled. "
        "Provide a dataset path or enable judge_labeling."
    )


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
        chosen_ids, chosen_prompt_len = _tokenize_pair(
            example["prompt"], example["chosen"], tokenizer, max_length
        )
        rejected_ids, rejected_prompt_len = _tokenize_pair(
            example["prompt"], example["rejected"], tokenizer, max_length
        )

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

