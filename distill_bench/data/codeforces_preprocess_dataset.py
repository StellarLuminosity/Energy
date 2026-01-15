import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from transformers import AutoTokenizer

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

# -----------------------------------------------------------------------------
# Tokenizer caching per-process (works well with datasets.map(num_proc=...))
# -----------------------------------------------------------------------------
_TOKENIZER = None


def get_tokenizer(tokenizer_name: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    return _TOKENIZER


# -----------------------------------------------------------------------------
# Assistant-content postprocessing (optional)
# -----------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_CODEBLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", flags=re.DOTALL)


def strip_think(text: str) -> str:
    return re.sub(_THINK_RE, "", text).strip()


def keep_last_code_block(text: str) -> str:
    matches = list(re.finditer(_CODEBLOCK_RE, text))
    if not matches:
        return text.strip()
    last = matches[-1].group(0)  # include the ```...``` fences
    return last.strip()


def maybe_transform_assistant(
    assistant_text: str,
    *,
    strip_think_blocks: bool,
    code_only: bool,
) -> str:
    out = assistant_text
    if strip_think_blocks:
        out = strip_think(out)
    if code_only:
        out = keep_last_code_block(out)
    return out


# -----------------------------------------------------------------------------
# Core labeling logic (Tulu-style: labels only on assistant span)
# We compute response_start by tokenizing the PROMPT (all msgs except last)
# with add_generation_prompt=True, then label from there in the full sequence.
# -----------------------------------------------------------------------------
def build_prompt_and_full_text(
    messages: List[Dict[str, Any]],
    tokenizer,
) -> Dict[str, str]:
    # Full conversation (includes assistant content)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Prompt only (everything before the final assistant message),
    # with generation prompt turned on so it includes the assistant marker.
    prefix_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)

    return {"prompt_text": prompt_text, "full_text": full_text}


def preprocess_batch(
    batch: Dict[str, List[Any]],
    *,
    tokenizer_name: str,
    max_length: int,
    strip_think_blocks: bool,
    code_only: bool,
) -> Dict[str, List[Any]]:
    tokenizer = get_tokenizer(tokenizer_name)

    # 1) Normalize to a messages list
    raw_messages_list = batch.get("messages", None)
    prompts = batch.get("prompt", None)
    generations = batch.get("generation", None)

    normalized_messages: List[List[Dict[str, Any]]] = []
    for i in range(len(batch["id"])):
        msgs = None
        if raw_messages_list is not None and raw_messages_list[i] is not None:
            msgs = raw_messages_list[i]

        # Fallback if messages missing
        if not msgs:
            user_text = prompts[i] if prompts is not None else ""
            asst_text = generations[i] if generations is not None else ""
            msgs = [{"role": "user", "content": user_text}, {"role": "assistant", "content": asst_text}]

        # Ensure last message is assistant; if not, append generation if available
        if msgs[-1].get("role") != "assistant":
            asst_text = generations[i] if generations is not None else ""
            msgs = list(msgs) + [{"role": "assistant", "content": asst_text}]

        # Optionally transform assistant content (strip <think> / keep code only)
        msgs = [dict(m) for m in msgs]
        last = dict(msgs[-1])
        last["content"] = maybe_transform_assistant(
            last.get("content", ""),
            strip_think_blocks=strip_think_blocks,
            code_only=code_only,
        )
        msgs[-1] = last

        normalized_messages.append(msgs)

    # 2) Build prompt_text (prefix + assistant marker) and full_text
    prompt_texts: List[str] = []
    full_texts: List[str] = []
    for msgs in normalized_messages:
        texts = build_prompt_and_full_text(msgs, tokenizer)
        prompt_texts.append(texts["prompt_text"])
        full_texts.append(texts["full_text"])

    # 3) Tokenize
    full_tok = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=True,
    )
    prompt_tok = tokenizer(
        prompt_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        add_special_tokens=True,
    )

    input_ids_batch = full_tok["input_ids"]
    attn_batch = full_tok["attention_mask"]
    response_starts = [len(x) for x in prompt_tok["input_ids"]]

    # 4) Create labels: only assistant span (response_start .. last_valid)
    labels_batch: List[List[int]] = []
    for input_ids, attn, rs in zip(input_ids_batch, attn_batch, response_starts):
        # last valid token index is sum(attn)-1 when padding is at the end
        valid = sum(attn)
        end = valid  # slice end is exclusive
        labels = [-100] * len(input_ids)
        if 0 <= rs < end:
            labels[rs:end] = input_ids[rs:end]
        # padding already -100
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attn_batch,
        "labels": labels_batch,
        "response_start": response_starts,  # for debugging/filtering; we'll drop before save
    }


def keep_nonempty_labels(example: Dict[str, Any]) -> bool:
    labels = example["labels"]
    return any(l != -100 for l in labels)


def main(config, energy_tracker: Optional[EnergyTracker] = None, stage_name: str = "codeforces_cots_preprocess"):
    started_here = False
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True

    dataset_name = config.dataset_name  # should be "open-r1/codeforces-cots"
    dataset_subset = getattr(config, "dataset_subset", None) or "solutions"
    split = getattr(config, "dataset_split", None) or "train"

    max_length = int(getattr(config, "max_sequence_length", 1024))
    seed = int(getattr(config, "seed", 42))
    num_samples = int(getattr(config, "num_samples", 0) or 0)  # 0 => use all
    test_size = float(getattr(config, "test_size", 0.05) or 0.05)
    num_proc = int(getattr(config, "num_proc", 8) or 8)

    strip_think_blocks = bool(getattr(config, "strip_think_blocks", True))
    code_only = bool(getattr(config, "code_only", False))

    print("\n=== LOADING DATASET ===")
    print(f"Dataset: {dataset_name} | Subset: {dataset_subset} | Split: {split}")
    ds = datasets.load_dataset(dataset_name, dataset_subset, split=split)

    print(f"Original size: {len(ds)}")
    ds = ds.shuffle(seed=seed)
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
        print(f"After subsampling: {len(ds)}")

    # create a train/test split (dataset only ships with train in these subsets)
    split_ds = ds.train_test_split(test_size=test_size, seed=seed)
    print(f"Split: train={len(split_ds['train'])}, test={len(split_ds['test'])}")

    # Preprocess (batched for speed)
    print("\n=== TOKENIZING + LABELING ===")
    tokenized = split_ds.map(
        preprocess_batch,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={
            "tokenizer_name": config.tokenizer_name,
            "max_length": max_length,
            "strip_think_blocks": strip_think_blocks,
            "code_only": code_only,
        },
        desc="Preprocessing",
    )

    # Filter examples where assistant span got truncated away (all -100)
    print("\n=== FILTERING TRUNCATED / EMPTY ASSISTANT SPANS ===")
    before = (len(tokenized["train"]), len(tokenized["test"]))
    tokenized = tokenized.filter(keep_nonempty_labels, num_proc=num_proc, desc="Filtering")
    after = (len(tokenized["train"]), len(tokenized["test"]))
    print(f"Kept train={after[0]}/{before[0]}, test={after[1]}/{before[1]}")

    if after[0] == 0:
        raise ValueError("No train examples left after filtering. Try increasing max_sequence_length.")

    # Keep only Tulu-style columns
    keep_cols = {"input_ids", "attention_mask", "labels", "id"}
    drop_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    cleaned = tokenized.remove_columns(drop_cols)

    # Torch formatting at the end
    cleaned.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Save
    save_path = config.dataset_path
    print(f"\n=== SAVING TO {save_path} ===")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    cleaned.save_to_disk(save_path)
    print(f"Saved train={len(cleaned['train'])}, test={len(cleaned['test'])}")

    if energy_tracker and started_here:
        # If you want true token counts, compute sum(attention_mask) over dataset (costly).
        total_examples = len(cleaned["train"]) + len(cleaned["test"])
        energy_tracker.add_tokens(total_examples)
        energy_tracker.end_stage(tokens_processed=total_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess open-r1/codeforces-cots for SFT")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(getattr(cfg, "run_dir", None) or cfg.get("output.run_dir", None) or getattr(cfg, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="codeforces_cots_preprocess", config=cfg)
    main(cfg, energy_tracker=tracker, stage_name="codeforces_cots_preprocess")
    tracker.save_summary()
