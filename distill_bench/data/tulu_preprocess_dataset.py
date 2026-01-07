import torch
import random
import os
import shutil
import datasets
from transformers import AutoTokenizer
import torch.nn.functional as F
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

# ----------------------------------
# Helper Functions
# ----------------------------------
def create_response_labels(sample, tokenizer):
    if not isinstance(sample["input_ids"], torch.Tensor):
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)

    if not isinstance(sample["attention_mask"], torch.Tensor):
        sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)

    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    labels = input_ids.clone()
    labels.fill_(-100)

    response_ids = tokenizer("<|assistant|>\n", add_special_tokens=False)["input_ids"]        # Change according to different templates
    start_pos = -1
    for i in range(len(input_ids) - len(response_ids) + 1):
        if input_ids[i : i + len(response_ids)].tolist() == response_ids:
            start_pos = i + len(response_ids)
            break
    
    end_pos = len(input_ids)
    # last token with mask==1
    last_valid = attn.nonzero(as_tuple=True)[0].max().item()
    end_pos = last_valid + 1

    labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    labels = labels.masked_fill(attn == 0, -100)

    return labels

def format_chat_data(sample, tokenizer):
    return {"chat_text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}


def tokenize_text(sample, tokenizer):
    tokenized = tokenizer(
        sample["chat_text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",
    )

    sample["input_ids"] = tokenized["input_ids"].squeeze(0)
    sample["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    return sample


def add_labels(sample, tokenizer):
    sample["labels"] = create_response_labels(sample, tokenizer)
    return sample


def contains_complete_response_template(sample, tokenizer):
    """Check if the example contains the complete assistant response template."""
    response_template_ids = tokenizer("<|assistant|>\n")["input_ids"]       # Change based on model/tokenizer

    for start_idx in range(len(sample["input_ids"]) - len(response_template_ids) + 1):
        if sample["input_ids"][start_idx : start_idx + len(response_template_ids)].tolist() == response_template_ids:
            return True
    return False


def main(config, energy_tracker: EnergyTracker = None, stage_name: str = "tulu_preprocess_dataset"):
    """Main preprocessing function."""
    started_here = False
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage(stage_name)
        started_here = True
    # ----------------------------------
    # Load Dataset
    # ----------------------------------
    print("\n=== LOADING DATASET ===")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    print(f"Tokenizer loaded: {config.tokenizer_name}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Verify this is OLMo tokenizer (should be ~100k tokens)
    if len(tokenizer) < 90000 or len(tokenizer) > 110000:
        print(f"⚠️  WARNING: Tokenizer vocab size ({len(tokenizer)}) seems unusual for OLMo!")
        print(f"   Expected ~100,278 tokens for OLMo-2 models")
        print(f"   Are you sure you're using the right tokenizer?")
    else:
        print(f"✓ Tokenizer vocab size looks correct for OLMo")

    dataset = datasets.load_dataset(config.dataset_name, split="train")

    print(f"Original dataset size: {len(dataset)}")
    print(f"Original dataset features: {dataset.features}")

    print(f"Example raw message format:")
    print(dataset[random.randint(0, len(dataset) - 1)]["messages"])
    print(f"Another example raw message format:")
    print(dataset[random.randint(0, len(dataset) - 1)]["messages"])

    # ----------------------------------
    # Shuffle and Sample Dataset
    # ----------------------------------
    dataset = dataset.shuffle(config.seed)
    dataset = dataset.select(range(200_000))
    dataset = dataset.train_test_split(test_size=2000)
    print(f"\nAfter sampling - Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # ------------------------------------------
    # Apply preprocessing to format chat data
    # ------------------------------------------
    print("\n=== APPLYING CHAT TEMPLATE ===")
    processed_dataset = dataset.map(lambda x: format_chat_data(x, tokenizer), num_proc=32)

    print(f"Examples after chat formatting:")
    print(f"Train example chat_text (first 300 chars):\n{processed_dataset['train'][0]['chat_text'][:300]}...")
    print(f"Test example chat_text (first 300 chars):\n{processed_dataset['test'][0]['chat_text'][:300]}...")

    # --------------------------
    # Tokenize the text
    # --------------------------
    print("\n=== TOKENIZING TEXT ===")
    tokenized_dataset = processed_dataset.map(lambda x: tokenize_text(x, tokenizer), remove_columns=["messages", "source"], num_proc=32)
    print(f"Dataset features after tokenization: {tokenized_dataset['train'].features}")

    print(f"Train example input_ids shape: {torch.tensor(tokenized_dataset['train'][0]['input_ids']).shape}")
    print(f"Train example attention_mask shape: {torch.tensor(tokenized_dataset['train'][0]['attention_mask']).shape}")
    print(f"Train example id: {tokenized_dataset['train'][0]['id']}")

    labeled_dataset = tokenized_dataset.map(lambda x: add_labels(x, tokenizer), num_proc=32)
    print(f"Dataset features after adding labels: {labeled_dataset['train'].features}")
    print(f"ID column preserved - example id: {labeled_dataset['train'][0]['id']}")

    # -----------------------------------------
    # Filter out samples which were truncated
    # -----------------------------------------
    print("\n=== FILTERING EXAMPLES ===")

    labeled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "id"])
    num_train_before = len(labeled_dataset["train"])
    train_keep_count = sum(
        1
        for _ in filter(
            lambda x: contains_complete_response_template(x, tokenizer),
            (labeled_dataset["train"][i] for i in range(min(1000, num_train_before))),
        )
    )
    print(
        f"Estimated percentage of train examples to keep: {train_keep_count/min(1000, num_train_before)*100:.2f}% (based on 1000 samples)"
    )
    final_dataset = labeled_dataset.filter(lambda x: contains_complete_response_template(x, tokenizer), num_proc=32)
    print(f"Dataset size after filtering - Train: {len(final_dataset['train'])}, Test: {len(final_dataset['test'])}")

    # ------------------------------
    # Save the processed dataset
    # ------------------------------
    print("\n=== SAVING DATASET ===")

    save_path = config.dataset_path
    if os.path.exists(save_path):
        shutil.rmtree(save_path) 
    final_dataset.save_to_disk(save_path)
    print(f"Dataset saved to: {save_path}")

    # ------ Save a clean version with only required columns for training ------
    clean_dataset = final_dataset.remove_columns(['chat_text'])  # Keep id for potential future use
    clean_save_path = save_path + "_clean"
    if os.path.exists(clean_save_path):
        shutil.rmtree(clean_save_path)
    clean_dataset.save_to_disk(clean_save_path)
    print(f"Clean dataset (no chat_text) saved to: {clean_save_path}")
    # ------- End saving code ------------

    # ------ Verify token IDs are within vocabulary range ------
    print("\n=== VERIFYING TOKEN IDs ===")
    print("Checking token ID ranges in first 1000 samples...")

    max_token_id = 0
    min_token_id = float('inf')

    for i, example in enumerate(final_dataset['train'].select(range(min(1000, len(final_dataset['train']))))):
        input_ids = example['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        max_token_id = max(max_token_id, max(input_ids))
        min_token_id = min(min_token_id, min(input_ids))

    print(f"Token ID range: {min_token_id} to {max_token_id}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    if max_token_id >= len(tokenizer):
        print(f"❌ ERROR: Max token ID ({max_token_id}) >= vocab size ({len(tokenizer)})")
        print(f"   This will cause CUDA errors during training!")
        print(f"   Check that you're using the correct tokenizer.")
    else:
        print(f"✓ All token IDs are within vocabulary range!")

    print("\nDataset processing complete!")
    if energy_tracker and started_here:
        total_examples = len(final_dataset["train"]) + len(final_dataset["test"])
        energy_tracker.add_tokens(total_examples)
        energy_tracker.end_stage(tokens_processed=total_examples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Tulu dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_dir = Path(getattr(config, "run_dir", None) or config.get("output.run_dir", None) or getattr(config, "output_dir", "logs"))
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = EnergyTracker(run_dir=str(run_dir), experiment_name="tulu_preprocess_dataset", config=config)

    main(config, energy_tracker=tracker, stage_name="tulu_preprocess_dataset")
    tracker.save_summary()
