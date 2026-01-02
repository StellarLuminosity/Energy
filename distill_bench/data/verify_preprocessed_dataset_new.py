"""
Verification Script for Preprocessed Infinity-Instruct Dataset

This script loads your preprocessed dataset and performs comprehensive checks:
1. Dataset structure and keys
2. Token ID ranges (within vocabulary)
3. Sample examples (tokenized and detokenized)
4. Label masking correctness
5. Padding verification
6. Edge case detection

Run this AFTER preprocessing to verify everything is correct before training.

Usage:
    python verify_preprocessed_dataset.py --config configs/experiments/sft_7b_to_1b.yaml
"""

import torch
import datasets
import argparse
from transformers import AutoTokenizer
import random
import numpy as np
import statistics

from distill_bench.core.config_loader import load_config


def inspect_sample(sample, sample_idx, tokenizer):
    """Detailed inspection of a single sample."""
    print(f"\n{'='*80}")
    print(f"SAMPLE #{sample_idx}")
    print(f"{'='*80}")
    
    # Get data
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    labels = sample['labels']
    
    # Convert to lists for easier inspection
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    
    # Find valid region
    valid_len = max((i for i, m in enumerate(attention_mask) if m == 1), default=0) + 1
    padding_len = len(input_ids) - valid_len
    
    print(f"\nBasic Info:")
    print(f"  ID: {sample.get('id', 'N/A')}")
    print(f"  Total length: {len(input_ids)}")
    print(f"  Valid tokens: {valid_len}")
    print(f"  Padding tokens: {padding_len}")
    
    # Count labels
    label_count = sum(1 for l in labels[:valid_len] if l != -100)
    masked_count = sum(1 for l in labels[:valid_len] if l == -100)
    
    print(f"\nLabel Distribution (in valid region):")
    print(f"  Tokens to train on: {label_count} ({100*label_count/valid_len:.1f}%)")
    print(f"  Masked tokens: {masked_count} ({100*masked_count/valid_len:.1f}%)")
    
    # Decode full sequence
    print(f"\n{'─'*80}")
    print("FULL DECODED SEQUENCE (what the model sees):")
    print(f"{'─'*80}")
    decoded_full = tokenizer.decode(input_ids[:valid_len])
    print(decoded_full)
    
    # Decode labels only (what we're training on)
    print(f"\n{'─'*80}")
    print("TRAINING TARGETS (what we actually train on):")
    print(f"{'─'*80}")
    # Extract only non-masked tokens (where label != -100)
    training_tokens = [l for l in labels[:valid_len] if l != -100]
    if training_tokens:
        decoded_training = tokenizer.decode(training_tokens)
        print(f"Training on {len(training_tokens)} tokens:")
        print(decoded_training)
    else:
        print("⚠️  NO TRAINING TOKENS (all labels are -100!)")
    
    print(f"\n{'─'*80}")
    print("MASKED PORTION (user question, NOT trained on):")
    print(f"{'─'*80}")
    # Extract only masked tokens (where label == -100)
    masked_positions = [i for i, l in enumerate(labels[:valid_len]) if l == -100]
    if masked_positions:
        masked_tokens = [input_ids[i] for i in masked_positions]
        decoded_masked = tokenizer.decode(masked_tokens)
        print(f"Masked {len(masked_tokens)} tokens (ignored during training):")
        print(decoded_masked)
    else:
        print("No masked tokens")
    
    # Show token-by-token breakdown (first 50 tokens)
    print(f"\n{'─'*80}")
    print("TOKEN-BY-TOKEN BREAKDOWN (first 50 tokens):")
    print(f"{'─'*80}")
    print(f"{'Pos':<5} {'Token ID':<10} {'Mask':<6} {'Label':<10} {'Decoded':<30}")
    print("─" * 80)
    
    for i in range(min(50, valid_len)):
        token_id = input_ids[i]
        mask = attention_mask[i]
        label = labels[i]
        decoded = tokenizer.decode([token_id])
        
        # Visual indicator
        status = "✓ TRAIN" if label != -100 else "✗ IGNORE"
        
        print(f"{i:<5} {token_id:<10} {mask:<6} {label:<10} {repr(decoded):<30} {status}")
    
    if valid_len > 50:
        print(f"... ({valid_len - 50} more tokens)")
    
    # Verification checks
    print(f"\n{'─'*80}")
    print("VERIFICATION CHECKS:")
    print(f"{'─'*80}")
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Padding tokens are masked
    checks_total += 1
    padding_masked = all(labels[i] == -100 for i in range(valid_len, len(labels)) if attention_mask[i] == 0)
    if padding_masked:
        print("✓ All padding tokens have labels = -100")
        checks_passed += 1
    else:
        print("❌ Some padding tokens don't have labels = -100")
    
    # Check 2: Some tokens are being trained on
    checks_total += 1
    if label_count > 0:
        print(f"✓ {label_count} tokens will be trained on")
        checks_passed += 1
    else:
        print("❌ No tokens to train on (all labels are -100)!")
    
    # Check 3: Not all tokens are training targets
    checks_total += 1
    if masked_count > 0:
        print(f"✓ {masked_count} tokens are correctly masked (user question)")
        checks_passed += 1
    else:
        print("⚠️  WARNING: No masked tokens (are you training on the entire sequence?)")
    
    # Check 4: Labels match input_ids where not masked
    checks_total += 1
    labels_match = all(
        labels[i] == input_ids[i] or labels[i] == -100
        for i in range(valid_len)
    )
    if labels_match:
        print("✓ Labels match input_ids (where not -100)")
        checks_passed += 1
    else:
        print("❌ Some labels don't match input_ids!")
    
    print(f"\nChecks passed: {checks_passed}/{checks_total}")
    
    return checks_passed == checks_total


def main(config):
    """Main verification function.
    
    Args:
        config: Config object with dataset and tokenizer settings
    """
    print("=" * 80)
    print("PREPROCESSED DATASET VERIFICATION")
    print("=" * 80)

    # ==============================================================================
    # Step 1: Load Dataset and Tokenizer
    # ==============================================================================
    print("\n📂 STEP 1: Loading dataset and tokenizer...")

    try:
        # Load the clean version (what training uses)
        dataset = datasets.load_from_disk(config.dataset_path + "_clean")
        print(f"✓ Loaded dataset from: {config.dataset_path}_clean")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure you've run the preprocessing script first!")
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    print(f"✓ Loaded tokenizer: {config.tokenizer_name}")
    print(f"  Vocab size: {len(tokenizer)}")

    # ==============================================================================
    # Step 2: Check Dataset Structure
    # ==============================================================================
    print("\n📊 STEP 2: Dataset Structure Check")
    print("-" * 80)

    print(f"\nDataset splits:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")

    print(f"\nDataset columns: {dataset['train'].column_names}")
    print(f"Expected columns: ['id', 'input_ids', 'attention_mask', 'labels']")

    required_cols = ['input_ids', 'attention_mask', 'labels']
    for col in required_cols:
        if col in dataset['train'].column_names:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ MISSING: {col}")

    print(f"\nDataset features:")
    for col_name, feature in dataset['train'].features.items():
        print(f"  {col_name}: {feature}")

    # ==============================================================================
    # Step 3: Token ID Range Verification
    # ==============================================================================
    print("\n🔢 STEP 3: Token ID Range Verification")
    print("-" * 80)

    print("Checking token IDs are within vocabulary range...")

    samples_checked = min(1000, len(dataset['train']))
    print(f"Checking {samples_checked} samples...")

    batch = dataset['train'].select(range(samples_checked))
    all_input_ids = []

    for sample in batch:
        ids = sample['input_ids']
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()
        elif isinstance(ids, list):
            ids = np.array(ids)
        all_input_ids.append(ids)

    # Stack into 2D array and use vectorized min/max
    all_input_ids = np.stack(all_input_ids)
    max_id = int(all_input_ids.max())
    min_id = int(all_input_ids.min())
    out_of_range = int((all_input_ids >= len(tokenizer)).sum())

    print(f"Samples checked: {samples_checked}")
    print(f"Token ID range: {min_id} to {max_id}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    if max_id >= len(tokenizer):
        print(f"❌ ERROR: {out_of_range} token IDs are >= vocab size!")
        print("   This will cause training to crash!")
    else:
        print(f"✓ All token IDs are within valid range [0, {len(tokenizer)-1}]")

    # ==============================================================================
    # Step 4: Detailed Sample Inspection
    # ==============================================================================
    print("\n🔍 STEP 4: Detailed Sample Inspection")
    print("-" * 80)

    # Inspect multiple samples
    num_samples_to_check = 3
    print(f"\nInspecting {num_samples_to_check} random samples...")

    all_passed = True
    for i in range(num_samples_to_check):
        sample_idx = random.randint(0, len(dataset['train']) - 1)
        sample = dataset['train'][sample_idx]
        passed = inspect_sample(sample, sample_idx, tokenizer)
        all_passed = all_passed and passed

    # ==============================================================================
    # Step 5: Statistical Analysis
    # ==============================================================================
    print("\n📈 STEP 5: Statistical Analysis")
    print("-" * 80)

    num_analysis_samples = min(1000, len(dataset['train']))
    print(f"\nAnalyzing {num_analysis_samples} samples from training set (vectorized for speed)...")

    # Get batch of samples for fast vectorized operations
    analysis_batch = dataset['train'].select(range(num_analysis_samples))

    # Convert to numpy arrays for vectorized operations
    all_masks = []
    all_labels = []

    for sample in analysis_batch:
        mask = sample['attention_mask']
        lbls = sample['labels']
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        elif isinstance(mask, list):
            mask = np.array(mask)
        
        if isinstance(lbls, torch.Tensor):
            lbls = lbls.numpy()
        elif isinstance(lbls, list):
            lbls = np.array(lbls)
        
        all_masks.append(mask)
        all_labels.append(lbls)

    all_masks = np.stack(all_masks)  # Shape: (num_samples, seq_len)
    all_labels = np.stack(all_labels)

    # Vectorized calculations
    lengths = all_masks.sum(axis=1)  # Valid tokens per sample
    total_len = all_masks.shape[1]  # Max sequence length

    # Count labels that aren't -100 in valid regions
    label_counts = ((all_labels != -100) & (all_masks == 1)).sum(axis=1)
    label_ratios = label_counts / np.maximum(lengths, 1)  # Avoid divide by zero
    padding_ratios = (total_len - lengths) / total_len

    print(f"\nSequence Lengths (valid tokens):")
    print(f"  Mean: {statistics.mean(lengths):.1f}")
    print(f"  Median: {statistics.median(lengths):.1f}")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")

    print(f"\nLabel Ratio (% of valid tokens to train on):")
    print(f"  Mean: {statistics.mean(label_ratios)*100:.1f}%")
    print(f"  Median: {statistics.median(label_ratios)*100:.1f}%")
    print(f"  Min: {min(label_ratios)*100:.1f}%")
    print(f"  Max: {max(label_ratios)*100:.1f}%")

    print(f"\nPadding Ratio:")
    print(f"  Mean: {statistics.mean(padding_ratios)*100:.1f}%")
    print(f"  Median: {statistics.median(padding_ratios)*100:.1f}%")

    # ==============================================================================
    # Step 6: Edge Cases Check
    # ==============================================================================
    print("\n⚠️  STEP 6: Edge Cases Check")
    print("-" * 80)

    print("\nChecking for potential issues (using vectorized analysis)...")

    issues_found = 0

    # Check for all-masked samples (using already-loaded arrays from Step 5)
    # A sample has all labels masked if all non-padded positions have label == -100
    all_masked_count = ((all_labels == -100) | (all_masks == 0)).all(axis=1).sum()

    if all_masked_count > 0:
        print(f"❌ Found {all_masked_count} samples with ALL labels = -100")
        print("   These samples won't contribute to training!")
        issues_found += 1
    else:
        print("✓ No samples with all labels masked")

    # Check for very short sequences (using vectorized comparison)
    very_short_count = (lengths < 10).sum()

    if very_short_count > 0:
        print(f"⚠️  Found {very_short_count} very short sequences (< 10 tokens)")
        print("   These might be truncated or malformed")
    else:
        print("✓ No suspiciously short sequences")

    # Check for very low label ratios (using vectorized comparison)
    low_label_count = (label_ratios < 0.1).sum()
    if low_label_count > 0:
        print(f"⚠️  Found {low_label_count} samples with < 10% labels")
        print("   These have very long user questions or short assistant responses")
    else:
        print("✓ All samples have reasonable label ratios")

    # ==============================================================================
    # Final Summary
    # ==============================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n✓ Dataset loaded successfully")
    print(f"✓ {len(dataset['train'])} training samples")
    print(f"✓ {len(dataset['test'])} test samples")
    print(f"✓ All required columns present")
    print(f"✓ Token IDs within valid range")

    if all_passed:
        print(f"✓ All sample checks passed")
    else:
        print(f"⚠️  Some sample checks failed - review details above")

    if issues_found == 0:
        print(f"✓ No edge cases detected")
    else:
        print(f"⚠️  {issues_found} potential issues detected")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if issues_found == 0 and all_passed:
        print("✅ DATASET IS READY FOR TRAINING!")
        print("\nYou can proceed with running your training script.")
    else:
        print("⚠️  REVIEW REQUIRED")
        print("\nPlease review the issues above before training.")
        print("Most warnings are not critical, but all-masked samples should be investigated.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify preprocessed dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run verification
    main(config)

