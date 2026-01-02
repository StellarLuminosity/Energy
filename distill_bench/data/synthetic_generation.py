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
from distill_bench.core.config_loader import Config


def generate_synthetic_dataset(
    config: Config,
    energy_tracker: Optional[EnergyTracker] = None,
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
    
    # Get generation config
    gen_config = config.get('synthetic_data.generation', {})
    temperature = gen_config.get('temperature', 0.7)
    top_p = gen_config.get('top_p', 0.9)
    max_new_tokens = gen_config.get('max_new_tokens', 512)
    decoding_strategy = gen_config.get('decoding_strategy', 'sampling')
    
    # Get dataset config
    prompt_dataset_name = config.get('synthetic_data.prompt_dataset', config.dataset_name)
    num_samples = config.get('synthetic_data.num_samples', 50000)
    prompt_field = config.get('synthetic_data.prompt_field', 'auto')  # 'auto', 'messages', or 'prompt'
    
    # Get paths
    synthetic_path = config.get('synthetic_data.synthetic_dataset_path')
    
    # Start energy tracking for teacher generation (single stage)
    if energy_tracker and energy_tracker.current_stage is None:
        energy_tracker.start_stage("teacher_generation")
    
    # Load tokenizer and teacher model
    print(f"Loading teacher model: {config.teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    teacher_model.eval()
    
    # Load prompt dataset
    print(f"Loading prompt dataset: {prompt_dataset_name}")
    prompt_dataset = datasets.load_dataset(prompt_dataset_name, split="train")
    
    # Sample prompts
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
                # Determine prompt field to use
                if prompt_field == 'auto':
                    # Auto-detect: prefer 'messages', fallback to 'prompt'
                    if 'messages' in example and example['messages']:
                        use_messages = True
                        use_prompt = False
                    elif 'prompt' in example and example['prompt']:
                        use_messages = False
                        use_prompt = True
                    else:
                        continue  # No valid field found
                elif prompt_field == 'messages':
                    use_messages = True
                    use_prompt = False
                else:  # prompt_field == 'prompt'
                    use_messages = False
                    use_prompt = True
                
                # Extract or format prompt
                if use_messages:
                    # Extract user message(s) from messages field
                    messages = example.get("messages", [])
                    user_messages = [m for m in messages if m.get("role") == "user"]
                    if not user_messages:
                        continue
                    
                    # Apply chat template
                    prompt_text = tokenizer.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                elif use_prompt:
                    # Use pre-formatted prompt field
                    prompt_text = example.get("prompt", "")
                    if not prompt_text:
                        continue
                    
                    # Apply chat template if needed (wrap as user message)
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # No valid prompt field found
                    continue
                
                # Tokenize prompt (no truncation - we want full prompt)
                prompt_inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=False,
                ).to(device)
                
                prompt_length = prompt_inputs['input_ids'].shape[1]
                
                # Skip if prompt itself is too long
                max_seq_len = config.get('data.max_sequence_length', 1024)
                if prompt_length >= max_seq_len - 10:  # Need space for at least short response
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
                    **prompt_inputs,
                    **generation_kwargs,
                )
                
                # Full generated sequence (prompt + response)
                full_sequence = outputs[0]
                generated_tokens = full_sequence[prompt_length:]
                
                # Create labels: mask prompt (-100), keep response
                labels = torch.full_like(full_sequence, fill_value=-100)
                labels[prompt_length:] = full_sequence[prompt_length:]
                
                # Apply filtering on response length and max length
                filtering_config = config.get('synthetic_data.filtering', {})
                if filtering_config.get('enabled', True):
                    min_length = filtering_config.get('min_length', 10)
                    max_length = filtering_config.get('max_length', max_seq_len)
                    response_length = len(generated_tokens)
                    total_length = len(full_sequence)
                    if response_length < min_length:
                        print(f"Response length shorter than min length - skipping idx {idx}")
                        continue
                    if total_length > max_length:
                        print(f"Total length (prompt + response) is greater than max length - skipping idx {idx}")
                        continue
                
                # Create attention mask (all 1s for valid sequence)
                attention_mask = torch.ones_like(full_sequence)
                
                # Store
                synthetic_data["input_ids"].append(full_sequence.cpu().tolist())
                synthetic_data["attention_mask"].append(attention_mask.cpu().tolist())
                synthetic_data["labels"].append(labels.cpu().tolist())
                
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


def load_or_generate_synthetic_dataset(
    config: Config,
    energy_tracker: Optional[EnergyTracker] = None,
) -> datasets.DatasetDict:
    """
    Load existing synthetic dataset or generate if it doesn't exist.
    
    Args:
        config: Configuration object
        energy_tracker: Optional energy tracker
    
    Returns:
        DatasetDict with synthetic data
    """
    synthetic_path = config.get('synthetic_data.synthetic_dataset_path')
    
    # Check if exists
    if synthetic_path and Path(synthetic_path).exists():
        print(f"Loading existing synthetic dataset from: {synthetic_path}")
        return datasets.load_from_disk(synthetic_path)
    
    # Generate new dataset
    print("Synthetic dataset not found, generating...")
    return generate_synthetic_dataset(config, energy_tracker)
