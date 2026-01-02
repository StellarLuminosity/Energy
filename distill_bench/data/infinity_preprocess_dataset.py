import torch
import random
import os
import shutil
import datasets
from transformers import AutoTokenizer
import torch.nn.functional as F

from distill_bench.core.config_loader import load_config

# Load config
config = load_config('configs/experiments/sft_7b_to_1b.yaml')

# ----------------------------------
# Helper Functions
# ----------------------------------
def compute_assistant_prefix_ids(tokenizer):
    """
    Automatically compute assistant prefix from tokenizer's chat template.
    This makes the code work with ANY chat template (OLMo, Llama, Mistral, etc.)
    """
    msgs = [{"role": "user", "content": "hi"}]
    
    # Tokenize WITH assistant prompt added
    with_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    
    # Tokenize WITHOUT assistant prompt
    without_prompt = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    
    # Normalize to lists (handle both tensor and list return types)
    wp = with_prompt[0].tolist() if hasattr(with_prompt, "shape") else with_prompt
    wop = without_prompt[0].tolist() if hasattr(without_prompt, "shape") else without_prompt
    
    # The difference is the assistant prefix!
    prefix = wp[len(wop):]
    
    if not prefix:
        print("WARNING: assistant prefix is empty—chat template may not add a header for the assistant.")
    
    return prefix


def convert_conversations_to_messages(sample):
    """Convert Infinity-Instruct format to standard messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in sample["conversations"]:
        # Map "human" -> "user", "gpt" -> "assistant", handle others gracefully
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        messages.append({"role": role, "content": turn.get("value", "")})
    sample["messages"] = messages
    return sample


def is_single_turn_qa(sample):
    """
    Keep only [human, gpt] two-turn conversations.
    This makes labeling simpler and avoids edge cases with multi-turn dialogues.
    """
    conv = sample.get("conversations", [])
    if len(conv) != 2:
        # print("WARNING: conversation is not a single-turn Q&A")
        return False
    return conv[0].get("from") == "human" and conv[1].get("from") == "gpt"


def create_response_labels(sample, assistant_prefix_ids):
    """
    Create labels that mask everything except the assistant's response.
    
    WHY WE DO THIS:
    During training, we only want the model to learn to generate the assistant's
    response, NOT the user's question. We achieve this by:
    1. Setting user question tokens to -100 (ignored by loss function)
    2. Keeping assistant response tokens as-is (these get trained on)
    
    EXAMPLE:
    Input:  [<user>, How, are, you, ?, <assistant>, I'm, fine, thanks]
    Labels: [-100,  -100, -100, -100, -100, -100,    I'm, fine, thanks]
            ↑____________________________↑   ↑___________________↑
                  Ignored during training    Actually trained on
    
    PARAMETERS:
    - sample: Dict with "input_ids" and "attention_mask"
    - assistant_prefix_ids: List of token IDs that mark assistant's turn
                            (computed by compute_assistant_prefix_ids)
    """
    # Step 1: Ensure input_ids is a PyTorch tensor
    if not isinstance(sample["input_ids"], torch.Tensor):
        sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long)

    # Step 2: Ensure attention_mask is also a tensor
    if not isinstance(sample["attention_mask"], torch.Tensor):
        sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long)

    # Step 3: Get the tensors we'll work with
    input_ids = sample["input_ids"]  # The actual token IDs
    attn = sample["attention_mask"]  # 1 = real token, 0 = padding
    
    # Step 4: Create labels array, initially all -100 (all ignored)
    labels = input_ids.clone()  # Copy input_ids
    labels.fill_(-100)  # Fill with -100 = "ignore this token"

    # Step 5: Use the computed assistant prefix (NOT hardcoded "<|assistant|>\n"!)
    # This is the list of token IDs we're searching for
    response_ids = assistant_prefix_ids
    start_pos = -1  # Will hold where assistant's response starts
    L = len(response_ids)  # Length of prefix (usually 1-3 tokens)
    
    # Step 6: Find where REAL tokens end (ignore padding)
    # IMPORTANT: Only search in the valid (non-padded) region!
    # 
    # Example attention_mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    #                          ↑___________↑  ↑____________↑
    #                           Real tokens      Padding
    # 
    # attn.nonzero() finds indices where mask == 1 (real tokens)
    # .max() gets the last real token position
    # +1 because we want length, not index
    valid_len = attn.nonzero(as_tuple=True)[0].max().item() + 1 if attn.sum() > 0 else len(input_ids)
    
    # Step 7: Extract only the valid (non-padded) part as a list for searching
    seq = input_ids[:valid_len].tolist()
    
    # Step 8: Search for the assistant prefix in the VALID region only
    # This is like searching for a pattern in a string
    # We check every position to see if the prefix matches
    for i in range(0, max(0, valid_len - L + 1)):
        # Check if tokens at position i match the prefix
        if seq[i:i + L] == response_ids:
            start_pos = i + L  # Start AFTER the prefix (we don't train on "<|assistant|>")
            break  # Found it! Stop searching
    
    # Step 9: Handle case where prefix wasn't found (truncated sample)
    if start_pos == -1:
        # No assistant header found -> keep all -100
        # This sample will be filtered out later
        return labels
    
    # Step 10: Set labels for the assistant's response
    # Everything from start_pos to end of valid tokens should be trained on
    end_pos = valid_len
    labels[start_pos:end_pos] = input_ids[start_pos:end_pos]
    
    # Step 11: Make absolutely sure padding tokens are -100
    # (In case something went wrong above)
    labels = labels.masked_fill(attn == 0, -100)

    return labels  # Returns tensor like: [-100, -100, -100, 345, 678, 901, -100, -100]

def format_chat_data(sample):
    return {"chat_text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}


def tokenize_text(sample):
    """
    Convert chat text to token IDs.
    
    IMPORTANT: add_special_tokens=False
    WHY: The chat template ALREADY added special tokens (BOS/EOS/etc) to the text.
    If we set add_special_tokens=True here, we'd add them AGAIN = duplicates!
    
    EXAMPLE OF THE PROBLEM:
    chat_text = "<BOS><|user|>\nHi<|assistant|>\nHello<EOS>"  (already has <BOS> and <EOS>)
    
    If add_special_tokens=True:
        Result: [BOS, BOS, user, Hi, assistant, Hello, EOS, EOS]  ← WRONG! Duplicates!
    
    If add_special_tokens=False:
        Result: [BOS, user, Hi, assistant, Hello, EOS]  ← CORRECT!
    
    PARAMETERS:
    - truncation=True: Cut off at max_length if too long
    - padding="max_length": Pad short sequences to max_length with 0s
    - max_length=1024: Maximum sequence length
    - return_tensors="pt": Return PyTorch tensors (not lists)
    """
    tokenized = tokenizer(
        sample["chat_text"],
        truncation=True,           # Cut off long sequences
        padding="max_length",      # Pad short sequences to 1024
        max_length=1024,           # Max length
        add_special_tokens=False,  # DON'T duplicate special tokens!
        return_tensors="pt",       # Return PyTorch tensors
    )

    # squeeze(0) removes the batch dimension: shape [1, 1024] → [1024]
    sample["input_ids"] = tokenized["input_ids"].squeeze(0)
    sample["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    return sample


def contains_assistant_prefix(sample, assistant_prefix_ids):
    """
    Check if the example contains the assistant prefix in the valid (non-padded) region.
    Used to filter out truncated samples.
    
    WHY WE NEED THIS:
    Some examples get truncated to max_length (1024 tokens). If truncation happens
    in the middle of the conversation, the assistant's response might be cut off
    or the assistant marker might be missing entirely. We filter these out because:
    1. They can't be properly labeled (no assistant marker = no labels)
    2. Training on incomplete responses is bad for model quality
    
    WHAT WE'RE CHECKING:
    Does this sample have the assistant prefix token(s) in the non-padded part?
    
    PARAMETERS:
    - sample: Dict with "input_ids" and "attention_mask"
    - assistant_prefix_ids: List of token IDs for assistant marker
    
    RETURNS:
    - True: Sample is good, has assistant prefix
    - False: Sample is bad, missing prefix (will be filtered out)
    """
    # Step 1: Convert tensors to lists for easier searching
    # (Lists are easier to work with for comparisons)
    ids = sample["input_ids"].tolist() if isinstance(sample["input_ids"], torch.Tensor) else sample["input_ids"]
    attn = sample["attention_mask"].tolist() if isinstance(sample["attention_mask"], torch.Tensor) else sample["attention_mask"]
    
    # Step 2: Find where the valid (non-padded) region ends
    # Generator expression: (i for i, m in enumerate(attn) if m == 1)
    # This finds all indices where attention_mask == 1 (real tokens)
    # max(...) gets the last one
    # default=-1 handles edge case where there are NO real tokens
    valid_len = max((i for i, m in enumerate(attn) if m == 1), default=-1) + 1
    
    # Step 3: Edge case - if no valid tokens at all, reject this sample
    if valid_len <= 0:
        return False
    
    # Step 4: Search for assistant prefix in the VALID region only
    L = len(assistant_prefix_ids)  # Usually 1-3 tokens
    
    # Loop through each possible starting position
    for i in range(0, max(0, valid_len - L + 1)):
        # Check if the tokens at position i match the prefix
        # Example: if prefix is [100279] and ids[5:6] is [100279], match!
        if ids[i:i + L] == assistant_prefix_ids:
            return True  # Found it! This is a good sample
    
    # Step 5: Didn't find the prefix anywhere - bad sample
    return False  # Filter this out


def main():
    """Main preprocessing function."""
    # ----------------------------------
    # Load Dataset
    # ----------------------------------
    print("\n=== LOADING DATASET ===")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    print(f"Tokenizer loaded: {config.tokenizer_name}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Verify this is OLMo tokenizer (should be ~100k tokens)
    if len(tokenizer) < 90000 or len(tokenizer) > 110000:
        print(f"   WARNING: Tokenizer vocab size ({len(tokenizer)}) seems unusual for OLMo!")
        print(f"   Expected ~100,278 tokens for OLMo-2 models")
        print(f"   Are you sure you're using the right tokenizer?")
    else:
        print(f"✓ Tokenizer vocab size looks correct for OLMo")

    # ============================================================================
    # CRITICAL STEP: Compute assistant prefix IDs ONCE (model-agnostic approach!)
    # ============================================================================
    # WHY WE DO THIS HERE:
    # - We compute the prefix ONCE at the start (not 200,000 times later!)
    # - This makes the code work with ANY tokenizer (OLMo, Llama, Mistral, etc.)
    # - We print it so you can verify it's detecting the right tokens
    #
    # WHAT YOU'LL SEE:
    # For OLMo: Assistant prefix token IDs: [100279]
    #           Decoded prefix: '<|assistant|>\n'
    # For Llama: Assistant prefix token IDs: [128006, 78191, 128007]  (different!)
    #
    # This prefix will be passed to:
    # 1. create_response_labels() - to find where assistant's answer starts
    # 2. contains_assistant_prefix() - to filter truncated samples
    print("\n=== COMPUTING ASSISTANT PREFIX ===")
    assistant_prefix_ids = compute_assistant_prefix_ids(tokenizer)
    print(f"Assistant prefix token IDs: {assistant_prefix_ids}")
    print(f"Decoded prefix: {repr(tokenizer.decode(assistant_prefix_ids))}")

    dataset = datasets.load_dataset(config.dataset_name, "Gen", split="train")

    print(f"Original dataset size: {len(dataset)}")
    print(f"Original dataset features: {dataset.features}")

    print(f"Example raw conversation format:")
    print(dataset[random.randint(0, len(dataset) - 1)]["conversations"])
    print(f"Another example raw conversation format:")
    print(dataset[random.randint(0, len(dataset) - 1)]["conversations"])

    # ----------------------------------
    # Shuffle and Sample Dataset
    # ----------------------------------
    dataset = dataset.shuffle(config.seed)
    dataset = dataset.select(range(min(200_000, len(dataset))))
    dataset = dataset.train_test_split(test_size=2000)
    print(f"\nAfter sampling - Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # ----------------------------------
    # Filter to Single-Turn Q&A (optional but recommended)
    # ----------------------------------
    print("\n=== FILTERING TO SINGLE-TURN Q&A ===")
    print("This keeps only [human, gpt] two-turn conversations for cleaner training...")
    print(f"Before single-turn filter - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    dataset['train'] = dataset['train'].filter(is_single_turn_qa, num_proc=32)
    dataset['test'] = dataset['test'].filter(is_single_turn_qa, num_proc=32)
    print(f"After single-turn filter - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    print("\n=== CONVERTING CONVERSATIONS TO MESSAGES ===")
    dataset = dataset.map(convert_conversations_to_messages, num_proc=32)

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
    tokenized_dataset = processed_dataset.map(lambda x: tokenize_text(x, tokenizer), remove_columns=["messages", "conversations", "label", "langdetect", "source"], num_proc=32)
    print(f"Dataset features after tokenization: {tokenized_dataset['train'].features}")

    print(f"Train example input_ids shape: {torch.tensor(tokenized_dataset['train'][0]['input_ids']).shape}")
    print(f"Train example attention_mask shape: {torch.tensor(tokenized_dataset['train'][0]['attention_mask']).shape}")
    print(f"Train example id: {tokenized_dataset['train'][0]['id']}")

    print("\n=== ADDING LABELS (masking everything except assistant response) ===")
    # ============================================================================
    # Create training labels by masking the user's question
    # ============================================================================
    # WHAT THIS DOES:
    # For each sample, we call create_response_labels() which:
    # 1. Searches for the assistant_prefix_ids in the tokenized sequence
    # 2. Sets all tokens BEFORE the assistant's response to -100 (ignored)
    # 3. Keeps tokens in the assistant's response as-is (trained on)
    #
    # WHY USE A LAMBDA:
    # We need to pass assistant_prefix_ids to the function, but map() can only
    # pass the sample itself. So we wrap it in a lambda (anonymous function):
    #   lambda s: creates a function that takes sample 's'
    #   {"labels": ...}: returns a dict with the new "labels" field
    #   create_response_labels(s, assistant_prefix_ids): calls our function
    #
    # EXAMPLE TRANSFORMATION:
    # Before: {"input_ids": [1,2,3,4,5,6,7,8], "attention_mask": [1,1,1,1,1,1,1,0]}
    # After:  {"input_ids": [1,2,3,4,5,6,7,8], "labels": [-100,-100,-100,4,5,6,7,-100]}
    #                                                       ↑_______↑ user Q   ↑___↑ response
    labeled_dataset = tokenized_dataset.map(
        lambda s: {"labels": create_response_labels(s, assistant_prefix_ids)}, 
        num_proc=32  # Process 32 samples in parallel for speed
    )
    print(f"Dataset features after adding labels: {labeled_dataset['train'].features}")
    print(f"ID column preserved - example id: {labeled_dataset['train'][0]['id']}")

    # ============================================================================
    # Filter out samples which were truncated
    # ============================================================================
    print("\n=== FILTERING (removing truncated samples without assistant prefix) ===")
    # WHY WE FILTER:
    # Some examples are too long and get truncated to max_length (1024 tokens).
    # If the truncation cuts off the assistant's response or removes the assistant
    # marker entirely, we can't train on it properly. Better to remove it.
    #
    # WHAT WE'RE CHECKING:
    # Does the sample contain assistant_prefix_ids in the valid (non-padded) region?
    # - YES: Keep it (has assistant response, can be trained on)
    # - NO: Remove it (truncated or malformed, can't be labeled properly)
    #
    # HOW IT WORKS:
    # .filter() keeps only samples where the lambda function returns True
    # lambda x: creates a function that takes sample 'x'
    # contains_assistant_prefix(x, assistant_prefix_ids): checks if prefix exists
    #
    # EXAMPLE:
    # Good sample: [user_tokens, ASSISTANT_PREFIX, response_tokens, padding] → KEEP
    # Bad sample:  [user_tokens, user_tokens, user_tokens, padding] → REMOVE (truncated)
    #
    # TYPICAL RESULTS:
    # Usually keeps ~95-98% of samples (only removes severely truncated ones)

    labeled_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "id"])
    final_dataset = labeled_dataset.filter(
        lambda x: contains_assistant_prefix(x, assistant_prefix_ids), 
        num_proc=32  # Process 32 samples in parallel
    )
    print(f"Kept -> Train: {len(final_dataset['train'])} | Test: {len(final_dataset['test'])}")
    print(f"Filtered out ~{100 * (1 - len(final_dataset['train']) / len(labeled_dataset['train'])):.1f}% of train samples")

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
    clean_dataset = final_dataset.remove_columns(['chat_text', 'reward'])  # Keep id for potential future use
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


# Helper functions need tokenizer parameter
def format_chat_data(sample, tokenizer):
    return {"chat_text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}


def tokenize_text(sample, tokenizer):
    """
    Convert chat text to token IDs.
    
    IMPORTANT: add_special_tokens=False
    WHY: The chat template ALREADY added special tokens (BOS/EOS/etc) to the text.
    If we set add_special_tokens=True here, we'd add them AGAIN = duplicates!
    
    EXAMPLE OF THE PROBLEM:
    chat_text = "<BOS><|user|>\nHi<|assistant|>\nHello<EOS>"  (already has <BOS> and <EOS>)
    
    If add_special_tokens=True:
        Result: [BOS, BOS, user, Hi, assistant, Hello, EOS, EOS]  ← WRONG! Duplicates!
    
    If add_special_tokens=False:
        Result: [BOS, user, Hi, assistant, Hello, EOS]  ← CORRECT!
    
    PARAMETERS:
    - truncation=True: Cut off at max_length if too long
    - padding="max_length": Pad short sequences to max_length with 0s
    - max_length=1024: Maximum sequence length
    - return_tensors="pt": Return PyTorch tensors (not lists)
    """
    tokenized = tokenizer(
        sample["chat_text"],
        truncation=True,           # Cut off long sequences
        padding="max_length",      # Pad short sequences to 1024
        max_length=1024,           # Max length
        add_special_tokens=False,  # DON'T duplicate special tokens!
        return_tensors="pt",       # Return PyTorch tensors
    )

    # squeeze(0) removes the batch dimension: shape [1, 1024] → [1024]
    sample["input_ids"] = tokenized["input_ids"].squeeze(0)
    sample["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    return sample


if __name__ == "__main__":
    main()

