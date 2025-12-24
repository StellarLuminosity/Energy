import os
import torch
from datetime import datetime
import glob

# Model and dataset setup
seed = 42 # 16, 20, 32, 36, 40 default: 42
# teacher_model_name = "Qwen/Qwen2.5-7B-Instruct"
# student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# student_vocab_size = 151936 # 152064
# tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"

teacher_model_name = "allenai/OLMo-2-1124-7B-SFT"
student_model_name = "allenai/OLMo-2-0425-1B-SFT"
student_vocab_size = 100278 # 152064
tokenizer_name = "allenai/OLMo-2-1124-7B-SFT"

teacher_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
student_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

teacher_eval = (0.7968094515065487, 2.218451499938965)

# ---------------- Output paths -------------
base_output_dir = "/scratch/klambert/model_log/single_logs"
logprob_cache_path = "/home/klambert/projects/aip-craffel/shared/slm_ensemble/"
dataset_path = "/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized"
synthetic_dataset_path = "/scratch/klambert/dataset/synthetic_dataset"

# ---------------- Data --------------------
dataset_name = "allenai/tulu-3-sft-mixture"
clustered_dataset_name = "Malikeh1375/clustered_tulu_3_8"
dataset_type = "full"  # "single" or "batch" or "full"
synthetic_data = False

domains = {
    "programming_and_code_development",
    "qanda_and_logical_reasoning",
    "creative_writing_and_general_tasks",
    "multilingual_and_translation",
    "safety_and_harmful_content",
    "word_problems_and_arithmetic",
    "non-english_mathematics",
    "advanced_mathematics_and_modeling",
}

# ---------------- Run and hyper parameters - to change during every run -----------------
run_name = "OLMo2 SFT"
ddp = True
steps_per_round = -1
num_train_epochs = 4
learning_rate = 7.5e-6 # 5e-5 for constant
# If loss spikes in first 50–200 steps: drop to 5e-6.
# If loss is stable but barely decreasing: raise to 1.0e-5.
kl_temperature = 1
alpha = 0

resume_from_checkpoint = False
checkpointed_dir = None # <output_path> of the directory which stores the checkpoints from which to resume from 

# Ensembles
ensemble_random_init = False
ensemble_path = [] # ["/scratch/klambert/model_log/26-07-2025/run_2_alpha07_hyperparameters/round_0/checkpoint-14000"]  # Full path of ensemble models which we want to load (ex. ~/models/run2/round_1/checkpoint-18000)
total_rounds = 8 # number of ensemble models (how many are loaded + how many we want trained)

# ---------------- Early stopping parameters ---------------- 
early_stop_patience = 20        # number of evaluations with no improvement
early_stop_min_delta = 1e-6    # minimum absolute improvement in loss

# ---------------- Training args ----------------------------
weight_decay = 0.05
lr_scheduler_type = "cosine"
warmup_steps = 50
eval_steps = 40
logging_steps = 40
ckpt_save_steps = 500
save_total_limit = 2
per_device_train_batch_size = 1
eval_batch_size = 1
gradient_accumulation_steps = 16
max_grad_norm = 1.0
ignore_index = -100

# ---------------- Logging columns -------------------------- 
CSV_COLUMNS = [
    "function",
    "timestamp", 
    "overall_elapsed",
    "round_num", 
    "epoch_num", 
    "phase", 
    "role", 
    "step",
    "train_loss", 
    "train_kl_loss", 
    "train_next_token_loss", 
    "eval_loss", 
    "eval_kl_loss",
    "grad_norm",
    "learning_rate",
    "alpha",
]

def get_training_args(output_dir):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=False,
        report_to="wandb",
        hub_model_id=None,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        bf16=True,
        remove_unused_columns=False,
        max_steps=steps_per_round,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="no",
    )

