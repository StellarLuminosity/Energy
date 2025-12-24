import os
import torch
from dataclasses import dataclass

@dataclass
class DistillationConfig:
    # Model configurations
    teacher_model_name = "allenai/OLMo-2-1124-7B-SFT"
    student_model_name = "allenai/OLMo-2-0425-1B-SFT"
    student_vocab_size = 100352
    tokenizer_name = "allenai/OLMo-2-1124-7B-SFT" 

    # Dataset and Path
    # dataset_path = "/scratch/klambert/dataset/tulu-3-sft-mixture-olmo-preprocessed"
    logprob_cache_path = "/scratch/klambert/dataset/logprob_cache"
    dataset_path = "/scratch/klambert/dataset/logprob_cache/teacher_logprobs" # with teacher logits
    output_dir = "/scratch/klambert/model_log/singular/test_runs/alpha1"
    dataset_name = "allenai/tulu-3-sft-mixture"
    
    # Training parameters
    num_epochs: int = 2
    batch_size: int = 4
    eval_batch_size: int = 2
    learning_rate: float = 5e-5
    num_warmup_steps: int = 100
    num_training_steps: int = 0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 16 
    
    # Distillation parameters
    alpha: float = 1  # Weight for CE loss vs KL loss (0 = pure KL, 1 = pure CE)
    kl_temperature: float = 1.0  # Temperature for distillation
    
    # Checkpointing and logging
    save_steps: int = 200
    eval_steps: int = 100
    resume_from_checkpoint: bool = False
    
    # System
    seed: int = 42
    
    # Debug/Testing mode
    debug_mode: bool = False  # Set to True for quick testing
    debug_max_steps: int = 40  # Stop training after this many steps in debug mode

    # Wandb logging
    wandb_project: str = "slm-distillation-full-pipeline"
    wandb_run_name: str = None  # Auto-generated if None
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# Global config instance
config = DistillationConfig()
