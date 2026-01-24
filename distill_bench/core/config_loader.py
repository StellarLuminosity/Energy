"""
Configuration loader for distillation experiments.
Loads YAML configs with inheritance (base + experiment-specific).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration object with nested dict access via attributes."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

        # Flatten nested dicts for backward compatibility
        # e.g., config.training.batch_size → config.batch_size
        self._flatten_config()

    def _flatten_config(self):
        """Create flat attribute access for common fields."""
        # Training
        training = self._config.get("training", {})
        self.batch_size = training.get("batch_size", 4)
        self.eval_batch_size = training.get("eval_batch_size", 2)
        self.gradient_accumulation_steps = training.get("gradient_accumulation_steps", 16)
        self.learning_rate = training.get("learning_rate", 5e-5)
        self.max_grad_norm = training.get("max_grad_norm", 1.0)
        self.token_budget = training.get("token_budget", 0)
        self.optimizer = training.get("optimizer", "adamw")
        self.schedule_type = training.get("schedule_type", "cosine")
        self.dtype = training.get("dtype", "bfloat16")
        self.num_warmup_steps = training.get("num_warmup_steps", 100)
        self.num_epochs = training.get("num_epochs", 2)
        self.num_training_steps = training.get("num_training_steps", 0)
        self.save_steps = training.get("save_steps", 200)
        self.eval_steps = training.get("eval_steps", 100)

        # Experiment
        exp = self._config.get("experiment", {})
        self.seed = exp.get("seed", 42)
        self.debug_mode = exp.get("debug_mode", False)
        self.debug_max_steps = exp.get("debug_max_steps", 40)
        self.experiment_name = exp.get("name", "experiment")

        # Data
        data = self._config.get("data", {})
        self.dataset_name = data.get("dataset_name", "allenai/tulu-3-sft-mixture")
        self.dataset_path = data.get("dataset_path", "")
        self.dataset_subset = data.get("dataset_subset", None)
        self.dataset_split = data.get("dataset_split", None)
        self.dataset_choice = data.get("dataset_choice", "tulu")
        dataset_options = data.get("datasets", {})
        selected_dataset = dataset_options.get(self.dataset_choice, None)
        if selected_dataset:
            self.dataset_name = selected_dataset.get("dataset_name", self.dataset_name)
            self.dataset_path = selected_dataset.get("dataset_path", self.dataset_path)
            self.dataset_subset = selected_dataset.get("dataset_subset", self.dataset_subset)
            self.dataset_split = selected_dataset.get("dataset_split", self.dataset_split)
        self.tokenizer_name = data.get("tokenizer_name", "allenai/OLMo-2-1124-7B-SFT")
        self.max_sequence_length = data.get("max_sequence_length", 1024)
        self.pad_token_id = data.get("pad_token_id", -100)
        self.dataset_teacher_logprobs = data.get("dataset_teacher_logprobs", "")

        # Data preprocessing defaults
        preprocessing = self._config.get("preprocessing", {})
        self.num_samples = self._config.get("num_samples", preprocessing.get("num_samples", 0))
        self.test_size = self._config.get("test_size", preprocessing.get("test_size", 0.05))
        self.num_proc = self._config.get("num_proc", preprocessing.get("num_proc", 8))
        self.strip_think_blocks = self._config.get("strip_think_blocks", preprocessing.get("strip_think_blocks", False))
        self.code_only = self._config.get("code_only", preprocessing.get("code_only", False))

        # Models
        model = self._config.get("model", {})
        self.teacher_model_name = model.get("teacher", "")
        self.student_model_name = model.get("student", "")
        self.reference_model_name = model.get("reference", self.student_model_name)
        self.policy_model_name = model.get("policy", self.student_model_name)
        self.judge_model_name = model.get("judge", self.teacher_model_name)
        self.student_vocab_size = model.get("student_vocab_size", 100352)

        # Output
        output = self._config.get("output", {})
        self.output_dir = output.get("output_dir", "./outputs")
        self.output_run_dir = output.get("run_dir", None)
        self.checkpoint_dir = output.get("checkpoint_dir", None)

        # Benchmark / evaluation
        benchmark = self._config.get("benchmark", {})
        self.benchmark_output_dir = benchmark.get("output_dir", self.output_dir)
        self.model_type = benchmark.get("model_type", self.output_dir)
        self.benchmark_model = benchmark.get(
            "model",
            benchmark.get("model_name", None),
        )
        self.benchmark_tasks = benchmark.get("tasks", None)
        self.benchmark_name = benchmark.get("subfolder_name", None)
        self.mt_bench_path = benchmark.get("mt_bench_101").get("data_path", None)

        # Pipeline
        self.pipeline = self._config.get("pipeline", "kd")

        # Distillation (KD-specific)
        distill = self._config.get("distillation", {})
        self.alpha = distill.get("alpha", 0.5)
        self.temperature = distill.get("temperature", 1.0)
        self.logprob_cache_path = distill.get("logprob_cache_path", "")
        self.top_k_logits = distill.get("top_k_logits", None)
        self.resume_from_checkpoint = distill.get("resume_from_checkpoint", False)

        # Synthetic data (SFT-specific)
        synth = self._config.get("synthetic_data", {})
        gen = synth.get("generation", {})
        filt = synth.get("filtering", {})
        self.synthetic_temperature = gen.get("temperature", None)
        self.synthetic_top_p = gen.get("top_p", None)
        self.synthetic_max_new_tokens = gen.get("max_new_tokens", None)
        self.synthetic_decoding_strategy = gen.get("decoding_strategy", None)
        self.synthetic_prompt_field = gen.get("prompt_field", None)
        self.synthetic_num_samples = synth.get("num_samples", None)
        self.synthetic_max_gen_examples = synth.get("max_gen_examples", None)
        self.max_gen_examples = self.synthetic_max_gen_examples
        self.synthetic_filter_enabled = filt.get("enabled", None)
        self.synthetic_filter_min_length = filt.get("min_length", None)
        self.synthetic_filter_max_length = filt.get("max_length", None)
        self.synthetic_dataset_path = synth.get("synthetic_dataset_path", None)
        self.synthetic_use_existing = synth.get("use_existing", None)

        # DPO-specific
        dpo = self._config.get("dpo", {})
        self.dpo_beta = dpo.get("beta", 0.1)
        self.preference_dataset_path = dpo.get("preference_dataset_path", "")
        judge_cfg = dpo.get("judge_labeling", {})
        self.dpo_judge_enabled = judge_cfg.get("enabled", False)
        self.dpo_judge_temperature = judge_cfg.get("temperature", 0.7)
        if self.pipeline == "dpo":
            # Only override the main temperature when actually running the DPO pipeline.
            self.temperature = self.dpo_judge_temperature
        self.dpo_judge_top_p = judge_cfg.get("top_p", None)
        self.dpo_judge_max_new_tokens = judge_cfg.get("max_new_tokens", None)
        self.dpo_scoring_method = judge_cfg.get("scoring_method", "likelihood")
        candidate_cfg = dpo.get("candidate_generation", {})
        self.dpo_candidate_generation_enabled = candidate_cfg.get("enabled", False)
        self.dpo_num_candidates_per_prompt = candidate_cfg.get("num_candidates_per_prompt", 2)
        self.dpo_candidate_temperature = candidate_cfg.get("temperature", 0.8)

        # W&B
        wandb = self._config.get("wandb", {})
        self.wandb_enabled = wandb.get("enabled", True)
        self.wandb_project = wandb.get("project", "distillation-energy-benchmark")
        self.wandb_entity = wandb.get("entity", None)
        self.wandb_run_name = wandb.get("run_name", None)

        # Energy
        energy = self._config.get("energy", {})
        self.energy_enabled = energy.get("enabled", True)
        self.energy_nvml_poll_ms = energy.get("nvml_poll_interval_ms", 500)
        self.energy_track_cpu = energy.get("track_cpu", True)
        self.energy_country_iso = energy.get("country_iso_code", "USA")
        self.energy_offline_mode = energy.get("offline_mode", True)
        self.energy_rapl_root = energy.get("rapl_root", "/sys/class/powercap/intel-rapl")
        self.energy_total_policy = energy.get("total_energy_policy", "measured")

        # Hardware assertions (optional)
        hardware = self._config.get("hardware", {})
        self.hardware_assert_gpu_name = hardware.get("assert_gpu_name", None)
        self.hardware_assert_gpu_count = hardware.get("assert_gpu_count", None)
        self.hardware_assert_cpu_brand = hardware.get("assert_cpu_brand", None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self._config

    def __getattr__(self, name: str) -> Any:
        """Allow dict-style access for nested configs."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return self._config.get(name, None)

    def override_run_dir(self, run_dir: Optional[str]) -> None:
        """
        Override output.run_dir at runtime (e.g., via CLI flag).
        """
        if not run_dir:
            return

        if "output" not in self._config or not isinstance(self._config["output"], dict):
            self._config["output"] = {}
        self._config["output"]["run_dir"] = run_dir

        # Keep a top-level alias for getattr(..., "run_dir", None)
        self._config["run_dir"] = run_dir
        self.output_run_dir = run_dir
        self.run_dir = run_dir


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    experiment_config: str,
    base_config: Optional[str] = None,
) -> Config:
    """
    Load experiment config with base config inheritance.

    Args:
        experiment_config: Path to experiment YAML config
        base_config: Path to base YAML config (default: configs/base.yaml)

    Returns:
        Config object
    """
    # Resolve paths
    exp_path = Path(experiment_config)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    # Default base config location
    if base_config is None:
        base_config = exp_path.parent.parent / "base.yaml"
    else:
        base_config = Path(base_config)

    # Load configs
    config_dict = {}

    # Load base if exists
    if base_config.exists():
        config_dict = load_yaml(base_config)

    # Load and merge experiment config
    exp_dict = load_yaml(exp_path)
    config_dict = merge_configs(config_dict, exp_dict)

    return Config(config_dict)


def validate_config(config: Config) -> Dict[str, Any]:
    """
    Validate that required config fields exist.

    Args:
        config: Config object

    Returns:
        Dictionary with validation results
    """
    required_fields = {
        "pipeline": ["kd", "sft", "dpo"],
        "teacher_model_name": None,
        "student_model_name": None,
        "dataset_path": None,
        "output_dir": None,
        "batch_size": None,
        "learning_rate": None,
    }

    errors = []
    warnings = []

    for field, allowed_values in required_fields.items():
        value = getattr(config, field, None)

        if value is None or value == "":
            errors.append(f"Missing required field: {field}")
        elif allowed_values and value not in allowed_values:
            errors.append(f"Invalid value for {field}: {value}. Must be one of {allowed_values}")

    # Check output dir is writable
    output_dir = Path(config.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory: {e}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
