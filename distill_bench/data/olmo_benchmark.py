"""
Evaluation Benchmark Script
"""
import os
import torch
import argparse
import subprocess
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

# Name of this benchmark script (used for directory names)
BENCHMARK_NAME = "olmo_benchmark"

# Default Tülu-style eval suite; adjust if you want a smaller set.
DEFAULT_TASKS = ["tulu_3_dev", "tulu_3_unseen"]


def _resolve_benchmark_run_dir(config, run_dir_arg: str | None) -> Path:
    """
    Resolve the root directory where this benchmark run should live.

    Priority:
      1. --run-dir CLI argument (if provided)
      2. config.benchmark_output_dir
      3. config.output_dir as a last fallback

    The benchmark script name (BENCHMARK_NAME) is always appended.
    """
    if run_dir_arg:
        base = Path(run_dir_arg)
    else:
        base = Path(getattr(config, "benchmark_output_dir", None) or config.output_dir)

    # Final run dir is: <base>/<BENCHMARK_NAME>
    run_dir = base / BENCHMARK_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _maybe_convert_checkpoint_to_hf(
    model_spec: str,
    config,
    run_dir: Path,
    subdir_name: str = "hf_from_checkpoint",
) -> str:
    """
    If model_spec is a .pt checkpoint file, load the base student model,
    apply the checkpoint weights, and save a HF-format model into
        <run_dir>/<subdir_name>.
    Returns the path that should be given to OLMES as --model.

    Otherwise, returns model_spec unchanged.
    """
    # If it's not a file or doesn't end with .pt, just use it as-is.
    if not os.path.isfile(model_spec) or not model_spec.endswith(".pt"):
        return model_spec

    base_model_name = getattr(config, "student_model_name", None) or getattr(
        config, "student_model", None
    )
    if not base_model_name:
        raise ValueError(
            "benchmark.model points to a .pt checkpoint, but no base student_model_name "
            "is configured. Please set model.student in your YAML so we know "
            "which architecture to load before applying the checkpoint."
        )

    hf_dir = run_dir / subdir_name
    if hf_dir.exists():
        # Assume it has already been created in a previous run.
        print(f"[{BENCHMARK_NAME}] Using existing HF-format dir: {hf_dir}")
        return str(hf_dir)

    hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{BENCHMARK_NAME}] Converting checkpoint to HF format:")
    print(f"  base model: {base_model_name}")
    print(f"  checkpoint: {model_spec}")
    print(f"  output dir: {hf_dir}")

    # 1) Load base HF model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2) Load checkpoint
    ckpt = torch.load(model_spec, map_location="cpu")

    # Try a few common keys; adjust here if your training script uses a different layout.
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # If it looks like a plain state dict, just use it directly
            state_dict = ckpt
    else:
        raise ValueError(
            f"Checkpoint file {model_spec} is not a dict. "
            "Please adjust _maybe_convert_checkpoint_to_hf to match your checkpoint format."
        )

    # 3) Load weights into the model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[{BENCHMARK_NAME}] Warning: missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # 4) Save HF-format model + tokenizer
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)

    return str(hf_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run OLMo benchmarks (Tülu-style) under EnergyTracker."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override benchmark.output_dir base for this run "
             "(<run-dir>/olmo_benchmark will be used).",
    )
    parser.add_argument(
        "--eval-stage-name",
        type=str,
        default=BENCHMARK_NAME,
        help="Name for the EnergyTracker stage (default: olmo_benchmark).",
    )

    # Anything not consumed here will be forwarded to `olmes`.
    args, olmes_extra = parser.parse_known_args()

    # Load experiment config
    config = load_config(args.config)

    # Resolve where this benchmark should write its logs and results
    run_dir = _resolve_benchmark_run_dir(config, args.run_dir)

    # Decide which model to evaluate
    model_str = getattr(config, "benchmark_model", None)
    if not model_str:
        raise ValueError(
            "No benchmark.model configured. Please set benchmark.model in your YAML "
            "(either a Hugging Face id like 'allenai/OLMo-2-1124-7B-SFT' or a local "
            "HF-format directory like "
            "'/scratch/.../kd_32b_to_1b_adamw/final_model/hf_format')."
        )

    # NEW: if it's a .pt checkpoint, convert to HF dir under run_dir
    model_path_for_olmes = _maybe_convert_checkpoint_to_hf(
        model_spec=model_str,
        config=config,
        run_dir=run_dir,
        subdir_name="hf_from_checkpoint",
    )

    # OLMES output goes into the same run_dir
    eval_output_dir = run_dir
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare EnergyTracker
    tracker = EnergyTracker(run_dir=str(run_dir), config=config)

    # Build OLMES command
    cmd = [
        "olmes",
        "--model",
        model_str,
        "--task",
        *DEFAULT_TASKS,
        "--output-dir",
        str(eval_output_dir),
    ]

    # Allow power users to pass extra flags directly to `olmes`
    if olmes_extra:
        cmd.extend(olmes_extra)

    print(f"[{BENCHMARK_NAME}] Running model: {model_path_for_olmes}")
    print(f"[{BENCHMARK_NAME}] Run dir: {run_dir}")
    print(f"[{BENCHMARK_NAME}] OLMES command: {' '.join(cmd)}")

    returncode = 1
    try:
        tracker.start_stage(args.eval_stage_name)
        # EnergyTracker will still record GPU/CPU energy even without token counts.
        result = subprocess.run(cmd, check=False, cwd=str(run_dir))
        returncode = result.returncode
        if returncode != 0:
            print(f"[{BENCHMARK_NAME}] OLMES exited with code {returncode}")
    finally:
        tracker.end_stage()       # tokens_processed left as default 0
        tracker.save_summary()

    sys.exit(returncode)


if __name__ == "__main__":
    main()
