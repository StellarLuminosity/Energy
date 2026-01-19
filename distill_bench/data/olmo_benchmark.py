"""
Evaluation Benchmark Script
"""

import argparse
import subprocess
import sys
from pathlib import Path

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

    print(f"[{BENCHMARK_NAME}] Running model: {model_str}")
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
