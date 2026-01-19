"""
Evaluation Benchmark Script
"""

import argparse
import subprocess
import sys
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

DEFAULT_TASKS = ["tulu_3_dev", "tulu_3_unseen"]


def _resolve_run_dir(config, run_dir_arg: str | None) -> str:
    """
    Prefer explicit --run-dir, then config.run_dir/output_run_dir, then output_dir.
    """
    if run_dir_arg:
        config.override_run_dir(run_dir_arg)

    # Config.override_run_dir sets config.run_dir and output.run_dir internally
    run_dir = getattr(config, "run_dir", None)
    if not run_dir:
        run_dir = getattr(config, "output_run_dir", None)
    if not run_dir:
        run_dir = getattr(config, "output_dir", None)

    if not run_dir:
        raise ValueError(
            "Could not resolve run_dir. Please set output.run_dir in config "
            "or pass --run-dir to run_experiment.py."
        )

    return str(run_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run OLMo benchmarks (Tülu-style) under EnergyTracker."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--run-dir", type=str, default=None, help="Override output.run_dir for this run")
    parser.add_argument(
        "--use-teacher",
        action="store_true",
        help="If set, use config.model.teacher instead of config.model.student as the eval model.",
    )
    parser.add_argument(
        "--eval-stage-name",
        type=str,
        default="olmo_benchmark",
        help="Name for the EnergyTracker stage (default: olmo_benchmark).",
    )

    # Anything we don't recognize here gets passed through to `olmes`
    args, olmes_extra = parser.parse_known_args()

    # Load experiment config
    config = load_config(args.config)
    run_dir = _resolve_run_dir(config, args.run_dir)

    # Decide which model to evaluate
    if args.use_teacher:
        model_name = config.teacher_model_name
    else:
        model_name = config.student_model_name

    if not model_name:
        raise ValueError(
            "No model name found for evaluation. "
            "Set model.student (and/or model.teacher) in your config, "
            "or extend this script to accept an explicit --model-name."
        )

    # Where OLMES should write its outputs
    run_dir_path = Path(run_dir)
    eval_output_dir = run_dir_path / "olmo_benchmark"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare EnergyTracker
    tracker = EnergyTracker(run_dir=run_dir, config=config)

    # Build OLMES command
    cmd = [
        "olmes",
        "--model",
        model_name,
        "--task",
        *DEFAULT_TASKS,
        "--output-dir",
        str(eval_output_dir),
    ]
    # Allow power users to pass extra flags directly to `olmes`
    if olmes_extra:
        cmd.extend(olmes_extra)

    print(f"[olmo_benchmark] Running model: {model_name}")
    print(f"[olmo_benchmark] Output dir: {eval_output_dir}")
    print(f"[olmo_benchmark] OLMES command: {' '.join(cmd)}")

    returncode = 1
    try:
        tracker.start_stage(args.eval_stage_name)
        # We don't currently count tokens here; EnergyTracker will still log energy.
        result = subprocess.run(cmd, cwd=str(run_dir_path), check=False)
        returncode = result.returncode
        if returncode != 0:
            print(f"[olmo_benchmark] OLMES exited with code {returncode}")
    finally:
        # tokens_processed left at default 0 for now; can be wired up later if desired.
        tracker.end_stage()
        tracker.save_summary()

    sys.exit(returncode)


if __name__ == "__main__":
    main()
