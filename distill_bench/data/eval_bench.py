"""
Evaluation Benchmark Script
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker


def _get_run_dir(config, cli_run_dir: str | None) -> str:
    """
    Resolve run_dir, preferring the CLI override if given,
    then falling back to config.run_dir or output.run_dir.
    """
    if cli_run_dir:
        return cli_run_dir

    # Try attribute first (many configs expose run_dir as a property)
    run_dir = getattr(config, "run_dir", None)

    # Fallback: dotted key lookup (like energy_logger._cfg does)
    if not run_dir and hasattr(config, "get"):
        run_dir = config.get("output.run_dir", None)

    if not run_dir:
        raise ValueError(
            "eval_benchmark requires either --run-dir or config.output.run_dir/config.run_dir to be set."
        )
    return str(run_dir)


def _cfg(config, key: str, default=None):
    """
    Helper to read dotted keys from the config, similar to EnergyTracker.
    """
    if hasattr(config, "get"):
        val = config.get(key, None)
        if val is not None:
            return val
    return default


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run OLMES benchmark suite under EnergyTracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override output.run_dir / config.run_dir for this eval run",
    )
    args, extra = parser.parse_known_args(argv)

    # Load config
    config = load_config(args.config)

    # Respect run_dir override if the config supports it (same pattern as run_experiment.py)
    if args.run_dir and hasattr(config, "override_run_dir"):
        config.override_run_dir(args.run_dir)

    run_dir = _get_run_dir(config, args.run_dir)
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Read eval configuration
    # ----------------------------
    # Required:
    #   eval.model_name: HF id or local path for the model to evaluate
    #   eval.tasks: list of OLMES task names (e.g., ["tulu_3_dev", "tulu_3_unseen"])
    #
    # Optional:
    #   eval.engine: "olmes" (only supported for now, default)
    #   eval.model_type: e.g. "hf" or "vllm"
    #   eval.model_args: dict or JSON string for --model-args
    #   eval.output_subdir: subdirectory under run_dir for OLMES output (default: "eval")
    model_name = _cfg(config, "eval.model_name")
    tasks = _cfg(config, "eval.tasks")
    engine = _cfg(config, "eval.engine", "olmes")
    model_type = _cfg(config, "eval.model_type", None)
    model_args = _cfg(config, "eval.model_args", None)
    output_subdir = _cfg(config, "eval.output_subdir", "eval")

    if engine != "olmes":
        raise ValueError(f"Currently only eval.engine='olmes' is supported, got {engine!r}")

    if not model_name:
        raise ValueError(
            "Missing eval.model_name in config. "
            "Set this to either a Hugging Face repo id (e.g. 'allenai/OLMo-2-0325-32B-SFT') "
            "or a local checkpoint path."
        )

    if not tasks:
        raise ValueError(
            "Missing eval.tasks in config. "
            "Set this to a list of OLMES task names, e.g. ['tulu_3_dev', 'tulu_3_unseen']."
        )

    if isinstance(tasks, str):
        tasks_list = [tasks]
    else:
        tasks_list = list(tasks)

    eval_output_dir = run_dir_path / output_subdir
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build OLMES command
    # ----------------------------
    cmd = [
        "olmes",
        "--model",
        str(model_name),
        "--output-dir",
        str(eval_output_dir),
        "--task",
        *tasks_list,
    ]

    if model_type:
        cmd += ["--model-type", str(model_type)]

    if model_args:
        # allow either a dict or a raw JSON string in the config
        if isinstance(model_args, str):
            cmd += ["--model-args", model_args]
        else:
            cmd += ["--model-args", json.dumps(model_args)]

    # If any extra args were passed via run_experiment, forward them to OLMES
    if extra:
        cmd += list(extra)

    print("[eval_benchmark] Run directory:", run_dir_path)
    print("[eval_benchmark] OLMES output dir:", eval_output_dir)
    print("[eval_benchmark] Model:", model_name)
    print("[eval_benchmark] Tasks:", tasks_list)
    print("[eval_benchmark] Command:")
    print("  " + " ".join(cmd))
    print()

    # ----------------------------
    # Energy tracking
    # ----------------------------
    tracker = EnergyTracker(
        run_dir=run_dir,
        experiment_name="eval",
        config=config,
    )

    # Single stage for now; you can split into multiple later if needed
    tracker.start_stage("eval_benchmark")

    result = subprocess.run(cmd)

    # We don't have token counts from OLMES yet, so tokens_processed=0
    tracker.end_stage(tokens_processed=0)
    tracker.save_summary(
        additional_metadata={
            "eval": {
                "model_name": model_name,
                "tasks": tasks_list,
                "engine": engine,
                "model_type": model_type,
                "output_subdir": str(output_subdir),
            },
            "olmes_exit_code": result.returncode,
        }
    )

    if result.returncode != 0:
        print(f"[eval_benchmark] OLMES exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
