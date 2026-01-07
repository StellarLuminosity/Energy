#!/usr/bin/env python3
"""
Unified experiment launcher for distillation pipelines.

Dispatches to KD/SFT/DPO based on config.pipeline field.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.utils import _resolve_data_script


def main():
    parser = argparse.ArgumentParser(
        description="Run distillation experiment or data script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--data-script",
        type=str,
        default=None,
        help="Optional data script to run instead of a training pipeline (e.g., logit_caching)",
    )
    args, extra = parser.parse_known_args()

    # Load config to determine pipeline type
    config = load_config(args.config)

    # If a data script is specified, run it and exit
    if args.data_script:
        script_path = _resolve_data_script(args.data_script)
        cmd = [sys.executable, str(script_path), "--config", args.config, *extra]
        print(f"Running data script: {script_path}")
        print(f"With command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    pipeline = config.pipeline

    print(f"Running pipeline: {pipeline}")
    print(f"Config: {args.config}")
    if extra:
        print(f"Extra args (ignored by pipeline dispatcher): {extra}")
    print()

    # Dispatch to appropriate pipeline
    if pipeline == "kd":
        from distill_bench.pipelines import kd_main

        kd_args = argparse.Namespace(config=args.config)
        kd_main.main(kd_args)

    elif pipeline == "sft":
        from distill_bench.pipelines import sft_main

        sft_main.main(args)

    elif pipeline == "dpo":
        from distill_bench.pipelines import dpo_main

        dpo_main.main(args)

    else:
        print(f"Error: Unknown pipeline type '{pipeline}'")
        print(f"Expected one of: kd, sft, dpo")
        sys.exit(1)


if __name__ == "__main__":
    main()
