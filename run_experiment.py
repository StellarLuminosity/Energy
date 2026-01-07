#!/usr/bin/env python3
"""
Unified experiment launcher for distillation pipelines.

Dispatches to KD/SFT/DPO based on config.pipeline field.
"""

import argparse
import sys
from pathlib import Path

from distill_bench.core.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run distillation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--run-other",
        type=str,
        action="store_true",
        help="Run other script before launching the main pipeline",
    )
    args = parser.parse_args()

    # Load config to determine pipeline type
    config = load_config(args.config)
    pipeline = config.pipeline

    print(f"Running pipeline: {pipeline}")
    print(f"Config: {args.config}")
    print()

    # Dispatch to appropriate pipeline
    if args.run_other:
        str = "run_prerun"
        # Quick validation before the main run
        from distill_bench.core.prerun import quick_validation

        prerun_dir = Path(args.prerun_output) if args.prerun_output else Path(config.output_dir) / "prerun_validation"
        quick_ok = quick_validation(
            output_dir=str(prerun_dir),
            config=config,
        )
        if not quick_ok:
            print("Prerun validation failed; aborting.")
            sys.exit(1)

    elif pipeline == "kd":
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
