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
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Use mixed precision training (KD only)"
    )
    
    args = parser.parse_args()
    
    # Load config to determine pipeline type
    config = load_config(args.config)
    pipeline = config.pipeline
    
    print(f"Running pipeline: {pipeline}")
    print(f"Config: {args.config}")
    print()
    
    # Dispatch to appropriate pipeline
    if pipeline == "kd":
        from distill_bench.pipelines import kd_main
        kd_main.main(args)
    
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

