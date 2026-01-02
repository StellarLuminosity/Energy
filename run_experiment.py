#!/usr/bin/env python3
"""
Unified Experiment Launcher for Distillation Energy Benchmarking.

Loads YAML config and dispatches to the appropriate pipeline (KD, SFT, or DPO).

Usage:
    python run_experiment.py --config configs/experiments/kd_7b_to_1b.yaml
    python run_experiment.py --config configs/experiments/sft_7b_to_1b.yaml [--skip-validation]
"""

import argparse
import os
import sys
from pathlib import Path

from distill_bench.core.config_loader import load_config
from distill_bench.core.environment import collect_environment, save_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Run distillation experiment with energy tracking")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-run validation (idle baseline, burn-in tests)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Determine pipeline type
    pipeline_type = config.pipeline.lower()
    print(f"Pipeline type: {pipeline_type}")
    
    # Collect and save environment metadata
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting hardware/software environment metadata...")
    env_metadata = collect_environment()
    save_environment(env_metadata, output_dir / "environment.json")
    
    # Run pre-run validation if requested
    if not args.skip_validation:
        print("\n" + "="*60)
        print("Running pre-run validation...")
        print("="*60)
        try:
            from distill_bench.core.prerun import run_prerun_validation
            run_prerun_validation(config, output_dir)
            print("✓ Pre-run validation passed\n")
        except ImportError:
            print("⚠ Pre-run validation not implemented yet, skipping...\n")
        except Exception as e:
            print(f"⚠ Pre-run validation failed: {e}")
            print("Continuing anyway...\n")
    
    # Dispatch to appropriate pipeline
    print("="*60)
    print(f"Starting {pipeline_type.upper()} pipeline...")
    print("="*60 + "\n")
    
    if pipeline_type == "kd":
        from distill_bench.pipelines.kd_main import main as kd_main
        # kd_main expects argparse-style args, so create a simple namespace
        class SimpleArgs:
            mixed_precision = True
        kd_main(SimpleArgs())
        
    elif pipeline_type == "sft":
        from distill_bench.pipelines.sft_main import main as sft_main
        sft_main(config)
        
    elif pipeline_type == "dpo":
        from distill_bench.pipelines.dpo_main import main as dpo_main
        dpo_main(config)
        
    else:
        print(f"ERROR: Unknown pipeline type '{pipeline_type}'")
        print(f"Supported types: kd, sft, dpo")
        sys.exit(1)
    
    print("\n" + "="*60)
    print(f"✓ {pipeline_type.upper()} pipeline completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

