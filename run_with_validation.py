#!/usr/bin/env python3
"""
Example: Running an experiment with pre-run validation.

This shows the recommended workflow for integrating pre-run validation
before launching expensive energy benchmarking experiments.
"""

import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from distill_bench.core.prerun import run_prerun_validation
from distill_bench.core.environment import collect_environment, print_environment


def main():
    parser = argparse.ArgumentParser(description="Run experiment with pre-run validation")
    parser.add_argument("--experiment-dir", type=str, required=True,
                      help="Directory for experiment outputs")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to experiment config YAML")
    parser.add_argument("--skip-validation", action="store_true",
                      help="Skip pre-run validation (not recommended)")
    parser.add_argument("--quick-validation", action="store_true",
                      help="Use quick validation with reduced durations")
    parser.add_argument("--expected-gpu", type=str, default=None,
                      help="Expected GPU type (e.g., 'A100')")
    parser.add_argument("--min-vram-gb", type=float, default=16.0,
                      help="Minimum required VRAM in GB")
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EXPERIMENT SETUP WITH VALIDATION")
    print("="*70)
    print(f"Experiment: {experiment_dir}")
    print(f"Config: {args.config}")
    print()
    
    # Step 1: Collect and display environment metadata
    print("Step 1: Collecting Environment Metadata...")
    env = collect_environment()
    print_environment(env)
    
    # Step 2: Pre-run validation
    if not args.skip_validation:
        print("\nStep 2: Running Pre-Run Validation...")
        print("(This may take 5-15 minutes depending on settings)")
        print()
        
        validation_dir = experiment_dir / "prerun_validation"
        
        if args.quick_validation:
            print("Using QUICK validation (reduced durations)...")
            idle_duration = 1.0
            burn_in_steps = 100
        else:
            print("Using FULL validation (recommended for production)...")
            idle_duration = 5.0
            burn_in_steps = 500
        
        report = run_prerun_validation(
            output_dir=str(validation_dir),
            idle_duration_minutes=idle_duration,
            burn_in_steps=burn_in_steps,
            expected_gpu_type=args.expected_gpu,
        )
        
        if not report.all_checks_passed:
            print("\n" + "="*70)
            print("⚠ VALIDATION FAILED")
            print("="*70)
            print("\nCritical issues detected:")
            for warning in report.critical_warnings:
                print(f"  - {warning}")
            
            print("\nPlease address these issues before running the experiment.")
            print("Validation report saved to:")
            print(f"  {validation_dir / 'prerun_validation_report.json'}")
            print("="*70)
            
            return 1
        
        print("\n" + "="*70)
        print("✓ VALIDATION PASSED")
        print("="*70)
        print("\nAll pre-run checks completed successfully!")
        print(f"Idle baseline: {report.idle_baseline.gpu_avg_power_watts:.2f} W")
        print(f"Burn-in energy: {report.burn_in.gpu_energy_joules:.2f} J")
        print(f"Recommended polling: {report.sampling_interval.recommended_interval_ms}ms")
        print("="*70)
        print()
    else:
        print("\nStep 2: Skipping Pre-Run Validation (not recommended)")
        print()
    
    # Step 3: Run actual experiment
    print("Step 3: Running Experiment...")
    print("(This is where you would call your actual training pipeline)")
    print()
    
    # Example: How to integrate with actual pipelines
    print("Example integration:")
    print()
    print("  from distill_bench.pipelines.kd_main import run_kd")
    print("  from distill_bench.core.config_loader import load_config")
    print()
    print("  config = load_config(args.config)")
    print("  run_kd(config=config, output_dir=str(experiment_dir))")
    print()
    
    # For demonstration, just print what would happen
    print(f"✓ Would run experiment with config: {args.config}")
    print(f"✓ Results would be saved to: {experiment_dir}")
    print()
    
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nRecommended next steps:")
    print("1. Check validation report in prerun_validation/")
    print("2. Review energy logs in energy_logs/")
    print("3. Analyze results with provided visualization tools")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

