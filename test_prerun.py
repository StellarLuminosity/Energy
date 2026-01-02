#!/usr/bin/env python3
"""
Test script for pre-run validation functionality.

This demonstrates how to use the prerun validation module before
running expensive energy benchmarking experiments.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from distill_bench.core.prerun import (
    quick_validation,
    run_prerun_validation,
    measure_idle_baseline,
    validate_hardware,
)


def main():
    """Run test of prerun validation."""
    print("Testing Pre-Run Validation Module")
    print("="*70)
    print()
    
    # Option 1: Quick validation (for testing/debugging)
    print("Running QUICK validation (reduced durations)...")
    print()
    
    success = quick_validation(output_dir="./test_prerun_output")
    
    if success:
        print("\n✓ Quick validation passed!")
    else:
        print("\n✗ Quick validation failed - check warnings")
    
    print("\n" + "="*70)
    print("\nTo run FULL validation (for actual experiments), use:")
    print("  python distill_bench/core/prerun.py --output-dir ./prerun_validation")
    print("\nOr in your code:")
    print("  from distill_bench.core.prerun import run_prerun_validation")
    print("  report = run_prerun_validation(output_dir='./validation', idle_duration_minutes=5.0)")
    print("="*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

