#!/usr/bin/env python3
"""
Convert a single StageMetrics JSON file into (or into part of)
an experiment_summary.json in the same directory.

Usage:
    python convert_stage_to_experiment_summary.py path/to/stage.json [more_stage.json ...]
"""

import json
import sys
from pathlib import Path


def convert_stage_file(stage_path: Path) -> Path:
    """Wrap a single stage JSON into experiment_summary.json format."""
    with stage_path.open("r") as f:
        stage = json.load(f)

    # Determine a stage name; prefer explicit field, fall back to filename stem.
    stage_name = (
        stage.get("stage_name")
        or stage.get("stage_id")
        or stage_path.stem
    )

    # experiment_summary.json will live in the same directory as the stage JSON
    summary_path = stage_path.parent / "experiment_summary.json"

    if summary_path.exists():
        # Update existing summary
        with summary_path.open("r") as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                # If it's corrupted, start fresh
                summary = {}
        if not isinstance(summary.get("stages"), dict):
            summary["stages"] = {}
        # Fill in experiment_id/name if missing
        summary.setdefault(
            "experiment_id",
            stage.get("experiment_id") or stage.get("stage_id") or stage_name,
        )
        summary.setdefault(
            "experiment_name",
            stage.get("experiment_name") or stage_name,
        )
    else:
        # Create a brand new summary
        summary = {
            "experiment_id": stage.get("experiment_id")
                            or stage.get("stage_id")
                            or stage_name,
            "experiment_name": stage.get("experiment_name") or stage_name,
            "stages": {},
        }

    # Insert / overwrite this stage entry
    summary["stages"][stage_name] = stage

    # Write back to disk
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[OK] Wrote {summary_path} with stage '{stage_name}'")
    return summary_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python convert_stage_to_experiment_summary.py stage1.json [stage2.json ...]")
        return 1

    for arg in argv[1:]:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            print(f"[WARN] Skipping {arg}: file not found")
            continue
        if path.suffix.lower() != ".json":
            print(f"[WARN] Skipping {arg}: not a .json file")
            continue
        convert_stage_file(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
