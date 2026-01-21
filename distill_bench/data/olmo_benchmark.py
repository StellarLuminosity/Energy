"""
Evaluation Benchmark Script
"""
import os
import torch
import argparse
import subprocess
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill_bench.core.config_loader import load_config
from distill_bench.core.energy_logger import EnergyTracker

# Fallback tasks if none are specified in config.benchmark.tasks.
FALLBACK_TASKS = [
    "core_9mcqa::olmes",
    "mmlu:mc::olmes",
    "olmo_2_generative::olmes",
    "olmo_2_heldout::olmes",
]


def _resolve_benchmark_run_dir(config, run_dir_arg: str | None) -> Path:
    """
    Resolve the root directory where this benchmark run should live.

    Priority:
      1. --run-dir CLI argument (if provided)
      2. config.benchmark_output_dir
      3. config.output_dir as a last fallback
    """
    if run_dir_arg:
        base = Path(run_dir_arg)
    else:
        base = Path(getattr(config, "benchmark_output_dir", None) or config.output_dir)

    # Final run dir is: <base>/<benchmark_name>
    benchmark_name = config.benchmark_name
    run_dir = base / benchmark_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def _maybe_convert_checkpoint_to_hf(
    model_spec: str,
    config,
    run_dir: Path,
    subdir_name: str = "hf_from_checkpoint",
) -> str:
    """
    Normalize benchmark.model so OLMES can use it:

    1) If it's a Hugging Face name like "allenai/OLMo-2-0325-32B-SFT":
       - return the string unchanged.

    2) If it's a local directory with HF-style configs (e.g.
       ".../final_model" or ".../final_model/hf_format"):
       - return the HF directory path (preferring "hf_format" if present),
       - make sure tokenizer files exist there (using tokenizer from config).

    3) If it's a checkpoint file (e.g. ".../checkpoint_epoch0_step26400.pt"):
       - use benchmark.model_type from the config as the HF base model,
       - load that HF model,
       - load the checkpoint state dict into it,
       - save HF-format weights + tokenizer into run_dir/subdir_name,
       - return that directory path.
    """
    benchmark_name = getattr(config, "benchmark_name", "olmo_benchmark")
    p = Path(model_spec)

    # -------------------------
    # Case 1: pure HF name (doesn't exist as a local path)
    # -------------------------
    if not p.exists():
        print(f"[{benchmark_name}] Using Hugging Face model id: {model_spec}")
        return model_spec

    # -------------------------
    # Case 2: local directory (final_model / hf_format)
    # -------------------------
    if p.is_dir():
        # If there is a final_model/hf_format layout, prefer hf_format
        if (p / "hf_format" / "config.json").exists():
            hf_dir = p / "hf_format"
        else:
            hf_dir = p

        if not (hf_dir / "config.json").exists():
            raise ValueError(
                f"[{benchmark_name}] Directory '{hf_dir}' does not look like a "
                f"Hugging Face model (config.json missing)."
            )

        # Ensure tokenizer files exist in this dir, using the tokenizer from config.
        tokenizer_marker_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        ]
        has_tokenizer = any((hf_dir / f).exists() for f in tokenizer_marker_files)

        if not has_tokenizer:
            tokenizer_id = (
                getattr(config, "tokenizer_name", None)
                or getattr(config, "data_tokenizer_name", None)
            )
            if not tokenizer_id:
                raise ValueError(
                    f"[{benchmark_name}] Need tokenizer_name in config (e.g. "
                    f"data.tokenizer_name in base.yaml) to populate tokenizer "
                    f"files for directory '{hf_dir}'."
                )

            print(
                f"[{benchmark_name}] No tokenizer files in {hf_dir}, "
                f"saving tokenizer '{tokenizer_id}' there."
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            tokenizer.save_pretrained(hf_dir)

        print(f"[{benchmark_name}] Using local HF directory: {hf_dir}")
        return str(hf_dir)

    # -------------------------
    # Case 3: local checkpoint file (.pt / .bin)
    # -------------------------
    if p.suffix not in {".pt", ".bin"}:
        raise ValueError(
            f"[{benchmark_name}] benchmark.model points to file '{p}', but "
            f"extension '{p.suffix}' is not a recognized checkpoint type."
        )

    base_model_id = config.model_type
    if not base_model_id:
        raise ValueError(
            f"[{benchmark_name}] benchmark.model is a checkpoint, so you must "
            f"set benchmark.model_type in the config to a Hugging Face model id "
            f"(e.g. 'allenai/OLMo-2-1124-7B-SFT')."
        )

    print(
        f"[{benchmark_name}] Converting checkpoint '{p}' using base model "
        f"'{base_model_id}'"
    )

    # 1. Load base HF model
    model = AutoModelForCausalLM.from_pretrained(base_model_id)

    # 2. Load checkpoint and extract state dict
    ckpt = torch.load(p, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            # Common pattern: torch.save(model.state_dict())
            state_dict = ckpt
    else:
        raise ValueError(
            f"[{benchmark_name}] Expected checkpoint '{p}' to be a dict, "
            f"got type {type(ckpt)}."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"[{benchmark_name}] Missing keys when loading checkpoint "
            f"({len(missing)}), first few: {missing[:5]}"
        )
    if unexpected:
        print(
            f"[{benchmark_name}] Unexpected keys in checkpoint "
            f"({len(unexpected)}), first few: {unexpected[:5]}"
        )

    # 3. Save HF-format model + tokenizer to run_dir/subdir_name
    hf_dir = run_dir / subdir_name
    hf_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_id = (
        getattr(config, "tokenizer_name", None)
        or getattr(config, "data_tokenizer_name", None)
        or base_model_id
    )

    print(
        f"[{benchmark_name}] Saving HF model + tokenizer to {hf_dir} "
        f"(tokenizer='{tokenizer_id}')"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.save_pretrained(hf_dir)
    model.save_pretrained(hf_dir)

    return str(hf_dir)


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
        default="olmo_benchmark",
        help="Name for the EnergyTracker stage (default: olmo_benchmark).",
    )

    # Anything not consumed here will be forwarded to `olmes`.
    args, olmes_extra = parser.parse_known_args()

    # Load experiment config
    config = load_config(args.config)

    # Resolve where this benchmark should write its logs and results
    run_dir = _resolve_benchmark_run_dir(config, args.run_dir)
    eval_stage_name = args.eval_stage_name or config.benchmark_name or "olmo_benchmark"

    # Decide which model to evaluate
    model_str = getattr(config, "benchmark_model", None)
    if not model_str:
        raise ValueError(
            "No benchmark.model configured. Please set benchmark.model in your YAML "
            "(either a Hugging Face id like 'allenai/OLMo-2-1124-7B-SFT' or a local "
            "HF-format directory like "
            "'/scratch/.../kd_32b_to_1b_adamw/final_model/hf_format')."
        )

    # NEW: if it's a .pt checkpoint, convert to HF dir under run_dir
    model_path_for_olmes = _maybe_convert_checkpoint_to_hf(
        model_spec=model_str,
        config=config,
        run_dir=run_dir,
        subdir_name="hf_from_checkpoint",
    )
    
    tasks = getattr(config, "benchmark_tasks", None) or FALLBACK_TASKS
    benchmark_name = config.benchmark_name

    # OLMES output goes into the same run_dir
    eval_output_dir = run_dir
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare EnergyTracker
    tracker = EnergyTracker(run_dir=str(run_dir), config=config)

    # Build OLMES command
    cmd = [
        "olmes",
        "--model",
        model_path_for_olmes,
        "--task",
        *tasks,
        "--output-dir",
        str(eval_output_dir),
    ]
    
    # Allow power users to pass extra flags directly to `olmes`
    if olmes_extra:
        cmd.extend(olmes_extra)

    print(f"[{benchmark_name}] Running model: {model_path_for_olmes}")
    print(f"[{benchmark_name}] Tasks: {tasks}")
    print(f"[{benchmark_name}] Run dir: {run_dir}")
    print(f"[{benchmark_name}] OLMES command: {' '.join(cmd)}")

    returncode = 1
    try:
        tracker.start_stage(eval_stage_name)
        # EnergyTracker will still record GPU/CPU energy even without token counts.
        result = subprocess.run(cmd, check=False, cwd=str(run_dir))
        returncode = result.returncode
        if returncode != 0:
            print(f"[{benchmark_name}] OLMES exited with code {returncode}")
    finally:
        tracker.end_stage()       # tokens_processed left as default 0
        tracker.save_summary()

    sys.exit(returncode)


if __name__ == "__main__":
    main()
