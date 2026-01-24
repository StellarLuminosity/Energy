"""
Evaluation Benchmark Script
"""
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
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


@dataclass
class TaskSpec:
    name: str
    runner: Callable
    description: str
    requires_model: bool = True


@dataclass
class TaskRequest:
    name: str
    params: Dict
    stage_name: str


TASK_REGISTRY: Dict[str, TaskSpec] = {}


def _register_task(name: str, runner: Callable, description: str, requires_model: bool = True):
    if name in TASK_REGISTRY:
        raise ValueError(f"Task '{name}' is already registered.")
    TASK_REGISTRY[name] = TaskSpec(name=name, runner=runner, description=description, requires_model=requires_model)


def _list_available_tasks() -> None:
    print("Available tasks:")
    for name, spec in TASK_REGISTRY.items():
        print(f"  - {name}: {spec.description}")



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


def _load_model_and_tokenizer(model_id: str, device: torch.device, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def _build_generate_fn(model, tokenizer, device: torch.device, default_max_new_tokens: int, default_temperature: float, default_top_p: float, do_sample: bool):
    """
    Create a simple text generator usable by all adapters.
    Supports single-string or list-of-string prompts.
    """

    def generate(prompts, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
        was_str = isinstance(prompts, str)
        prompt_list = [prompts] if was_str else list(prompts)

        outputs: List[str] = []
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or default_max_new_tokens,
            "temperature": temperature if temperature is not None else default_temperature,
            "top_p": top_p if top_p is not None else default_top_p,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }

        model.eval()
        with torch.no_grad():
            for prompt in prompt_list:
                encoded = tokenizer(prompt, return_tensors="pt").to(device)
                generated = model.generate(**encoded, **gen_kwargs)
                gen_text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                outputs.append(gen_text)

        return outputs[0] if was_str else outputs

    return generate


def _resolve_tasks(tasks_arg: Optional[str], config_tasks: Optional[List[str]]) -> List[str]:
    """
    Convert CLI/config task inputs into a list of task labels (strings).
    - If tasks_arg is provided, split on commas.
    - Otherwise, use config_tasks or FALLBACK_TASKS.
    """
    if tasks_arg:
        raw = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    else:
        raw = config_tasks or FALLBACK_TASKS

    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return list(raw)


def _normalize_task_requests(raw_tasks: List[str], eval_stage_prefix: str) -> List[TaskRequest]:
    """
    Map raw task strings to TaskRequest entries.
    - Known task names are looked up in TASK_REGISTRY.
    - Unknown names are assumed to be OLMES tasks (passed through to the olmes adapter).
    """
    requests: List[TaskRequest] = []
    olmes_tasks: List[str] = []

    for task in raw_tasks:
        if task == "olmes":
            continue  # explicit adapter name; config tasks will be added separately
        if task in TASK_REGISTRY:
            requests.append(TaskRequest(name=task, params={}, stage_name=f"{eval_stage_prefix}:{task}"))
        else:
            olmes_tasks.append(task)

    # If there are passthrough tasks (or user explicitly wants olmes), add one olmes request.
    if olmes_tasks or ("olmes" in raw_tasks):
        stage_name = f"{eval_stage_prefix}:olmes"
        requests.insert(0, TaskRequest(name="olmes", params={"tasks": olmes_tasks or FALLBACK_TASKS}, stage_name=stage_name))

    # Deduplicate while preserving order
    seen = set()
    deduped: List[TaskRequest] = []
    for req in requests:
        key = (req.name, tuple(req.params.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(req)
    return deduped


def _run_olmes_task(context, tasks: List[str], extra_args: List[str]) -> int:
    """
    Fallback adapter that shells out to `olmes` with the provided tasks.
    """
    benchmark_name = getattr(context.config, "benchmark_name", "olmo_benchmark")
    cmd = [
        "olmes",
        "--model",
        context.model_path,
        "--task",
        *tasks,
        "--output-dir",
        str(context.run_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[{benchmark_name}] OLMES command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=str(context.run_dir))
    return result.returncode


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
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to run (use 'list' to show available tasks).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per task (adapter-dependent).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Default max_new_tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Default generation top_p.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run and print tasks without executing them.",
    )

    # Anything not consumed here will be forwarded to `olmes`.
    args, passthrough = parser.parse_known_args()

    # Load experiment config
    config = load_config(args.config)

    # Resolve where this benchmark should write its logs and results
    run_dir = _resolve_benchmark_run_dir(config, args.run_dir)
    eval_stage_name = args.eval_stage_name or config.benchmark_name or "olmo_benchmark"

    # Register tasks (adapters will be added in later steps).
    _register_task(
        name="olmes",
        runner=None,  # placeholder; handled separately
        description="Run OLMES CLI tasks (pass-through).",
        requires_model=False,
    )
    _register_task(
        name="gsm8k",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("gsm8k adapter not implemented yet.")),
        description="GSM8K via lm-eval-harness (adapter pending).",
    )
    _register_task(
        name="mmlu",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("mmlu adapter not implemented yet.")),
        description="MMLU via lm-eval-harness (adapter pending).",
    )
    _register_task(
        name="ifeval",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("ifeval adapter not implemented yet.")),
        description="IFEval via lm-eval-harness (adapter pending).",
    )
    _register_task(
        name="alpaca_eval",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("AlpacaEval adapter not implemented yet.")),
        description="AlpacaEval 2 via official library (adapter pending).",
    )
    _register_task(
        name="mt_bench_101",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("MT-Bench-101 adapter not implemented yet.")),
        description="MT-Bench-101 via official repo (adapter pending).",
    )

    if args.tasks == "list":
        _list_available_tasks()
        return

    # Decide which model to evaluate
    model_str = getattr(config, "benchmark_model", None)
    if not model_str:
        raise ValueError(
            "No benchmark.model configured. Please set benchmark.model in your YAML "
            "(either a Hugging Face id like 'allenai/OLMo-2-1124-7B-SFT' or a local "
            "HF-format directory like "
            "'/scratch/.../kd_32b_to_1b_adamw/final_model/hf_format')."
        )

    raw_tasks = _resolve_tasks(args.tasks, getattr(config, "benchmark_tasks", None))
    benchmark_name = config.benchmark_name
    task_requests = _normalize_task_requests(raw_tasks, eval_stage_name)

    if args.dry_run:
        print(f"[{benchmark_name}] Dry-run enabled. Tasks to run (in order): {[r.name for r in task_requests]}")
        print(f"[{benchmark_name}] Run dir: {run_dir}")
        return

    # NEW: if it's a .pt checkpoint, convert to HF dir under run_dir
    model_path_for_use = _maybe_convert_checkpoint_to_hf(
        model_spec=model_str,
        config=config,
        run_dir=run_dir,
        subdir_name="hf_from_checkpoint",
    )

    # Prepare EnergyTracker
    tracker = EnergyTracker(run_dir=str(run_dir), config=config)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = torch.bfloat16

    # Load model/tokenizer only if any task requires it
    needs_model = any(TASK_REGISTRY[req.name].requires_model for req in task_requests)
    model = tokenizer = generate_fn = None
    if needs_model:
        print(f"[{benchmark_name}] Loading model into memory: {model_path_for_use} (device={device})")
        model, tokenizer = _load_model_and_tokenizer(model_path_for_use, device=device, torch_dtype=torch_dtype)
        generate_fn = _build_generate_fn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            default_max_new_tokens=args.max_new_tokens,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            do_sample=True,
        )
    else:
        print(f"[{benchmark_name}] No in-memory model load required for selected tasks.")

    @dataclass
    class BenchmarkContext:
        config: any
        run_dir: Path
        model_path: str
        model: any
        tokenizer: any
        generate_fn: Callable
        device: torch.device
        max_samples: Optional[int]

    context = BenchmarkContext(
        config=config,
        run_dir=run_dir,
        model_path=model_path_for_use,
        model=model,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        device=device,
        max_samples=args.max_samples,
    )

    returncode = 0
    for req in task_requests:
        spec = TASK_REGISTRY.get(req.name)
        if not spec:
            print(f"[{benchmark_name}] Unknown task '{req.name}'. Skipping.")
            continue

        tracker.start_stage(req.stage_name)
        try:
            if spec.name == "olmes":
                returncode = _run_olmes_task(context, tasks=req.params.get("tasks", []), extra_args=passthrough)
            else:
                spec.runner(context=context, params=req.params)
        except NotImplementedError as e:
            print(f"[{benchmark_name}] Task '{req.name}' not implemented yet: {e}")
            returncode = 1
        finally:
            tracker.end_stage()  # tokens_processed left as default 0

        if returncode != 0:
            print(f"[{benchmark_name}] Task '{req.name}' exited with code {returncode}")
            break

    tracker.save_summary()
    sys.exit(returncode)


if __name__ == "__main__":
    main()
