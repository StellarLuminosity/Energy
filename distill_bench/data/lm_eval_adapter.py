"""
Adapter for running lm-eval-harness tasks (GSM8K, MMLU, IFEval).

This module exposes a register_tasks(register_fn) helper so the main benchmark
script can plug runners into its task registry without pulling lm-eval at import
time. Each runner invokes `lm_eval.evaluator.simple_evaluate` with the HF model
identifier (or local path) derived from the benchmark config.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional


SUPPORTED_TASKS = {
    "gsm8k": "GSM8K via lm-eval-harness",
    "mmlu": "MMLU via lm-eval-harness",
    "ifeval": "IFEval via lm-eval-harness",
}


def _build_model_args(model_path: str, device: str) -> str:
    """
    Build lm-eval model_args string for hf-causal backend.
    """
    args = [
        f"pretrained={model_path}",
        f"tokenizer={model_path}",
        "dtype=bfloat16",
        "trust_remote_code=True",
    ]
    if device.startswith("cuda"):
        args.append("device_map=auto")
    else:
        args.append(f"device={device}")
    return ",".join(args)


def _run_single_task(task_name: str, model_path: str, device: str, limit: Optional[int], output_dir: Path) -> int:
    try:
        from lm_eval import evaluator
    except ImportError as e:
        raise ImportError(
            "lm-eval-harness is not installed. Install with `pip install lm-eval`."
        ) from e

    model_args = _build_model_args(model_path, device)
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=model_args,
        tasks=[task_name],
        batch_size="auto",
        max_batch_size=16,
        limit=limit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"lm_eval_{task_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{task_name}] lm-eval results saved to {out_path}")
    return 0


def _make_runner(task_name: str) -> Callable:
    def _runner(context, params: Dict):
        return _run_single_task(
            task_name=task_name,
            model_path=context.model_path,
            device=str(context.device),
            limit=params.get("limit") or context.max_samples,
            output_dir=context.run_dir,
        )

    return _runner


def register_tasks(register_fn: Callable[[str, Callable, str, bool], None]) -> None:
    """
    Register lm-eval tasks into the benchmark task registry.
    """
    for task, desc in SUPPORTED_TASKS.items():
        register_fn(
            name=task,
            runner=_make_runner(task),
            description=desc,
            requires_model=False,  # lm-eval will load the model itself
        )
