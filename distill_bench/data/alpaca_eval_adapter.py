"""
Adapter for AlpacaEval 2 using the official alpaca_eval library.

We expose a register_tasks(register_fn) helper so the main benchmark script can
plug the runner into its task registry without importing alpaca_eval unless
requested.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _make_model_fn(generate_fn: Callable) -> Callable:
    """
    Wrap the shared generate_fn so alpaca_eval can call it with either a single
    prompt or a list of prompts.
    """

    def _model_fn(prompts: Any, **_: Any):
        return generate_fn(prompts)

    return _model_fn


def _run_alpaca_eval(context, params: Dict) -> int:
    try:
        from alpaca_eval import evaluate
    except ImportError as e:
        raise ImportError(
            "alpaca_eval is not installed. Install with `pip install alpaca_eval`."
        ) from e

    limit = params.get("limit") or context.max_samples
    output_dir = Path(context.run_dir) / "alpaca_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_fn = _make_model_fn(context.generate_fn)

    # Try to pass limit if supported; fall back if not.
    try:
        results = evaluate(
            model=model_fn,
            model_name=str(context.model_path),
            output_dir=str(output_dir),
            eval_type="alpaca_eval",
            limit=limit,
        )
    except TypeError:
        results = evaluate(
            model=model_fn,
            model_name=str(context.model_path),
            output_dir=str(output_dir),
            eval_type="alpaca_eval",
        )

    out_path = output_dir / "alpaca_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[alpaca_eval] Results saved to {out_path}")
    return 0


def register_tasks(register_fn: Callable[[str, Callable, str, bool], None]) -> None:
    register_fn(
        name="alpaca_eval",
        runner=_run_alpaca_eval,
        description="AlpacaEval 2 via official library.",
        requires_model=True,
    )
