"""
Adapter for MT-Bench-101 using the official repo.

We assume the repo code is importable as `mt_bench_101`. This adapter wires the
shared generator into the MT-Bench-101 evaluation flow, runs a limited subset
if requested, and saves scores plus raw conversations.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _maybe_slice(lst: List, limit: Optional[int]) -> List:
    if limit is None or limit <= 0:
        return lst
    return lst[:limit]


def _run_mt_bench_101(context, params: Dict) -> int:
    try:
        from mt_bench_101 import data as mt_data
        from mt_bench_101 import eval as mt_eval
    except ImportError as e:
        raise ImportError(
            "mt-bench-101 is not installed. Install from https://github.com/mtbench101/mt-bench-101."
        ) from e

    limit = params.get("limit") or context.max_samples
    output_dir = Path(context.run_dir) / "mt_bench_101"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = mt_data.load_questions()
    if limit:
        prompts = _maybe_slice(prompts, limit)

    # Simple generator that follows the MT-Bench-101 expected callable signature.
    def _model_fn(prompt: str) -> str:
        return context.generate_fn(prompt)

    conversations = mt_eval.run_model_on_questions(
        model_name=str(context.model_path),
        questions=prompts,
        chat_fn=_model_fn,
    )

    # Score using the repo's judge (self-contained)
    scores = mt_eval.score_conversations(conversations)

    (output_dir / "conversations.json").write_text(json.dumps(conversations, indent=2))
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(f"[mt_bench_101] Conversations and scores saved under {output_dir}")
    return 0


def register_tasks(register_fn: Callable[[str, Callable, str, bool], None]) -> None:
    register_fn(
        name="mt_bench_101",
        runner=_run_mt_bench_101,
        description="MT-Bench-101 via official repo.",
        requires_model=True,
    )
