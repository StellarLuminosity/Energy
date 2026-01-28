#!/usr/bin/env python3
"""
Compute benchmark scores per evaluation folder (e.g., olmo_benchmark_*).

Priority:
  1) metrics-all.jsonl  (preferred)
  2) metrics.json
  3) task-*-metrics.json files (fallback)

Outputs:
  benchmark_results/summary.csv
  benchmark_results/<model_dir_name>/tasks.json
  benchmark_results/<model_dir_name>/scores.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Helpers
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    if isinstance(x, str):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except ValueError:
            return None
    if isinstance(x, dict):
        # Common patterns seen in metric outputs
        for k in ("value", "mean", "avg", "score"):
            if k in x:
                return _safe_float(x.get(k))
    return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_task_name(record: Dict[str, Any], fallback: str) -> str:
    for k in ("task_name", "name", "task"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback


def _extract_num_instances(record: Dict[str, Any]) -> Optional[int]:
    # Most common in OLMES records
    for k in ("num_instances", "n_instances", "num_examples", "n_examples"):
        v = record.get(k)
        if isinstance(v, int) and v > 0:
            return v
        if isinstance(v, str) and v.isdigit():
            iv = int(v)
            if iv > 0:
                return iv

    # Sometimes nested
    tc = record.get("task_config")
    if isinstance(tc, dict):
        for k in ("num_instances", "n_instances", "num_examples", "n_examples"):
            v = tc.get(k)
            if isinstance(v, int) and v > 0:
                return v
            if isinstance(v, str) and v.isdigit():
                iv = int(v)
                if iv > 0:
                    return iv
    return None


def _extract_primary_metric(record: Dict[str, Any]) -> Optional[str]:
    tc = record.get("task_config")
    if isinstance(tc, dict):
        pm = tc.get("primary_metric")
        if isinstance(pm, str) and pm.strip():
            return pm.strip()
    pm = record.get("primary_metric")
    if isinstance(pm, str) and pm.strip():
        return pm.strip()
    return None


def _extract_primary_score(record: Dict[str, Any]) -> Optional[float]:
    """
    Tries the canonical location first:
      record["metrics"]["primary_score"]

    If missing, tries:
      - record["primary_score"]
      - infer from primary_metric and metrics dict
    """
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        v = _safe_float(metrics.get("primary_score"))
        if v is not None:
            return v

    v = _safe_float(record.get("primary_score"))
    if v is not None:
        return v

    # Infer using primary_metric
    pm = _extract_primary_metric(record)
    if pm and isinstance(metrics, dict):
        if pm in metrics:
            return _safe_float(metrics.get(pm))
        # Sometimes metric keys are nested like "acc,none" or "exact_match,none"
        for k, val in metrics.items():
            if isinstance(k, str) and (k == pm or k.startswith(pm + ",")):
                vv = _safe_float(val)
                if vv is not None:
                    return vv

    return None


def _is_leaf_record(record: Dict[str, Any]) -> bool:
    """
    For metrics-all.jsonl, suite aggregate records often have task_idx = null.
    Leaf tasks usually have an integer task_idx.
    If task_idx is missing entirely, treat as leaf (best-effort).
    """
    if "task_idx" not in record:
        return True
    return record.get("task_idx") is not None


def _task_name_from_filename(p: Path) -> str:
    # Examples:
    #   task-000-arc_easy:mc-metrics.json  -> arc_easy:mc
    #   task-017-winogrande-metrics.json   -> winogrande
    m = re.match(r"^task-\d+-(.+)-metrics\.json$", p.name)
    if m:
        return m.group(1)
    # fallback
    return p.stem.replace("task-", "")


@dataclass
class TaskScore:
    task_name: str
    score: float
    num_instances: Optional[int]
    source: str


@dataclass
class ModelScores:
    model_dir: str
    method_used: str
    macro_avg: Optional[float]
    micro_avg: Optional[float]
    n_tasks: int
    n_missing_weights: int
    tasks: List[TaskScore]


def _aggregate(tasks: List[TaskScore]) -> Tuple[Optional[float], Optional[float], int]:
    if not tasks:
        return None, None, 0

    scores = [t.score for t in tasks]
    macro = sum(scores) / len(scores)

    # micro weighted by num_instances where available, else weight=1
    missing_w = 0
    weighted_sum = 0.0
    weight_total = 0.0
    for t in tasks:
        w = t.num_instances
        if w is None:
            missing_w += 1
            w = 1
        weighted_sum += t.score * float(w)
        weight_total += float(w)

    micro = (weighted_sum / weight_total) if weight_total > 0 else None
    return macro, micro, missing_w


# ----------------------------
# Loaders (priority order)
# ----------------------------

def load_from_metrics_all(model_path: Path) -> Optional[List[TaskScore]]:
    p = model_path / "metrics-all.jsonl"
    if not p.exists():
        return None

    tasks: List[TaskScore] = []
    for rec in _read_jsonl(p):
        if not isinstance(rec, dict):
            continue
        if not _is_leaf_record(rec):
            continue
        score = _extract_primary_score(rec)
        if score is None:
            continue
        task_name = _extract_task_name(rec, fallback="unknown_task")
        n = _extract_num_instances(rec)
        tasks.append(TaskScore(task_name=task_name, score=score, num_instances=n, source=str(p.name)))

    return tasks if tasks else None


def _iter_records_from_metrics_json(obj: Any) -> Iterable[Dict[str, Any]]:
    # metrics.json can be list[dict] or dict with a list inside
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
        return

    if isinstance(obj, dict):
        for key in ("records", "tasks", "results", "data"):
            v = obj.get(key)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return
        # fallback: search for a list of dicts at top-level values
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for x in v:
                    yield x
                return


def load_from_metrics_json(model_path: Path) -> Optional[List[TaskScore]]:
    p = model_path / "metrics.json"
    if not p.exists():
        return None

    obj = _read_json(p)
    tasks: List[TaskScore] = []
    for rec in _iter_records_from_metrics_json(obj):
        # metrics.json may include aggregates; leaf detection is best-effort
        if not _is_leaf_record(rec):
            continue
        score = _extract_primary_score(rec)
        if score is None:
            continue
        task_name = _extract_task_name(rec, fallback="unknown_task")
        n = _extract_num_instances(rec)
        tasks.append(TaskScore(task_name=task_name, score=score, num_instances=n, source=str(p.name)))

    return tasks if tasks else None


def load_from_task_metric_files(model_path: Path) -> Optional[List[TaskScore]]:
    metric_files = sorted(model_path.glob("task-*-metrics.json"))
    if not metric_files:
        return None

    tasks: List[TaskScore] = []
    for mf in metric_files:
        try:
            rec = _read_json(mf)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue

        score = _extract_primary_score(rec)
        if score is None:
            # Some task metric files might store metrics at top-level differently;
            # best-effort: look for a top-level "primary_score"
            score = _safe_float(rec.get("primary_score"))
        if score is None:
            continue

        task_name = _extract_task_name(rec, fallback=_task_name_from_filename(mf))
        n = _extract_num_instances(rec)
        tasks.append(TaskScore(task_name=task_name, score=score, num_instances=n, source=str(mf.name)))

    return tasks if tasks else None


def compute_model_scores(model_path: Path) -> ModelScores:
    model_dir_name = model_path.name

    tasks = load_from_metrics_all(model_path)
    method = "metrics-all.jsonl"
    if tasks is None:
        tasks = load_from_metrics_json(model_path)
        method = "metrics.json"
    if tasks is None:
        tasks = load_from_task_metric_files(model_path)
        method = "task-*-metrics.json"
    if tasks is None:
        tasks = []
        method = "no-metrics-found"

    macro, micro, missing_w = _aggregate(tasks)
    return ModelScores(
        model_dir=model_dir_name,
        method_used=method,
        macro_avg=macro,
        micro_avg=micro,
        n_tasks=len(tasks),
        n_missing_weights=missing_w,
        tasks=tasks,
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".",
                    help="Run directory (default: current dir).")
    ap.add_argument("--out", type=str, default="benchmark_results",
                    help="Output directory (default: ./benchmark_results).")
    ap.add_argument("--pattern", type=str, default="olmo_benchmark_*",
                    help="Glob pattern for model eval folders.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([p for p in root.glob(args.pattern) if p.is_dir()])

    all_rows: List[Dict[str, Any]] = []
    for mp in model_dirs:
        ms = compute_model_scores(mp)

        # Write per-model artifacts
        model_out = out_dir / ms.model_dir
        model_out.mkdir(parents=True, exist_ok=True)

        (model_out / "tasks.json").write_text(
            json.dumps([asdict(t) for t in ms.tasks], indent=2),
            encoding="utf-8",
        )
        (model_out / "scores.json").write_text(
            json.dumps({
                "model_dir": ms.model_dir,
                "method_used": ms.method_used,
                "macro_avg": ms.macro_avg,
                "micro_avg": ms.micro_avg,
                "n_tasks": ms.n_tasks,
                "n_missing_weights": ms.n_missing_weights,
            }, indent=2),
            encoding="utf-8",
        )

        all_rows.append({
            "model_dir": ms.model_dir,
            "method_used": ms.method_used,
            "macro_avg": ms.macro_avg,
            "micro_avg": ms.micro_avg,
            "n_tasks": ms.n_tasks,
            "n_missing_weights": ms.n_missing_weights,
            "path": str(mp),
        })

    # Write global summary CSV
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model_dir", "method_used", "macro_avg", "micro_avg", "n_tasks", "n_missing_weights", "path"],
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote per-model details under: {out_dir}")


if __name__ == "__main__":
    main()
