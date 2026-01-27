#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


# In[8]:


LOG_ROOTS = [
    Path("logs"),
    Path("trillium-logs"),
    Path("runpod2_logs"),
]
ROOT_CLUSTER = {
    "logs": "killarney",
    "trillium-logs": "trillium",
    "runpod2_logs": "runpod",
}



# In[ ]:


# Codecarbon Helper

def load_codecarbon_logs(log_roots: List[Path]) -> pd.DataFrame:
    """
    Load CodeCarbon emissions.csv from each root into a single DataFrame.

    Returns columns including:
        root, cluster, project_name, experiment_id,
        duration, cpu_energy, gpu_energy, ram_energy, energy_consumed, emissions, ...
    """
    cc_rows = []

    for root in log_roots:
        cc_dir = root / "codecarbon"
        if not cc_dir.exists():
            continue

        # Prefer the main emissions.csv; ignore .bak variants here
        cc_path = cc_dir / "emissions.csv"
        if not cc_path.exists():
            continue

        try:
            df = pd.read_csv(cc_path)
        except Exception as e:
            print(f"[WARN] Failed to read CodeCarbon CSV at {cc_path}: {e}")
            continue

        df = df.copy()
        df["root"] = str(root)
        df["cluster"] = ROOT_CLUSTER.get(root.name, root.name)
        cc_rows.append(df)

    if not cc_rows:
        return pd.DataFrame()

    cc_df = pd.concat(cc_rows, ignore_index=True)

    # Normalize names we use often
    cc_df.rename(
        columns={
            "energy_consumed": "energy_consumed_kwh",
            "cpu_energy": "cpu_energy_kwh",
            "gpu_energy": "gpu_energy_kwh",
            "ram_energy": "ram_energy_kwh",
        },
        inplace=True,
    )

    return cc_df



# In[ ]:


# Stage Metrics Normalization

STAGE_DEFAULTS: Dict[str, Any] = {
    # identity / meta
    "root": None,
    "cluster": None,
    "stage_dir": None,
    "experiment_id": None,
    "experiment_name": None,
    "stage_id": None,
    "stage_name": None,
    "source": None,  # "summary", "stage_json", "snapshot", "codecarbon_only"

    # snapshot info
    "is_snapshot": False,
    "snapshot_step": None,
    "snapshot_type": None,
    "snapshot_time": None,

    # config metadata
    "total_energy_policy": None,
    "pipeline": None,
    "student_size": None,
    "dataset_choice": None,
    "kd_temperature": None,
    "kd_alpha": None,
    "sft_max_new_tokens": None,

    # timing / tokens
    "start_time": None,
    "end_time": None,
    "duration_seconds": None,
    "tokens_processed": None,
    "tokens_per_second": None,

    # GPU metrics
    "gpu_energy_joules": None,
    "gpu_avg_power_watts": None,
    "gpu_peak_power_watts": None,
    "nvml_poll_interval_ms": None,

    # CPU + total
    "cpu_energy_joules": None,
    "total_energy_joules": None,
    "total_energy_kwh": None,
    "joules_per_token": None,
    "kwh_total": None,

    # CodeCarbon normalized
    "total_codecarbon_energy_kwh": None,
    "codecarbon_emissions_kg": None,
    "codecarbon_cpu_energy_kwh": None,
    "codecarbon_gpu_energy_kwh": None,
    "codecarbon_ram_energy_kwh": None,
}


def _normalize_stage_metrics_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a StageMetrics-like dict (from stage JSON or experiment_summary)
    into the canonical keys in STAGE_DEFAULTS (no root/cluster/stage_dir/source).
    """
    out = {}

    # Basic identifiers
    out["stage_id"] = raw.get("stage_id")
    out["stage_name"] = raw.get("stage_name")

    # Timing / tokens
    out["start_time"] = raw.get("start_time")
    out["end_time"] = raw.get("end_time")
    out["duration_seconds"] = raw.get("duration_seconds")
    out["tokens_processed"] = raw.get("tokens_processed")
    out["tokens_per_second"] = raw.get("tokens_per_second")

    # GPU
    out["gpu_energy_joules"] = raw.get("gpu_energy_joules")
    out["gpu_avg_power_watts"] = raw.get("gpu_avg_power_watts")
    out["gpu_peak_power_watts"] = raw.get("gpu_peak_power_watts")
    out["nvml_poll_interval_ms"] = raw.get("nvml_poll_interval_ms")

    # CPU
    out["cpu_energy_joules"] = raw.get("cpu_energy_joules")

    # CodeCarbon variants:
    # new-style: total_codecarbon_energy_kwh
    # old-style:  codecarbon_energy_kwh
    cc_total = raw.get("total_codecarbon_energy_kwh", None)
    if cc_total is None:
        cc_total = raw.get("codecarbon_energy_kwh", None)
    out["total_codecarbon_energy_kwh"] = cc_total

    out["codecarbon_emissions_kg"] = raw.get("codecarbon_emissions_kg")
    out["codecarbon_cpu_energy_kwh"] = raw.get("codecarbon_cpu_energy_kwh")
    out["codecarbon_gpu_energy_kwh"] = raw.get("codecarbon_gpu_energy_kwh")
    out["codecarbon_ram_energy_kwh"] = raw.get("codecarbon_ram_energy_kwh")

    # Totals / derived
    out["total_energy_joules"] = raw.get("total_energy_joules")
    out["total_energy_kwh"] = raw.get("total_energy_kwh")
    out["joules_per_token"] = raw.get("joules_per_token")
    out["kwh_total"] = raw.get("kwh_total")

    # Snapshot info (may or may not be present)
    out["is_snapshot"] = bool(raw.get("snapshot", False))
    out["snapshot_step"] = raw.get("snapshot_step")
    out["snapshot_type"] = raw.get("snapshot_type")
    out["snapshot_time"] = raw.get("snapshot_time")

    return out


# In[ ]:


# Config Metadata extraction

def _infer_pipeline_and_student(exp_name: str) -> (Optional[str], Optional[str]):
    s = exp_name.lower()
    pipeline = None
    if s.startswith("kd_"):
        pipeline = "kd"
    elif s.startswith("sft_"):
        pipeline = "sft"
    elif "synthetic" in s:
        pipeline = "synthetic_sft"

    student_size = None
    if "to_1b" in s:
        student_size = "1B"
    elif "to_7b" in s:
        student_size = "7B"
    elif "to_13b" in s or "13b" in s:
        student_size = "13B"

    return pipeline, student_size


def load_config_meta(log_roots: List[Path]) -> pd.DataFrame:
    """
    Scan all config_*.json files and extract per-(root, stage_dir, stage_name) metadata:
        experiment_name, total_energy_policy, pipeline, student_size, kd_temperature, kd_alpha,
        sft_max_new_tokens, dataset_choice, etc.
    """
    rows: List[Dict[str, Any]] = []

    for root in log_roots:
        cluster = ROOT_CLUSTER.get(root.name, root.name)
        for cfg_path in root.rglob("config_*.json"):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read config at {cfg_path}: {e}")
                continue

            stage_name = cfg.get("stage_name")
            stage_id = cfg.get("stage_id")

            config = cfg.get("config", {})
            exp_cfg = config.get("experiment", {})
            data_cfg = config.get("data", {})
            train_cfg = config.get("training", {})
            kd_cfg = config.get("kd", config.get("distillation", {}))  # handle naming
            energy_cfg = config.get("energy", {})

            exp_name = exp_cfg.get("name", stage_name)
            pipeline, student_size = _infer_pipeline_and_student(exp_name)

            rows.append(
                {
                    "root": str(root),
                    "cluster": cluster,
                    "stage_dir": str(cfg_path.parent),
                    "stage_name": stage_name,
                    "stage_id": stage_id,
                    "experiment_name": exp_name,
                    "total_energy_policy": energy_cfg.get("total_energy_policy"),
                    "pipeline": pipeline,
                    "student_size": student_size,
                    "dataset_choice": data_cfg.get("dataset_choice"),
                    "kd_temperature": kd_cfg.get("temperature"),
                    "kd_alpha": kd_cfg.get("alpha"),
                    "sft_max_new_tokens": train_cfg.get("max_new_tokens"),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# In[ ]:


# Stage folder -> records

def _collect_from_experiment_summary(
    summary_path: Path,
    root: Path,
    cluster: str,
    cfg_meta: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Given an experiment_summary.json, return a list of normalized stage records (source='summary').
    """
    records: List[Dict[str, Any]] = []

    with open(summary_path) as f:
        summary = json.load(f)

    exp_id = summary.get("experiment_id")
    exp_name = summary.get("experiment_name")
    stages = summary.get("stages", {})

    for stage_name, raw in stages.items():
        base = dict(STAGE_DEFAULTS)
        base["root"] = str(root)
        base["cluster"] = cluster
        base["stage_dir"] = str(summary_path.parent)
        base["experiment_id"] = exp_id
        base["experiment_name"] = exp_name
        base["source"] = "summary"

        # Normalize metrics
        norm = _normalize_stage_metrics_dict(raw)
        base.update(norm)

        # Attach config meta if available (match by stage_dir + stage_name)
        m = cfg_meta[
            (cfg_meta["root"] == str(root))
            & (cfg_meta["stage_dir"] == str(summary_path.parent))
            & (cfg_meta["stage_name"] == stage_name)
        ]
        if not m.empty:
            meta_row = m.iloc[0].to_dict()
            for k in [
                "total_energy_policy",
                "pipeline",
                "student_size",
                "dataset_choice",
                "kd_temperature",
                "kd_alpha",
                "sft_max_new_tokens",
            ]:
                base[k] = meta_row.get(k)

        records.append(base)

    return records


def _is_stage_metrics_json(path: Path) -> bool:
    """
    Heuristic: JSON files that look like StageMetrics but are not config/env/summary.
    Includes snapshots.
    """
    name = path.name
    if not name.endswith(".json"):
        return False
    if name.startswith("config_") or name.startswith("environment_"):
        return False
    if name == "experiment_summary.json":
        return False
    # This will match stage.json and stage__step_*.json (snapshots)
    return True


def _collect_stage_jsons_in_dir(
    stage_dir: Path,
    root: Path,
    cluster: str,
    cfg_meta: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Collect stage records from standalone stage JSON files in a stage directory
    (finals + snapshots), using source='stage_json' or 'snapshot'.
    """
    records: List[Dict[str, Any]] = []

    # Try to find associated experiment_name and config meta for this dir
    m_dir = cfg_meta[
        (cfg_meta["root"] == str(root)) & (cfg_meta["stage_dir"] == str(stage_dir))
    ]
    cfg_row = m_dir.iloc[0].to_dict() if not m_dir.empty else {}

    for path in stage_dir.glob("*.json"):
        if not _is_stage_metrics_json(path):
            continue

        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read stage JSON at {path}: {e}")
            continue

        base = dict(STAGE_DEFAULTS)
        base["root"] = str(root)
        base["cluster"] = cluster
        base["stage_dir"] = str(stage_dir)
        base["experiment_name"] = cfg_row.get("experiment_name")
        base["source"] = "snapshot" if raw.get("snapshot") else "stage_json"

        norm = _normalize_stage_metrics_dict(raw)
        base.update(norm)

        # Config meta
        for k in [
            "total_energy_policy",
            "pipeline",
            "student_size",
            "dataset_choice",
            "kd_temperature",
            "kd_alpha",
            "sft_max_new_tokens",
        ]:
            base[k] = cfg_row.get(k)

        records.append(base)

    return records



# In[ ]:


# Full-stage Dataframe

def build_stage_dataframe(log_roots: List[Path]) -> pd.DataFrame:
    """
    Main entry point:
      - loads config metadata,
      - walks all log roots,
      - collects StageMetrics from experiment_summary.json and individual stage JSONs,
      - returns one big DataFrame with standardized columns.
    """
    cfg_meta = load_config_meta(log_roots)
    cc_df = load_codecarbon_logs(log_roots)  # not yet used as fallback, but available

    all_records: List[Dict[str, Any]] = []

    for root in log_roots:
        cluster = ROOT_CLUSTER.get(root.name, root.name)
        if not root.exists():
            continue

        # 1) experiment_summary.json files (per stage or sometimes per pipeline)
        for summary_path in root.rglob("experiment_summary.json"):
            # You might want to skip top-level ones (like trillium-logs/experiment_summary.json)
            # if they are "meta" – check structure if needed.
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read experiment_summary at {summary_path}: {e}")
                continue

            if "stages" in summary:
                all_records.extend(
                    _collect_from_experiment_summary(summary_path, root, cluster, cfg_meta)
                )
            else:
                # Some summaries might be in an older/global format; skip or handle specially.
                pass

        # 2) Standalone stage directories: often under root/stages/*, but also
        #    top-level experiment dirs like trillium-logs/sft_32b_to_1b_tulu_3500
        for stage_dir in root.rglob("*"):
            if not stage_dir.is_dir():
                continue

            # Heuristic: a "stage dir" is one that contains some StageMetrics JSON
            has_stage_json = any(_is_stage_metrics_json(p) for p in stage_dir.glob("*.json"))
            if not has_stage_json:
                continue

            # We already handled those covered by experiment_summary.json – but that's okay.
            # The downstream logic can deduplicate or prefer summary over stage_json if needed.
            records = _collect_stage_jsons_in_dir(stage_dir, root, cluster, cfg_meta)
            all_records.extend(records)

    if not all_records:
        return pd.DataFrame(columns=STAGE_DEFAULTS.keys())

    stage_df = pd.DataFrame(all_records)

    # Optional: deduplicate (e.g., you might want to drop stage_json records
    # that correspond exactly to summary records). For now, keep everything
    # and let later analysis decide which to use.
    return stage_df


# In[ ]:


df = build_stage_dataframe(LOG_ROOTS)
print(df.head())
df.to_csv("stage_metrics_standardized.csv", index=False)
print(f"Saved stage_metrics_standardized.csv with {len(df)} rows.")


# ## Create Dataframe

# In[ ]:


import json
import pandas as pd

def load_stage_df(log_roots):
    rows = []

    for root in log_roots:
        for summary_path in root.rglob("experiment_summary.json"):
            with open(summary_path) as f:
                summary = json.load(f)

            cluster = ROOT_CLUSTER.get(root.name, root.name)

            exp_id = summary.get("experiment_id")
            exp_name = summary.get("experiment_name")

            # New-style: stages dict
            stages = summary.get("stages", {})

            for stage_name, s in stages.items():
                rows.append({
                    "cluster": cluster,
                    "root": str(root),
                    "exp_dir": str(summary_path.parent),
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "stage_name": stage_name,

                    # core metrics
                    "duration_s": s.get("duration_seconds"),
                    "tokens": s.get("tokens_processed"),
                    "total_energy_j": s.get("total_energy_joules"),
                    "total_energy_kwh": s.get("total_energy_kwh"),
                    "j_per_token": s.get("joules_per_token"),
                    "tps": s.get("tokens_per_second"),

                    # gpu detail
                    "gpu_energy_j": s.get("gpu_energy_joules"),
                    "gpu_avg_power_w": s.get("gpu_avg_power_watts"),
                    "gpu_peak_power_w": s.get("gpu_peak_power_watts"),

                    # cpu if present
                    "cpu_energy_j": s.get("cpu_energy_joules", 0.0),
                })

    return pd.DataFrame(rows)

stage_df = load_stage_df(LOG_ROOTS)


# In[5]:


stage_df


# In[ ]:


# Ingest Logs into dataframe
import json, glob, pathlib
import pandas as pd

def load_all_experiment_summaries(root="."):
    rows = []
    for path in glob.glob(f"{root}/**/experiment_summary.json", recursive=True):
        with open(path) as f:
            summary = json.load(f)
        exp_id = summary["experiment_id"]
        exp_name = summary["experiment_name"]
        for stage_name, s in summary["stages"].items():
            rows.append({
                "exp_id": exp_id,
                "exp_name": exp_name,
                "stage_name": stage_name,

                # core metrics
                "duration_s": s["duration_seconds"],
                "tokens": s["tokens_processed"],
                "total_energy_j": s["total_energy_joules"],
                "total_energy_kwh": s["total_energy_kwh"],
                "j_per_token": s["joules_per_token"],
                "tps": s["tokens_per_second"],

                # gpu detail
                "gpu_energy_j": s["gpu_energy_joules"],
                "gpu_avg_power_w": s["gpu_avg_power_watts"],
                "gpu_peak_power_w": s["gpu_peak_power_watts"],

                # cpu if available
                "cpu_energy_j": s.get("cpu_energy_joules", 0.0),
            })
    return pd.DataFrame(rows)

stage_df = load_all_experiment_summaries("/path/to/your/log/root")


# In[ ]:


# Label each stage with a role
def infer_stage_role(stage_name: str) -> str:
    s = stage_name.lower()
    if "preprocess" in s:
        return "teacher_preprocess"
    if "synthetic" in s or "generation" in s:
        return "teacher_generation"
    if "logprob" in s or "logit" in s or "cache" in s:
        return "teacher_logit_cache"
    if "eval" in s or "benchmark" in s:
        return "eval"
    # default: main training stage
    return "student_train"

stage_df["stage_role"] = stage_df["stage_name"].apply(infer_stage_role)


# In[ ]:


# Attach Metadata
def load_config_metadata(root="."):
    rows = []
    for path in glob.glob(f"{root}/**/config_*.json", recursive=True):
        with open(path) as f:
            cfg = json.load(f)
        stage_name = cfg["stage_name"]
        exp_cfg = cfg["config"]["experiment"]
        data_cfg = cfg["config"]["data"]
        train_cfg = cfg["config"].get("training", {})
        kd_cfg = cfg["config"].get("kd", {})

        exp_name = exp_cfg["name"]
        # heuristic: parse pipeline + student size from exp_name
        # e.g. "kd_olmo2_32b_to_13b_nosft"
        s = exp_name.lower()
        if s.startswith("kd_"):
            pipeline = "kd"
        elif s.startswith("sft_"):
            pipeline = "sft"
        elif "synthetic" in s:
            pipeline = "synthetic_sft"
        else:
            pipeline = "other"

        # crude parse of student size from name; you can refine
        student_size = None
        if "to_1b" in s:
            student_size = "1B"
        elif "to_7b" in s:
            student_size = "7B"
        elif "to_13b" in s or "13b" in s:
            student_size = "13B"

        rows.append({
            "exp_name": exp_name,
            "stage_name": stage_name,
            "pipeline": pipeline,
            "student_size": student_size,
            # KD hyperparams if present
            "kd_temperature": kd_cfg.get("temperature"),
            "kd_alpha": kd_cfg.get("alpha"),
            # SFT hyperparams
            "sft_max_new_tokens": train_cfg.get("max_new_tokens"),
            "dataset_choice": data_cfg.get("dataset_choice"),
        })

    return pd.DataFrame(rows)

cfg_meta = load_config_metadata("/path/to/your/log/root")



# In[ ]:


# Merge metadata into stage_df
stage_df = stage_df.merge(
    cfg_meta,
    on=["exp_name", "stage_name"],
    how="left",
)


# ## Experiment-Level Metrics

# In[ ]:


teacher_roles = {"teacher_preprocess", "teacher_generation", "teacher_logit_cache"}


# In[ ]:


# Aggregate metrics across stages

agg_cols = ["total_energy_kwh", "total_energy_joules", "tokens", "duration_s"]

# Teacher-only totals
teacher = (
    stage_df[stage_df["stage_role"].isin(teacher_roles)]
    .groupby(["exp_id", "exp_name"])
    [agg_cols]
    .sum()
    .add_prefix("teacher_")
)

# Student-only totals (everything else, including eval)
student = (
    stage_df[~stage_df["stage_role"].isin(teacher_roles)]
    .groupby(["exp_id", "exp_name"])
    [agg_cols]
    .sum()
    .add_prefix("student_")
)

# Combine into run-level df
run_df = teacher.join(student, how="outer").reset_index()

# Fill NaNs where experiments don't have teacher stages, etc.
run_df = run_df.fillna(0.0)


# In[ ]:


# Derive key per-run metrics

# Total pipeline energy
run_df["total_energy_kwh"] = (
    run_df["teacher_total_energy_kwh"] + run_df["student_total_energy_kwh"]
)
run_df["total_energy_j"] = (
    run_df["teacher_total_energy_joules"] + run_df["student_total_energy_joules"]
)

# Total tokens (for per-token normalization)
run_df["total_tokens"] = run_df["teacher_tokens"] + run_df["student_tokens"]

# Per-token metrics
run_df["student_j_per_token"] = run_df["student_total_energy_joules"] / run_df["student_tokens"]
run_df["teacher_j_per_token"] = run_df["teacher_total_energy_joules"] / run_df["teacher_tokens"]
run_df["overall_j_per_token"] = run_df["total_energy_j"] / run_df["total_tokens"]

# Throughput and avg power
run_df["total_duration_s"] = run_df["teacher_duration_s"] + run_df["student_duration_s"]
run_df["overall_tps"] = run_df["total_tokens"] / run_df["total_duration_s"]
run_df["overall_avg_power_w"] = run_df["total_energy_j"] / run_df["total_duration_s"]


# In[ ]:


# Group
# Pull a single representative row (pipeline, student_size, etc.) per experiment
meta_per_exp = (
    stage_df
    .groupby(["exp_id", "exp_name"])
    .agg({
        "pipeline": "first",
        "student_size": "first",
        "dataset_choice": "first",
        "kd_temperature": "first",
        "kd_alpha": "first",
        "sft_max_new_tokens": "first",
    })
    .reset_index()
)

run_df = run_df.merge(meta_per_exp, on=["exp_id", "exp_name"], how="left")


# ## Results & Tables

# In[ ]:


# Choose which runs to include in the "core 3x3", e.g. main config only
core = run_df.query("pipeline in ['sft', 'kd', 'synthetic_sft'] and student_size in ['1B','7B','13B']")

# If you have multiple seeds per cell, pick median energy or best-quality run, etc.
# Example: median by energy
core_med = (
    core
    .groupby(["pipeline", "student_size"])
    .agg({
        "total_energy_kwh": "median",
        "overall_avg_power_w": "median",
        "overall_j_per_token": "median",
        "student_j_per_token": "median",
        "overall_tps": "median",
    })
    .reset_index()
)


# In[ ]:


# Pivot into grids for tables
energy_grid = core_med.pivot(index="pipeline", columns="student_size", values="total_energy_kwh")
jpt_grid    = core_med.pivot(index="pipeline", columns="student_size", values="overall_j_per_token")
power_grid  = core_med.pivot(index="pipeline", columns="student_size", values="overall_avg_power_w")


# In[ ]:


# Compare against baseline SFT by normalizing
# Merge SFT baseline onto other pipelines for same student_size
sft_baseline = core_med[core_med["pipeline"] == "sft"][["student_size", "total_energy_kwh"]].rename(columns={"total_energy_kwh": "sft_energy_kwh"})

core_med = core_med.merge(sft_baseline, on="student_size", how="left")
core_med["energy_vs_sft"] = core_med["total_energy_kwh"] / core_med["sft_energy_kwh"]


# ### Pareto Frontiers

# In[ ]:


# Hook into benchmarks
# model_id = maybe exp_name or a path to weights, mapped back to exp_name
bench_df = pd.DataFrame([
    # one row per (experiment, benchmark)
    # exp_name, benchmark, score
])

# Option 1: a single scalar per model (e.g. MT-Bench-101 or composite)
quality = (
    bench_df
    .groupby("exp_name")
    .agg({"score": "mean"})  # or a weighted combo
    .rename(columns={"score": "quality_score"})
    .reset_index()
)

run_q = run_df.merge(quality, on="exp_name", how="left")


# In[ ]:


# For teacher-ignored vs teacher-included
run_q["energy_teacher_ignored"] = run_q["student_total_energy_kwh"]
run_q["energy_teacher_included"] = run_q["total_energy_kwh"]


# In[ ]:


def pareto_front(df, energy_col="total_energy_kwh", quality_col="quality_score"):
    # smaller energy is better, higher quality is better
    points = df.sort_values(energy_col).reset_index(drop=True)
    best_q = -float("inf")
    mask = []
    for _, row in points.iterrows():
        if row[quality_col] > best_q:
            best_q = row[quality_col]
            mask.append(True)
        else:
            mask.append(False)
    return points[mask]

frontier = pareto_front(run_q.query("pipeline == 'kd'"))


# ### Stage-wise breakdown

# In[ ]:


# Sum energy per role × pipeline × student size
stage_energy = (
    stage_df
    .groupby(["pipeline", "student_size", "stage_role"])
    ["total_energy_kwh"]
    .sum()
    .reset_index()
)


# ### Sensitivity Analysis

# In[ ]:


kd = run_q.query("pipeline == 'kd'")

# Example: line plots of energy vs quality for different temperatures
# One panel per τ, x-axis alpha, y-axis quality / energy, etc.

# For a simple table:
kd_summary = (
    kd.groupby(["student_size", "kd_temperature", "kd_alpha"])
    .agg({
        "total_energy_kwh": "median",
        "overall_j_per_token": "median",
        "quality_score": "median",
    })
    .reset_index()
)


# In[ ]:


sft = run_q.query("pipeline == 'sft'")

sft_summary = (
    sft.groupby(["student_size", "sft_max_new_tokens", "dataset_choice"])
    .agg({
        "total_energy_kwh": "median",
        "overall_j_per_token": "median",
        "quality_score": "median",
    })
    .reset_index()
)


# ## Latex

# In[ ]:


# Suppose core_med has columns:
# pipeline, student_size, total_energy_kwh, overall_j_per_token, overall_avg_power_w

# Pivot to a 3×3 grid of total kWh
energy_grid = core_med.pivot(index="pipeline", columns="student_size", values="total_energy_kwh")

latex_table = energy_grid.to_latex(
    index=True,
    float_format="%.3f",
    caption="Total energy (kWh) for each pipeline and student size.",
    label="tab:energy_core_grid",
    escape=False,
    bold_rows=False,
    column_format="lccc",  # adjust as needed
    longtable=False,
    multicolumn=True,
    multicolumn_format='c',
    na_rep="--",
    bold_header=True if hasattr(energy_grid, 'style') else False,  # optional
    buf=None,
    header=True,
    show_dimensions=False
)

with open("tables/energy_core_grid.tex", "w") as f:
    f.write(latex_table)


# In Latex:
# ```
# \begin{table}[t]
#     \centering
#     \input{tables/energy_core_grid.tex}
#     \vspace{-0.5em}
# \end{table}
# ```
# 

# ```
# \begin{table}[t]
#   \centering
#   \small
#   \setlength{\tabcolsep}{4pt}
#   \input{tables/energy_core_grid.tex}
#   \caption{Total energy (kWh) per pipeline and student size. Lower is better.}
#   \label{tab:energy_core_grid}
# \end{table}
# ```

# In[ ]:




