#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


# In[ ]:


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
    elif "true" in s:
        pipeline = "true_sft"

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
        # Default: parent of the summary (e.g., run_dir); overridden if config meta is found
        base["stage_dir"] = str(summary_path.parent)
        base["experiment_id"] = exp_id
        base["experiment_name"] = exp_name
        base["source"] = "summary"

        # Normalize metrics
        norm = _normalize_stage_metrics_dict(raw)
        base.update(norm)

        # Attach config meta if available.
        # Match by root + stage_name, then prefer the config's stage_dir.
        m = cfg_meta[
            (cfg_meta["root"] == str(root))
            & (cfg_meta["stage_name"] == stage_name)
        ]
        if not m.empty:
            meta_row = m.iloc[0].to_dict()

            # Prefer the config's notion of the stage_dir (actual stage folder)
            stage_dir_cfg = meta_row.get("stage_dir")
            if stage_dir_cfg:
                base["stage_dir"] = stage_dir_cfg

            # Optionally override stage_id if missing
            if base.get("stage_id") is None and meta_row.get("stage_id"):
                base["stage_id"] = meta_row["stage_id"]

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
    Collect stage records from standalone stage JSON files in a stage directory,
    aggregate snapshots, and return ONE row per logical stage.

    Rules:
      - If a final stage JSON exists (source='stage_json', not a snapshot),
        use that as the base row.
      - If only snapshots exist, pick the latest snapshot (by snapshot_step, then end_time).
      - For stages with both final and snapshots, final wins; we can still
        use the last snapshot to fill missing fields if needed.
    """
    # Match config for this directory (pipeline, student_size, etc.)
    m_dir = cfg_meta[
        (cfg_meta["root"] == str(root)) & (cfg_meta["stage_dir"] == str(stage_dir))
    ]
    cfg_row = m_dir.iloc[0].to_dict() if not m_dir.empty else {}

    stage_records: List[Dict[str, Any]] = []

    for path in stage_dir.glob("*.json"):
        if not _is_stage_metrics_json(path):
            continue

        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read stage JSON at {path}: {e}")
            continue

        # Skip JSONs that aren't dicts (or single-element list of dict)
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], dict):
                raw = raw[0]
            else:
                print(
                    f"[INFO] Skipping JSON at {path} "
                    f"(top-level list, not a StageMetrics dict)"
                )
                continue
        elif not isinstance(raw, dict):
            print(
                f"[INFO] Skipping JSON at {path} "
                f"(top-level {type(raw).__name__}, expected dict)"
            )
            continue

        base = dict(STAGE_DEFAULTS)
        base["root"] = str(root)
        base["cluster"] = cluster
        base["stage_dir"] = str(stage_dir)
        base["experiment_name"] = cfg_row.get("experiment_name")
        base["source"] = "snapshot" if raw.get("snapshot") else "stage_json"

        # Normalize StageMetrics-style dict into our standard fields
        norm = _normalize_stage_metrics_dict(raw)
        base.update(norm)

        # If JSON didn't carry stage_name, fall back to folder name
        if not base.get("stage_name"):
            base["stage_name"] = stage_dir.name

        # Attach config meta
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

        stage_records.append(base)

    if not stage_records:
        return []

    # --- Aggregate to ONE row per logical stage in this directory ---

    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for rec in stage_records:
        key = rec.get("stage_id") or rec["stage_name"]
        by_stage.setdefault(key, []).append(rec)

    aggregated: List[Dict[str, Any]] = []

    for key, recs in by_stage.items():
        finals = [
            r
            for r in recs
            if r.get("source") != "snapshot" and not r.get("is_snapshot", False)
        ]
        snapshots = [r for r in recs if r.get("source") == "snapshot"]

        if finals:
            # Prefer the final metrics JSON; if multiple, take the one with the latest end_time.
            best = max(finals, key=lambda r: (r.get("end_time") or 0.0))

            # Optional: use the latest snapshot as a fallback for missing fields.
            if snapshots:
                snaps_sorted = sorted(
                    snapshots,
                    key=lambda r: (
                        r.get("snapshot_step") if r.get("snapshot_step") is not None else -1,
                        r.get("end_time") or 0.0,
                    ),
                )
                last_snap = snaps_sorted[-1]
                for field in STAGE_DEFAULTS.keys():
                    if best.get(field) in (None, 0) and last_snap.get(field) not in (None, 0):
                        best[field] = last_snap[field]

            aggregated.append(best)
        else:
            # No final file: only snapshots. Pick the latest snapshot as the representative row.
            snaps_sorted = sorted(
                recs,
                key=lambda r: (
                    r.get("snapshot_step") if r.get("snapshot_step") is not None else -1,
                    r.get("end_time") or 0.0,
                ),
            )
            aggregated.append(snaps_sorted[-1])

    return aggregated



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

        # 1) experiment_summary.json files (per run)
        for summary_path in root.rglob("experiment_summary.json"):
            # Skip copies written into individual stage dirs:
            # .../<root>/stages/<stage>/experiment_summary.json
            parent = summary_path.parent
            if parent.parent.name == "stages":
                continue

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
        for stage_dir in root.rglob("*"):
            if not stage_dir.is_dir():
                continue

            # Skip the container folder itself (we only want its children)
            if stage_dir == root / "stages":
                continue

            # Heuristic: a "stage dir" is one that contains some StageMetrics JSON
            has_stage_json = any(_is_stage_metrics_json(p) for p in stage_dir.glob("*.json"))
            if not has_stage_json:
                continue

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


def build_stage_dataframe_for_path(path: str | Path) -> pd.DataFrame:
    """
    Convenience helper to build a standardized stage DataFrame for a specific
    log root or stage directory.

    Examples
    --------
    - build_stage_dataframe_for_path("runpod2_logs")
    - build_stage_dataframe_for_path("runpod2_logs/stages/sft_32b_to_13b_tulu_nosft")
    - build_stage_dataframe_for_path("/abs/path/to/runpod2_logs/stages/sft_32b_to_13b_tulu_nosft")
    """
    path = Path(path).resolve()

    # If they passed a specific stage dir under .../stages/<stage_name>
    if path.is_dir() and path.name != "stages" and path.parent.name == "stages":
        # /.../<log_root>/stages/<stage_name>
        # For /project/.../Energy/runpod2_logs/stages/sft_32b_to_1b_math_nosft
        # we want log_root = /project/.../Energy/runpod2_logs
        log_root = path.parent.parent  # == path.parents[1]
        filter_prefix = str(path)
    elif path.is_dir() and path.name == "stages":
        # They pointed at the stages/ directory: restrict to that subtree
        log_root = path.parent
        filter_prefix = str(path)
    else:
        # Treat as a log root
        log_root = path
        filter_prefix = str(log_root)

    df = build_stage_dataframe([log_root])

    if df.empty:
        return df

    # If they gave a root, no extra filtering
    if filter_prefix == str(log_root):
        return df.reset_index(drop=True)

    # Otherwise restrict to that specific stage subtree
    stage_dirs = df["stage_dir"].astype(str)
    mask = stage_dirs.str.startswith(filter_prefix)
    return df[mask].reset_index(drop=True)


# In[ ]:


# Choose what you want to process
# - Leave `paths` empty to use default LOG_ROOTS
# - Or set it to one or more specific paths, e.g. a single stage dir
paths = [
    "/home/klambert/projects/aip-craffel/klambert/Energy/runpod2_logs/",
    "/home/klambert/projects/aip-craffel/klambert/Energy/logs",
    "/home/klambert/projects/aip-craffel/klambert/Energy/trillium-logs",    
    ]
output = "stage_metrics.csv"

if paths:
    dfs = [build_stage_dataframe_for_path(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
else:
    df = build_stage_dataframe(LOG_ROOTS)

output_path = Path(output)
file_exists = output_path.exists()

display(df.head())
df.to_csv(
    output_path,
    mode="a" if file_exists else "w",   # append if exists, else write
    header=not file_exists,            # write header only if new file
    index=False,
)
print(f"Saved {output} with {len(df)} rows.")


# --------------------
# ## Create Dataframe

# In[28]:


import numpy as np
from pathlib import Path

# Path to the aggregated stage metrics CSV
if "output_path" in globals():
    stage_metrics_path = output_path
else:
    stage_metrics_path = Path("stage_metrics.csv")

stage_df_raw = pd.read_csv(stage_metrics_path)
print(f"\n=== Loaded aggregated stage metrics ===")
print(f"Path: {stage_metrics_path}")
print(f"Rows: {len(stage_df_raw)}")
print(f"Columns: {len(stage_df_raw.columns)}")
print("First few columns:", list(stage_df_raw.columns[:10]))

# -------------------------------------------------------------------------
# Ensure numeric dtypes for core metric columns
# -------------------------------------------------------------------------
numeric_cols = [
    "duration_seconds",
    "tokens_processed",
    "tokens_per_second",
    "gpu_energy_joules",
    "gpu_avg_power_watts",
    "gpu_peak_power_watts",
    "nvml_poll_interval_ms",
    "cpu_energy_joules",
    "total_energy_joules",
    "total_energy_kwh",
    "joules_per_token",
    "kwh_total",
    "total_codecarbon_energy_kwh",
    "codecarbon_emissions_kg",
    "codecarbon_cpu_energy_kwh",
    "codecarbon_gpu_energy_kwh",
    "codecarbon_ram_energy_kwh",
]

for col in numeric_cols:
    if col in stage_df_raw.columns:
        stage_df_raw[col] = pd.to_numeric(stage_df_raw[col], errors="coerce")

stage_df_all = stage_df_raw.copy()

print("\n=== Non-null counts for numeric columns ===")
print(stage_df_all[numeric_cols].notna().sum().sort_values())

# -------------------------------------------------------------------------
# Canonical energy / throughput columns
# -------------------------------------------------------------------------

# Best-effort total energy in kWh
energy_kwh = None
energy_sources_used = []

if "total_energy_kwh" in stage_df_all.columns:
    energy_kwh = stage_df_all["total_energy_kwh"].copy()
    energy_sources_used.append("total_energy_kwh")

if "total_codecarbon_energy_kwh" in stage_df_all.columns:
    if energy_kwh is None:
        energy_kwh = stage_df_all["total_codecarbon_energy_kwh"].copy()
    else:
        missing_mask = energy_kwh.isna() & stage_df_all["total_codecarbon_energy_kwh"].notna()
        energy_kwh = energy_kwh.fillna(stage_df_all["total_codecarbon_energy_kwh"])
        if missing_mask.any():
            print(
                f"[INFO] Filled energy_kwh from total_codecarbon_energy_kwh "
                f"for {missing_mask.sum()} rows"
            )
    energy_sources_used.append("total_codecarbon_energy_kwh")

if "kwh_total" in stage_df_all.columns:
    if energy_kwh is None:
        energy_kwh = stage_df_all["kwh_total"].copy()
    else:
        missing_mask = energy_kwh.isna() & stage_df_all["kwh_total"].notna()
        energy_kwh = energy_kwh.fillna(stage_df_all["kwh_total"])
        if missing_mask.any():
            print(
                f"[INFO] Filled energy_kwh from kwh_total "
                f"for {missing_mask.sum()} rows"
            )
    energy_sources_used.append("kwh_total")

if energy_kwh is None:
    stage_df_all["energy_kwh"] = np.nan
    print("[WARN] Could not construct energy_kwh from any known columns.")
else:
    stage_df_all["energy_kwh"] = energy_kwh

print("\n=== energy_kwh summary ===")
print("Non-null rows:", stage_df_all["energy_kwh"].notna().sum())
print("Rows with NaN energy_kwh:", stage_df_all["energy_kwh"].isna().sum())
if energy_sources_used:
    print("Sources considered for energy_kwh:", ", ".join(energy_sources_used))

# GPU energy in kWh: prefer direct Joules, fall back to CodeCarbon
stage_df_all["gpu_energy_kwh"] = np.nan
if "gpu_energy_joules" in stage_df_all.columns:
    mask_gpu_j = stage_df_all["gpu_energy_joules"].notna()
    stage_df_all.loc[mask_gpu_j, "gpu_energy_kwh"] = (
        stage_df_all.loc[mask_gpu_j, "gpu_energy_joules"] / 3.6e6
    )
    print(
        f"[INFO] Converted gpu_energy_joules -> gpu_energy_kwh "
        f"for {mask_gpu_j.sum()} rows"
    )

if "codecarbon_gpu_energy_kwh" in stage_df_all.columns:
    mask_fill = stage_df_all["gpu_energy_kwh"].isna() & stage_df_all["codecarbon_gpu_energy_kwh"].notna()
    stage_df_all.loc[mask_fill, "gpu_energy_kwh"] = stage_df_all.loc[
        mask_fill, "codecarbon_gpu_energy_kwh"
    ]
    print(
        f"[INFO] Filled gpu_energy_kwh from codecarbon_gpu_energy_kwh "
        f"for {mask_fill.sum()} rows"
    )

# CPU energy in kWh: prefer direct Joules, fall back to CodeCarbon
stage_df_all["cpu_energy_kwh"] = np.nan
if "cpu_energy_joules" in stage_df_all.columns:
    mask_cpu_j = stage_df_all["cpu_energy_joules"].notna()
    stage_df_all.loc[mask_cpu_j, "cpu_energy_kwh"] = (
        stage_df_all.loc[mask_cpu_j, "cpu_energy_joules"] / 3.6e6
    )
    print(
        f"[INFO] Converted cpu_energy_joules -> cpu_energy_kwh "
        f"for {mask_cpu_j.sum()} rows"
    )

if "codecarbon_cpu_energy_kwh" in stage_df_all.columns:
    mask_fill = stage_df_all["cpu_energy_kwh"].isna() & stage_df_all["codecarbon_cpu_energy_kwh"].notna()
    stage_df_all.loc[mask_fill, "cpu_energy_kwh"] = stage_df_all.loc[
        mask_fill, "codecarbon_cpu_energy_kwh"
    ]
    print(
        f"[INFO] Filled cpu_energy_kwh from codecarbon_cpu_energy_kwh "
        f"for {mask_fill.sum()} rows"
    )

# Joules per token: prefer precomputed, else total_energy_joules / tokens_processed
if "joules_per_token" in stage_df_all.columns:
    stage_df_all["energy_j_per_token"] = stage_df_all["joules_per_token"]
else:
    stage_df_all["energy_j_per_token"] = np.nan

mask_need_jpt = stage_df_all["energy_j_per_token"].isna()
if "total_energy_joules" in stage_df_all.columns and "tokens_processed" in stage_df_all.columns:
    denom = stage_df_all["tokens_processed"].replace({0: np.nan})
    jpt_mask = mask_need_jpt & stage_df_all["total_energy_joules"].notna() & denom.notna()
    stage_df_all.loc[jpt_mask, "energy_j_per_token"] = (
        stage_df_all.loc[jpt_mask, "total_energy_joules"] / denom[jpt_mask]
    )
    if jpt_mask.any():
        print(
            f"[INFO] Computed energy_j_per_token as total_energy_joules/tokens_processed "
            f"for {jpt_mask.sum()} rows"
        )

print("\n=== energy_j_per_token summary ===")
print("Non-null rows:", stage_df_all["energy_j_per_token"].notna().sum())

# Tokens per second: prefer precomputed, else tokens_processed / duration_seconds
if "tokens_per_second" in stage_df_all.columns:
    stage_df_all["tokens_per_sec"] = stage_df_all["tokens_per_second"]
else:
    stage_df_all["tokens_per_sec"] = np.nan

mask_need_tps = stage_df_all["tokens_per_sec"].isna()
if "duration_seconds" in stage_df_all.columns and "tokens_processed" in stage_df_all.columns:
    dur = stage_df_all["duration_seconds"].replace({0: np.nan})
    tps_mask = mask_need_tps & dur.notna() & stage_df_all["tokens_processed"].notna()
    stage_df_all.loc[tps_mask, "tokens_per_sec"] = (
        stage_df_all.loc[tps_mask, "tokens_processed"] / dur[tps_mask]
    )
    if tps_mask.any():
        print(
            f"[INFO] Computed tokens_per_sec as tokens_processed/duration_seconds "
            f"for {tps_mask.sum()} rows"
        )

print("\n=== tokens_per_sec summary ===")
print("Non-null rows:", stage_df_all["tokens_per_sec"].notna().sum())

# -------------------------------------------------------------------------
# Filtered view: drop rows that will mess up per-token metrics
# -------------------------------------------------------------------------
if "tokens_processed" in stage_df_all.columns:
    mask_valid_tokens = (
        stage_df_all["tokens_processed"].notna()
        & (stage_df_all["tokens_processed"] > 0)
    )
    stage_df_clean = stage_df_all[mask_valid_tokens].copy()
else:
    stage_df_clean = stage_df_all.copy()

print("\n=== Clean vs all rows ===")
print(f"stage_df_all:   {len(stage_df_all)} rows")
print(
    f"stage_df_clean: {len(stage_df_clean)} rows "
    "(tokens_processed > 0 where available)"
)

display(stage_df_clean.head())


# In[29]:


# ## Tag stages, runs, and cells (Step 2)

def _normalize_str(s: pd.Series, to_lower: bool = True) -> pd.Series:
    """Basic string normalization helper."""
    s = s.astype(str).str.strip()
    # convert literal "nan" (created by astype(str)) back to real NaN
    s = s.replace({"nan": np.nan})
    if to_lower:
        s = s.str.lower()
    return s

# Normalize key identifier columns for easier grouping
for col in ["pipeline", "student_size", "dataset_choice", "stage_name", "experiment_name", "source"]:
    if col in stage_df_all.columns:
        # pipeline & dataset_choice we want consistently lowercased
        to_lower = col in ["pipeline", "dataset_choice", "stage_name", "source"]
        stage_df_all[col] = _normalize_str(stage_df_all[col], to_lower=to_lower)

print("\n=== Identifier column samples ===")
for col in ["pipeline", "student_size", "dataset_choice", "stage_name"]:
    if col in stage_df_all.columns:
        print(f"{col}: {stage_df_all[col].dropna().unique()[:5]}")

# -------------------------------------------------------------------------
# Stage role inference
# -------------------------------------------------------------------------
def infer_stage_role(row: pd.Series) -> str:
    name = str(row.get("stage_name", "") or "").lower()
    pipeline = str(row.get("pipeline", "") or "").lower()
    source = str(row.get("source", "") or "").lower()

    # Synthetic / teacher-side work
    if "synthetic" in name:
        if "generation" in name:
            return "teacher_generation"
        return "teacher_processing"

    # Dataset preprocessing
    if "preprocess" in name:
        return "data_preprocess"

    # Evaluation stages (future-proofed for eval names)
    if (
        "eval" in name
        or "gsm8k" in name
        or "mmlu" in name
        or "alpaca" in name
        or "ifeval" in name
    ):
        return "evaluation"

    # Main training pipelines
    if pipeline in {"sft", "kd", "dpo"}:
        # Distinguish between summary and per-stage json if you want
        if source == "summary":
            return "train_summary"
        return "student_train"

    # Fallbacks
    if source == "summary":
        return "summary_only"

    return "other"

stage_df_all["stage_role"] = stage_df_all.apply(infer_stage_role, axis=1)

print("\n=== stage_role value counts ===")
print(stage_df_all["stage_role"].value_counts())

# -------------------------------------------------------------------------
# Run ID: group stages belonging to the same logical experiment
# -------------------------------------------------------------------------
if "experiment_name" in stage_df_all.columns:
    stage_df_all["run_id"] = stage_df_all["experiment_name"].fillna(
        stage_df_all["stage_name"]
    )
else:
    stage_df_all["run_id"] = stage_df_all["stage_name"]

print("\nUnique run_id count:", stage_df_all["run_id"].nunique())

# -------------------------------------------------------------------------
# 3×3 grid cell ID: pipeline × student_size × dataset_choice
# -------------------------------------------------------------------------
required_for_cell = ["pipeline", "student_size", "dataset_choice"]
for col in required_for_cell:
    if col in stage_df_all.columns:
        stage_df_all[col] = stage_df_all[col].replace("", np.nan)

if all(col in stage_df_all.columns for col in required_for_cell):
    missing_any = stage_df_all[required_for_cell].isna().any(axis=1)
    stage_df_all["cell_id"] = np.where(
        missing_any,
        np.nan,
        (
            stage_df_all["pipeline"].str.lower()
            + "_"
            + stage_df_all["student_size"].astype(str)
            + "_"
            + stage_df_all["dataset_choice"].str.lower()
        ),
    )
else:
    stage_df_all["cell_id"] = np.nan

print("\n=== cell_id summary ===")
print("Non-null cell_id rows:", stage_df_all["cell_id"].notna().sum())
print("Unique cell_ids:", stage_df_all["cell_id"].dropna().nunique())
print(stage_df_all["cell_id"].dropna().unique())

# -------------------------------------------------------------------------
# Keep stage_df_clean in sync with new columns
# -------------------------------------------------------------------------
if "tokens_processed" in stage_df_all.columns:
    valid = stage_df_all["tokens_processed"].notna() & (stage_df_all["tokens_processed"] > 0)
    stage_df_clean = stage_df_all[valid].copy()
else:
    stage_df_clean = stage_df_all.copy()

print("\n=== Preview with tagging columns ===")
display(
    stage_df_all[
        [
            "stage_name",
            "pipeline",
            "student_size",
            "dataset_choice",
            "stage_role",
            "run_id",
            "cell_id",
            "energy_kwh",
            "energy_j_per_token",
            "tokens_per_sec",
        ]
    ].head(15)
)


# In[30]:


# ## Aggregate pipeline-level metrics (Step 3)

print("\n\n=== Step 3: Aggregating pipeline-level (cell) metrics ===")

# -------------------------------------------------------------------------
# Effective energy in Joules for aggregation
# -------------------------------------------------------------------------
if "total_energy_joules" in stage_df_all.columns:
    stage_df_all["energy_joules_eff"] = stage_df_all["total_energy_joules"]
    # Fill missing with kWh-based estimate if available
    if "energy_kwh" in stage_df_all.columns:
        mask_fill_j = stage_df_all["energy_joules_eff"].isna() & stage_df_all["energy_kwh"].notna()
        stage_df_all.loc[mask_fill_j, "energy_joules_eff"] = (
            stage_df_all.loc[mask_fill_j, "energy_kwh"] * 3.6e6
        )
        if mask_fill_j.any():
            print(
                f"[INFO] Filled energy_joules_eff from energy_kwh for "
                f"{mask_fill_j.sum()} rows"
            )
else:
    if "energy_kwh" in stage_df_all.columns:
        stage_df_all["energy_joules_eff"] = stage_df_all["energy_kwh"] * 3.6e6
        print("[INFO] Constructed energy_joules_eff from energy_kwh for all rows.")
    else:
        stage_df_all["energy_joules_eff"] = np.nan
        print("[WARN] No total_energy_joules or energy_kwh; energy_joules_eff is NaN.")

print("\n=== energy_joules_eff summary ===")
print("Non-null rows:", stage_df_all["energy_joules_eff"].notna().sum())
print("Rows with NaN energy_joules_eff:", stage_df_all["energy_joules_eff"].isna().sum())

# -------------------------------------------------------------------------
# Stage-role groupings for aggregation
# -------------------------------------------------------------------------
STUDENT_ROLES = {"student_train", "train_summary"}
TEACHER_ROLES = {"teacher_generation", "teacher_processing", "data_preprocess"}
EVAL_ROLES = {"evaluation"}

def _safe_div(num, denom):
    """Division with protection against zero / NaN denom."""
    if denom is None or pd.isna(denom) or denom == 0:
        return np.nan
    return num / denom

def aggregate_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stage-level metrics into pipeline-level (cell-level) metrics.

    Grouping keys:
        pipeline, student_size, dataset_choice, cell_id, run_id

    For each group, we compute:
        - student_tokens / teacher_tokens
        - student_duration_s / teacher_duration_s
        - student_kwh / teacher_kwh / eval_kwh / other_kwh / total_kwh_all
        - GPU/CPU kWh for student stages
        - total / student / teacher energy in Joules
        - tokens_per_sec_student
        - energy_j_per_token_total
        - energy_j_per_token_student
    """
    group_cols = ["pipeline", "student_size", "dataset_choice", "cell_id", "run_id"]
    group_cols = [c for c in group_cols if c in df.columns]

    print("\n[DEBUG] Aggregating with group columns:", group_cols)
    grouped = df.groupby(group_cols, dropna=False)

    records = []

    for key, rows in grouped:
        # key can be a scalar or tuple depending on number of group_cols
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))

        # Split by stage_role
        student_rows = rows[rows["stage_role"].isin(STUDENT_ROLES)]
        teacher_rows = rows[rows["stage_role"].isin(TEACHER_ROLES)]
        eval_rows = rows[rows["stage_role"].isin(EVAL_ROLES)]
        other_rows = rows[
            ~rows["stage_role"].isin(STUDENT_ROLES | TEACHER_ROLES | EVAL_ROLES)
        ]

        # Tokens
        student_tokens = (
            student_rows["tokens_processed"].sum()
            if "tokens_processed" in student_rows.columns
            else np.nan
        )
        teacher_tokens = (
            teacher_rows["tokens_processed"].sum()
            if "tokens_processed" in teacher_rows.columns
            else np.nan
        )

        # Durations
        student_dur = (
            student_rows["duration_seconds"].sum()
            if "duration_seconds" in student_rows.columns
            else np.nan
        )
        teacher_dur = (
            teacher_rows["duration_seconds"].sum()
            if "duration_seconds" in teacher_rows.columns
            else np.nan
        )

        # Energy in kWh
        student_kwh = student_rows["energy_kwh"].sum() if "energy_kwh" in student_rows.columns else np.nan
        teacher_kwh = teacher_rows["energy_kwh"].sum() if "energy_kwh" in teacher_rows.columns else np.nan
        eval_kwh = eval_rows["energy_kwh"].sum() if "energy_kwh" in eval_rows.columns else np.nan
        other_kwh = other_rows["energy_kwh"].sum() if "energy_kwh" in other_rows.columns else np.nan
        total_kwh_all = rows["energy_kwh"].sum() if "energy_kwh" in rows.columns else np.nan

        # Energy in Joules (effective)
        student_j = student_rows["energy_joules_eff"].sum() if "energy_joules_eff" in student_rows.columns else np.nan
        teacher_j = teacher_rows["energy_joules_eff"].sum() if "energy_joules_eff" in teacher_rows.columns else np.nan
        eval_j = eval_rows["energy_joules_eff"].sum() if "energy_joules_eff" in eval_rows.columns else np.nan
        other_j = other_rows["energy_joules_eff"].sum() if "energy_joules_eff" in other_rows.columns else np.nan
        total_j_all = rows["energy_joules_eff"].sum() if "energy_joules_eff" in rows.columns else np.nan

        # GPU / CPU energy (student-only for now; you can expand to teacher/eval later)
        student_gpu_kwh = (
            student_rows["gpu_energy_kwh"].sum()
            if "gpu_energy_kwh" in student_rows.columns
            else np.nan
        )
        student_cpu_kwh = (
            student_rows["cpu_energy_kwh"].sum()
            if "cpu_energy_kwh" in student_rows.columns
            else np.nan
        )

        # Tokens per second (student)
        tokens_per_sec_student = _safe_div(student_tokens, student_dur)

        # J/token:
        #  - total: teacher + student + eval + other, divided by student tokens
        #  - student-only: student energy divided by student tokens
        energy_j_per_token_total = _safe_div(total_j_all, student_tokens)
        energy_j_per_token_student = _safe_div(student_j, student_tokens)

        record.update(
            dict(
                student_tokens=student_tokens,
                teacher_tokens=teacher_tokens,
                student_duration_s=student_dur,
                teacher_duration_s=teacher_dur,
                student_kwh=student_kwh,
                teacher_kwh=teacher_kwh,
                eval_kwh=eval_kwh,
                other_kwh=other_kwh,
                total_kwh_all=total_kwh_all,
                student_energy_joules=student_j,
                teacher_energy_joules=teacher_j,
                eval_energy_joules=eval_j,
                other_energy_joules=other_j,
                total_energy_joules_all=total_j_all,
                student_gpu_kwh=student_gpu_kwh,
                student_cpu_kwh=student_cpu_kwh,
                tokens_per_sec_student=tokens_per_sec_student,
                energy_j_per_token_total=energy_j_per_token_total,
                energy_j_per_token_student=energy_j_per_token_student,
            )
        )

        records.append(record)

    agg_df = pd.DataFrame.from_records(records)

    print(f"[INFO] Aggregated into {len(agg_df)} rows (one per {group_cols} group).")
    return agg_df

cell_metrics_df = aggregate_cell_metrics(stage_df_all)

print("\n=== cell_metrics_df basic summary ===")
print("Rows:", len(cell_metrics_df))
print("Distinct cells (cell_id):", cell_metrics_df["cell_id"].dropna().nunique() if "cell_id" in cell_metrics_df.columns else "N/A")
print("Distinct run_id:", cell_metrics_df["run_id"].nunique() if "run_id" in cell_metrics_df.columns else "N/A")

print("\n=== Sample of aggregated pipeline-level metrics ===")
cols_to_show = [
    "pipeline",
    "student_size",
    "dataset_choice",
    "cell_id",
    "run_id",
    "student_tokens",
    "student_duration_s",
    "total_kwh_all",
    "student_kwh",
    "teacher_kwh",
    "eval_kwh",
    "tokens_per_sec_student",
    "energy_j_per_token_total",
    "energy_j_per_token_student",
]
cols_to_show = [c for c in cols_to_show if c in cell_metrics_df.columns]
display(cell_metrics_df[cols_to_show].head(15))

# -------------------------------------------------------------------------
# Optional hook: evaluation metrics (to be added later)
# -------------------------------------------------------------------------
print(
    "\n[NOTE] Evaluation metrics (GSM8K, MMLU, AlpacaEval, etc.) "
    "will be merged into cell_metrics_df in a later step once we "
    "have a structured eval_metrics CSV."
)


# In[31]:


# ## Helpers and core 3×3-style grid (Step 3b)

print("\n\n=== Step 3b: Helpers and core 3×3-style grid ===")

def filter_cells(
    df: pd.DataFrame,
    pipeline: str | list[str] | None = None,
    student_size: str | list[str] | None = None,
    dataset_choice: str | list[str] | None = None,
    run_id: str | list[str] | None = None,
    cell_id: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience filter on the aggregated cell_metrics_df.

    All filters are optional; each can be a single value or a list of values.
    """
    out = df.copy()

    def _normalize_filter(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple, set)):
            return list(val)
        return [val]

    pipeline = _normalize_filter(pipeline)
    student_size = _normalize_filter(student_size)
    dataset_choice = _normalize_filter(dataset_choice)
    run_id = _normalize_filter(run_id)
    cell_id = _normalize_filter(cell_id)

    if pipeline is not None and "pipeline" in out.columns:
        out = out[out["pipeline"].isin(pipeline)]
    if student_size is not None and "student_size" in out.columns:
        out = out[out["student_size"].isin(student_size)]
    if dataset_choice is not None and "dataset_choice" in out.columns:
        out = out[out["dataset_choice"].isin(dataset_choice)]
    if run_id is not None and "run_id" in out.columns:
        out = out[out["run_id"].isin(run_id)]
    if cell_id is not None and "cell_id" in out.columns:
        out = out[out["cell_id"].isin(cell_id)]

    print(
        f"[filter_cells] -> {len(out)} rows "
        f"(from {len(df)} input rows)"
    )
    return out


def build_core_grid(
    df: pd.DataFrame,
    primary_dataset: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate run-level cell metrics into a single row per
    (pipeline, student_size, dataset_choice).

    This is the basis for the 3×3 grid table:
        pipelines × student sizes (× dataset)

    Metrics:
        - total_student_tokens
        - total_student_duration_s
        - total_kwh_all
        - student_kwh, teacher_kwh, eval_kwh, other_kwh
        - tokens_per_sec_student_agg
        - energy_j_per_token_total_agg
        - energy_j_per_token_student_agg
        - summed student/teacher/eval Joules
    """
    work_df = df.copy()

    # Optionally restrict to a primary dataset, e.g. "tulu"
    if primary_dataset is not None and "dataset_choice" in work_df.columns:
        before = len(work_df)
        work_df = work_df[work_df["dataset_choice"] == primary_dataset]
        if verbose:
            print(
                f"[build_core_grid] Filtered to dataset_choice == '{primary_dataset}': "
                f"{before} -> {len(work_df)} rows"
            )

    # Group by pipeline × student_size × dataset_choice
    group_cols = ["pipeline", "student_size"]
    if "dataset_choice" in work_df.columns:
        group_cols.append("dataset_choice")

    group_cols = [c for c in group_cols if c in work_df.columns]

    if verbose:
        print("[build_core_grid] Grouping by:", group_cols)

    grouped = work_df.groupby(group_cols, dropna=False)

    grid_records = []

    for key, rows in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))

        # Aggregate tokens and duration (summing across runs)
        student_tokens_total = rows["student_tokens"].sum()
        student_duration_total = rows["student_duration_s"].sum()

        # Aggregate kWh
        student_kwh_total = rows["student_kwh"].sum()
        teacher_kwh_total = rows["teacher_kwh"].sum()
        eval_kwh_total = rows["eval_kwh"].sum()
        other_kwh_total = rows["other_kwh"].sum()
        total_kwh_all = rows["total_kwh_all"].sum()

        # Aggregate Joules
        student_j_total = rows["student_energy_joules"].sum()
        teacher_j_total = rows["teacher_energy_joules"].sum()
        eval_j_total = rows["eval_energy_joules"].sum()
        other_j_total = rows["other_energy_joules"].sum()
        total_j_all = rows["total_energy_joules_all"].sum()

        # Aggregate GPU/CPU energy for student stages
        student_gpu_kwh_total = rows["student_gpu_kwh"].sum()
        student_cpu_kwh_total = rows["student_cpu_kwh"].sum()

        # Recompute tokens/sec and J/token at this aggregated level
        tokens_per_sec_student_agg = (
            student_tokens_total / student_duration_total
            if (pd.notna(student_duration_total) and student_duration_total > 0)
            else np.nan
        )

        energy_j_per_token_total_agg = (
            total_j_all / student_tokens_total
            if (pd.notna(student_tokens_total) and student_tokens_total > 0)
            else np.nan
        )

        energy_j_per_token_student_agg = (
            student_j_total / student_tokens_total
            if (pd.notna(student_tokens_total) and student_tokens_total > 0)
            else np.nan
        )

        rec.update(
            dict(
                runs_in_cell=len(rows),
                total_student_tokens=student_tokens_total,
                total_student_duration_s=student_duration_total,
                total_kwh_all=total_kwh_all,
                student_kwh=student_kwh_total,
                teacher_kwh=teacher_kwh_total,
                eval_kwh=eval_kwh_total,
                other_kwh=other_kwh_total,
                total_energy_joules_all=total_j_all,
                student_energy_joules=student_j_total,
                teacher_energy_joules=teacher_j_total,
                eval_energy_joules=eval_j_total,
                other_energy_joules=other_j_total,
                student_gpu_kwh=student_gpu_kwh_total,
                student_cpu_kwh=student_cpu_kwh_total,
                tokens_per_sec_student=tokens_per_sec_student_agg,
                energy_j_per_token_total=energy_j_per_token_total_agg,
                energy_j_per_token_student=energy_j_per_token_student_agg,
            )
        )

        grid_records.append(rec)

    grid_df = pd.DataFrame.from_records(grid_records)

    if verbose:
        print(
            f"[build_core_grid] Created grid_df with {len(grid_df)} rows "
            f"(from {len(work_df)} input rows)"
        )
        if "pipeline" in grid_df.columns:
            print("  Pipelines:", grid_df["pipeline"].dropna().unique())
        if "student_size" in grid_df.columns:
            print("  Student sizes:", grid_df["student_size"].dropna().unique())
        if "dataset_choice" in grid_df.columns:
            print("  Datasets:", grid_df["dataset_choice"].dropna().unique())

    return grid_df


# You can change this to your primary headline dataset, e.g. "tulu"
PRIMARY_DATASET_FOR_CORE_GRID = None  # e.g. "tulu"

core_grid_df = build_core_grid(
    cell_metrics_df,
    primary_dataset=PRIMARY_DATASET_FOR_CORE_GRID,
    verbose=True,
)

print("\n=== core_grid_df preview (for 3×3-style table) ===")
cols_to_show = [
    "pipeline",
    "student_size",
    "dataset_choice",
    "runs_in_cell",
    "total_student_tokens",
    "total_student_duration_s",
    "total_kwh_all",
    "student_kwh",
    "teacher_kwh",
    "tokens_per_sec_student",
    "energy_j_per_token_total",
    "energy_j_per_token_student",
]
cols_to_show = [c for c in cols_to_show if c in core_grid_df.columns]
display(core_grid_df[cols_to_show].head(20))


# In[ ]:


# Example
kd_7b_tulu_cells = filter_cells(
    cell_metrics_df,
    pipeline="kd",
    student_size="7b",
    dataset_choice="tulu",
)


# In[33]:


# ## Utility: summarize arbitrary stage names (by name pattern)

print("\n\n=== Utility: summarize selected stage_names ===")

def summarize_stages_by_name(
    df: pd.DataFrame,
    stage_name_patterns: list[str],
    use_contains: bool = True,
) -> pd.DataFrame:
    """
    Sum energy + tokens for stages whose `stage_name` matches
    any of the provided patterns.

    Args:
        df: stage-level DataFrame (e.g., stage_df_all or stage_df_clean)
        stage_name_patterns: list of patterns, e.g.
            ["logit_caching", "kd_32b_to_7b", "eval_gsm8k"]
        use_contains: if True, pattern is substring; if False, exact match.

    Returns:
        DataFrame grouped by stage_name with:
            - tokens_processed_sum
            - duration_seconds_sum
            - energy_kwh_sum
            - energy_joules_eff_sum
            - tokens_per_sec (recomputed)
            - energy_j_per_token (recomputed)
    """
    if "stage_name" not in df.columns:
        raise ValueError("summarize_stages_by_name: df must have a 'stage_name' column.")

    patterns = list(stage_name_patterns)
    mask = pd.Series(False, index=df.index)

    for p in patterns:
        if use_contains:
            mask |= df["stage_name"].str.contains(p, na=False)
        else:
            mask |= (df["stage_name"] == p)

    subset = df[mask].copy()
    print(
        f"[summarize_stages_by_name] Selected {len(subset)} rows "
        f"matching patterns: {patterns}"
    )

    if subset.empty:
        return pd.DataFrame()

    group = subset.groupby("stage_name", dropna=False)
    rows = []

    for name, g in group:
        tokens_sum = g["tokens_processed"].sum() if "tokens_processed" in g.columns else np.nan
        dur_sum = g["duration_seconds"].sum() if "duration_seconds" in g.columns else np.nan
        kwh_sum = g["energy_kwh"].sum() if "energy_kwh" in g.columns else np.nan
        j_sum = g["energy_joules_eff"].sum() if "energy_joules_eff" in g.columns else np.nan

        tokens_per_sec = (tokens_sum / dur_sum) if (pd.notna(dur_sum) and dur_sum > 0) else np.nan
        energy_j_per_token = (j_sum / tokens_sum) if (pd.notna(tokens_sum) and tokens_sum > 0) else np.nan

        rows.append(
            dict(
                stage_name=name,
                rows_included=len(g),
                tokens_processed_sum=tokens_sum,
                duration_seconds_sum=dur_sum,
                energy_kwh_sum=kwh_sum,
                energy_joules_eff_sum=j_sum,
                tokens_per_sec=tokens_per_sec,
                energy_j_per_token=energy_j_per_token,
            )
        )

    out = pd.DataFrame(rows).sort_values("stage_name").reset_index(drop=True)
    print("[summarize_stages_by_name] Summary:")
    display(out)
    return out



# In[ ]:


# Example usage (commented out):
# summarize_stages_by_name(stage_df_all, ["logit_caching", "kd_32b_to_7b", "eval"], use_contains=True)


# In[34]:


# ## Stage-wise breakdown & teacher vs student vs eval

import matplotlib.pyplot as plt

print("\n\n=== Stage-wise breakdown helpers ===")

def stage_breakdown_for_cell(
    stage_df: pd.DataFrame,
    pipeline: str | None = None,
    student_size: str | None = None,
    dataset_choice: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """
    Compute stage-wise energy breakdown (kWh and fraction) for a given
    (pipeline, student_size, dataset_choice, run_id) subset.

    Returns a small DataFrame indexed by stage_role + stage_name.
    """
    df = stage_df.copy()

    if pipeline is not None and "pipeline" in df.columns:
        df = df[df["pipeline"] == str(pipeline).lower()]
    if student_size is not None and "student_size" in df.columns:
        df = df[df["student_size"] == str(student_size)]
    if dataset_choice is not None and "dataset_choice" in df.columns:
        df = df[df["dataset_choice"] == str(dataset_choice).lower()]
    if run_id is not None and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]

    if df.empty:
        print("[stage_breakdown_for_cell] No rows match the specified filters.")
        return pd.DataFrame()

    group_cols = ["stage_role"]
    if "stage_name" in df.columns:
        group_cols.append("stage_name")

    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for (role, name, *rest), g in grouped:
        role = role
        name = name
        kwh = g["energy_kwh"].sum() if "energy_kwh" in g.columns else np.nan
        j = g["energy_joules_eff"].sum() if "energy_joules_eff" in g.columns else np.nan

        rows.append(
            dict(
                stage_role=role,
                stage_name=name,
                energy_kwh=kwh,
                energy_joules_eff=j,
            )
        )

    out = pd.DataFrame(rows)
    total_kwh = out["energy_kwh"].sum()
    out["fraction_of_total_kwh"] = (
        out["energy_kwh"] / total_kwh if (pd.notna(total_kwh) and total_kwh > 0) else np.nan
    )
    out = out.sort_values("energy_kwh", ascending=False).reset_index(drop=True)

    print(
        "[stage_breakdown_for_cell] Breakdown for "
        f"pipeline={pipeline}, student_size={student_size}, "
        f"dataset_choice={dataset_choice}, run_id={run_id}"
    )
    print(f"Total energy_kwh: {total_kwh:.4f}")
    display(out)
    return out


def plot_teacher_student_eval_bar(
    df: pd.DataFrame,
    title: str = "Teacher vs Student vs Eval Energy (kWh)",
    figsize=(8, 5),
):
    """
    Given a small DataFrame with columns:
        label, student_kwh, teacher_kwh, eval_kwh, other_kwh
    produce a stacked bar chart.
    """
    required = ["label", "student_kwh", "teacher_kwh", "eval_kwh", "other_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[plot_teacher_student_eval_bar] Missing columns: {missing}")
        return

    x = np.arange(len(df))
    width = 0.6

    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(df))
    for col, legend_label in [
        ("teacher_kwh", "Teacher"),
        ("student_kwh", "Student"),
        ("eval_kwh", "Eval"),
        ("other_kwh", "Other"),
    ]:
        vals = df[col].fillna(0.0).to_numpy()
        ax.bar(x, vals, width, bottom=bottom, label=legend_label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def build_teacher_student_eval_comparison_from_core(
    core_df: pd.DataFrame,
    selector: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a compact table for teacher vs student vs eval energy from core_grid_df.

    Optionally pass a pre-filtered selector DataFrame (subset of core_grid_df).
    """
    work = selector if selector is not None else core_df
    if work.empty:
        print("[build_teacher_student_eval_comparison_from_core] Empty input.")
        return pd.DataFrame()

    rows = []
    for _, row in work.iterrows():
        label_parts = []
        if "pipeline" in row:
            label_parts.append(str(row["pipeline"]))
        if "student_size" in row:
            label_parts.append(str(row["student_size"]))
        if "dataset_choice" in row and pd.notna(row["dataset_choice"]):
            label_parts.append(str(row["dataset_choice"]))
        label = " ".join(label_parts)

        rows.append(
            dict(
                label=label,
                student_kwh=row.get("student_kwh", np.nan),
                teacher_kwh=row.get("teacher_kwh", np.nan),
                eval_kwh=row.get("eval_kwh", np.nan),
                other_kwh=row.get("other_kwh", np.nan),
                total_kwh=row.get("total_kwh_all", np.nan),
            )
        )

    out = pd.DataFrame(rows)
    print("[build_teacher_student_eval_comparison_from_core] Table:")
    display(out)
    return out


def compute_gpu_cpu_share_for_core(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GPU vs CPU energy share for student stages in core_grid_df.
    """
    df = core_df.copy()
    if not {"student_gpu_kwh", "student_cpu_kwh"}.issubset(df.columns):
        print("[compute_gpu_cpu_share_for_core] Required columns missing.")
        return df

    total_student = df["student_gpu_kwh"].fillna(0.0) + df["student_cpu_kwh"].fillna(0.0)
    df["student_gpu_share"] = np.where(
        total_student > 0,
        df["student_gpu_kwh"].fillna(0.0) / total_student,
        np.nan,
    )
    df["student_cpu_share"] = np.where(
        total_student > 0,
        df["student_cpu_kwh"].fillna(0.0) / total_student,
        np.nan,
    )

    print("\n[compute_gpu_cpu_share_for_core] GPU/CPU share:")
    cols = [
        "pipeline",
        "student_size",
        "dataset_choice",
        "student_gpu_kwh",
        "student_cpu_kwh",
        "student_gpu_share",
        "student_cpu_share",
    ]
    cols = [c for c in cols if c in df.columns]
    display(df[cols].head(20))
    return df

# Example usage (commented out):
# comparison_table = build_teacher_student_eval_comparison_from_core(core_grid_df)
# plot_teacher_student_eval_bar(comparison_table)
# core_grid_with_gpu_cpu = compute_gpu_cpu_share_for_core(core_grid_df)


# In[35]:


# ## Pareto frontier utilities (energy vs quality)

print("\n\n=== Pareto frontier helpers ===")

def mark_pareto_frontier(
    df: pd.DataFrame,
    energy_col: str,
    quality_col: str,
) -> pd.DataFrame:
    """
    Add a boolean column 'pareto_optimal' to df indicating which rows are
    on the Pareto frontier: no other row has BOTH lower (or equal) energy
    AND higher (or equal) quality, with at least one strict inequality.

    Returns a copy of df with the new column.
    """
    if energy_col not in df.columns or quality_col not in df.columns:
        print(f"[mark_pareto_frontier] Missing columns {energy_col} or {quality_col}")
        return df.copy()

    work = df[[energy_col, quality_col]].to_numpy()
    n = work.shape[0]
    pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not pareto[i]:
            continue
        ei, qi = work[i]
        if pd.isna(ei) or pd.isna(qi):
            pareto[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            ej, qj = work[j]
            if pd.isna(ej) or pd.isna(qj):
                continue
            # Check if j dominates i: lower or equal energy and higher or equal quality,
            # with at least one strict inequality
            if (ej <= ei) and (qj >= qi) and ((ej < ei) or (qj > qi)):
                pareto[i] = False
                break

    df_out = df.copy()
    df_out["pareto_optimal"] = pareto
    print(
        f"[mark_pareto_frontier] Marked {pareto.sum()} Pareto-optimal points "
        f"out of {n}."
    )
    return df_out


def plot_energy_quality_pareto(
    df: pd.DataFrame,
    energy_col: str,
    quality_col: str,
    pipeline_col: str = "pipeline",
    student_size_col: str = "student_size",
    title: str | None = None,
    figsize=(7, 5),
):
    """
    Scatter plot of energy vs quality with Pareto frontier highlighting.

    Expects df to already have 'pareto_optimal' column (from mark_pareto_frontier).
    """
    required = [energy_col, quality_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[plot_energy_quality_pareto] Missing columns: {missing}")
        return

    if "pareto_optimal" not in df.columns:
        print("[plot_energy_quality_pareto] 'pareto_optimal' column missing; call mark_pareto_frontier first.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Non-Pareto points
    mask_pareto = df["pareto_optimal"].fillna(False)
    non_pareto = df[~mask_pareto]
    pareto = df[mask_pareto]

    # Encode pipeline as marker and student_size as color via simple mapping
    pipeline_vals = non_pareto[pipeline_col].dropna().unique() if pipeline_col in df.columns else []
    student_vals = non_pareto[student_size_col].dropna().unique() if student_size_col in df.columns else []

    marker_map = {p: m for p, m in zip(pipeline_vals, ["o", "s", "D", "^", "v", "P"])}
    color_map = {s: i for i, s in enumerate(student_vals)}

    def _scatter(subset, edgecolors=None, linewidths=0.5, alpha=0.8, zorder=2):
        for _, row in subset.iterrows():
            e = row[energy_col]
            q = row[quality_col]
            if pd.isna(e) or pd.isna(q):
                continue
            marker = marker_map.get(row.get(pipeline_col), "o")
            color_idx = color_map.get(row.get(student_size_col), 0)
            ax.scatter(
                e,
                q,
                marker=marker,
                s=40,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=alpha,
                zorder=zorder,
            )

    # Plot non-Pareto
    _scatter(non_pareto, edgecolors=None, linewidths=0.5, alpha=0.5, zorder=1)

    # Plot Pareto with edge highlight
    _scatter(pareto, edgecolors="black", linewidths=1.2, alpha=0.9, zorder=3)

    ax.set_xlabel(energy_col)
    ax.set_ylabel(quality_col)
    ax.set_title(title or f"Energy vs Quality ({energy_col} vs {quality_col})")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage (after you have eval metrics merged in):
# gsm_df = mark_pareto_frontier(core_grid_df, energy_col="total_kwh_all", quality_col="gsm8k_acc")
# plot_energy_quality_pareto(gsm_df, "total_kwh_all", "gsm8k_acc", title="GSM8K: Energy vs Accuracy")


# In[36]:


# ## Evaluation metrics merge + LaTeX / CSV exports

print("\n\n=== Eval metrics + export helpers ===")

def merge_eval_metrics(
    base_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    on: list[str] | str = "run_id",
    suffixes=("", "_eval"),
) -> pd.DataFrame:
    """
    Merge evaluation metrics into base_df (cell_metrics_df or core_grid_df).

    `eval_df` should contain columns like:
        run_id, gsm8k_acc, mmlu_acc, alpacaeval_winrate, ifeval_score, ...

    Args:
        base_df: DataFrame with at least the key columns.
        eval_df: DataFrame with metrics.
        on: column name or list of column names to merge on.
    """
    merged = base_df.merge(eval_df, how="left", on=on, suffixes=suffixes)
    print(
        f"[merge_eval_metrics] Merged eval metrics: base {len(base_df)} rows, "
        f"eval {len(eval_df)} rows -> merged {len(merged)} rows"
    )
    return merged


def export_table_to_latex_and_csv(
    df: pd.DataFrame,
    cols: list[str],
    latex_path: str,
    csv_path: str | None = None,
    float_format: str = "%.3f",
    index: bool = False,
):
    """
    Export a DataFrame slice to LaTeX and optionally CSV.

    Args:
        df: the full DataFrame.
        cols: columns to include in the table.
        latex_path: path to .tex file.
        csv_path: optional path to .csv file.
    """
    subset = df[cols].copy()
    print(f"[export_table_to_latex_and_csv] Exporting {len(subset)} rows to {latex_path}")
    subset.to_latex(
        latex_path,
        float_format=float_format,
        index=index,
        escape=True,
    )
    if csv_path is not None:
        print(f"[export_table_to_latex_and_csv] Exporting CSV to {csv_path}")
        subset.to_csv(csv_path, index=index)

    display(subset.head(20))


# Example usage for your 3×3 headline table (adjust cols as needed):
# headline_cols = [
#     "pipeline",
#     "student_size",
#     "dataset_choice",
#     "total_student_tokens",
#     "total_kwh_all",
#     "energy_j_per_token_total",
#     "tokens_per_sec_student",
#     "gsm8k_acc",
#     "mmlu_acc",
#     "alpacaeval_winrate",
# ]
# headline_cols = [c for c in headline_cols if c in core_grid_df.columns]
# export_table_to_latex_and_csv(
#     core_grid_df,
#     cols=headline_cols,
#     latex_path="tables/core_3x3_grid.tex",
#     csv_path="tables/core_3x3_grid.csv",
# )


# In[37]:


# ## Manual overrides (for correcting / adding results)

print("\n\n=== Manual overrides helper ===")

def apply_overrides(
    base_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Apply overrides to base_df using key_cols as the identifier.

    For each row in overrides_df:
        - find matching row(s) in base_df by key_cols
        - overwrite non-null values in overrides_df into base_df

    If an override row does not match any base row, it is appended.
    """
    base = base_df.copy()
    ov = overrides_df.copy()

    for col in key_cols:
        if col not in base.columns:
            raise ValueError(f"apply_overrides: key column '{col}' missing from base_df.")
        if col not in ov.columns:
            raise ValueError(f"apply_overrides: key column '{col}' missing from overrides_df.")

    base["_override_key"] = base[key_cols].astype(str).agg("||".join, axis=1)
    ov["_override_key"] = ov[key_cols].astype(str).agg("||".join, axis=1)

    override_keys = set(ov["_override_key"].unique())
    base_keys = set(base["_override_key"].unique())

    to_update = override_keys & base_keys
    to_add = override_keys - base_keys

    print(
        f"[apply_overrides] {len(to_update)} rows will be updated, "
        f"{len(to_add)} rows will be appended."
    )

    # Update existing rows
    for key in to_update:
        base_mask = base["_override_key"] == key
        ov_row = ov[ov["_override_key"] == key].iloc[0]

        for col in ov.columns:
            if col in ("_override_key",) + tuple(key_cols):
                continue
            val = ov_row[col]
            if pd.notna(val):
                base.loc[base_mask, col] = val

    # Append new rows
    new_rows = []
    for key in to_add:
        ov_row = ov[ov["_override_key"] == key].iloc[0].to_dict()
        # Ensure all columns exist in base
        for col in base.columns:
            ov_row.setdefault(col, np.nan)
        new_rows.append(ov_row)

    if new_rows:
        base = pd.concat([base, pd.DataFrame(new_rows)[base.columns]], ignore_index=True)

    # Cleanup
    base = base.drop(columns=["_override_key"], errors="ignore")

    print("[apply_overrides] Overrides applied. New shape:", base.shape)
    return base

# Example usage:
# overrides_df = pd.DataFrame([
#     {
#         "pipeline": "kd",
#         "student_size": "7b",
#         "dataset_choice": "tulu",
#         "total_kwh_all": 123.456,  # corrected value
#         "gsm8k_acc": 0.745,       # manually entered
#     },
# ])
# core_grid_corrected = apply_overrides(core_grid_df, overrides_df, key_cols=["pipeline", "student_size", "dataset_choice"])

