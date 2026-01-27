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

# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd

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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


from pathlib import Path
import re
from typing import Iterable
import matplotlib as plt


# In[ ]:


# -------------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------------
def _norm_str(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()

def _norm_key(x: object) -> str:
    return _norm_str(x).lower()

def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _ci_mask(series: pd.Series, values: list[str]) -> pd.Series:
    vals = {_norm_key(v) for v in values}
    return series.astype(str).str.strip().str.lower().isin(vals)


# In[ ]:


# --------------------------
# Ensure required id columns exist
# --------------------------
def ensure_identifiers(stage_df: pd.DataFrame) -> pd.DataFrame:
    df = stage_df.copy()

    # Normalize some key cols if present
    for c in ["pipeline", "dataset_choice", "stage_name", "source"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower() == "nan", c] = np.nan
            if c in {"pipeline", "dataset_choice", "stage_name", "source"}:
                df[c] = df[c].str.lower()

    # run_id
    if "run_id" not in df.columns:
        if "experiment_name" in df.columns:
            df["run_id"] = df["experiment_name"].fillna(df.get("stage_name"))
        else:
            df["run_id"] = df.get("stage_name")

    # cell_id
    if "cell_id" not in df.columns:
        needed = ["pipeline", "student_size", "dataset_choice"]
        if all(c in df.columns for c in needed):
            miss = df[needed].isna().any(axis=1)
            df["cell_id"] = np.where(
                miss,
                np.nan,
                df["pipeline"].astype(str).str.lower()
                + "_"
                + df["student_size"].astype(str)
                + "_"
                + df["dataset_choice"].astype(str).str.lower()
            )
        else:
            df["cell_id"] = np.nan

    return df


def _safe_div(num, denom):
    if denom is None or pd.isna(denom) or denom == 0:
        return np.nan
    return num / denom


# In[ ]:


# --------------------------
# Stage -> run/cell aggregation
# --------------------------
STUDENT_ROLES = {"student_train", "train_summary"}
TEACHER_ROLES = {"teacher_generation", "teacher_processing", "data_preprocess"}
EVAL_ROLES = {"evaluation"}

def aggregate_cell_metrics(stage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stage-level rows into run-level/cell-level rows.
    Output is grouped by: pipeline, student_size, dataset_choice, cell_id, run_id
    """
    df = ensure_identifiers(stage_df)

    # stage_role must exist
    if "stage_role" not in df.columns:
        raise ValueError("aggregate_cell_metrics: stage_role missing. Run retag_stage_roles(...) first.")

    group_cols = [c for c in ["pipeline","student_size","dataset_choice","cell_id","run_id"] if c in df.columns]
    grouped = df.groupby(group_cols, dropna=False)

    recs = []
    for key, rows in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))

        student_rows = rows[rows["stage_role"].isin(STUDENT_ROLES)]
        teacher_rows = rows[rows["stage_role"].isin(TEACHER_ROLES)]
        eval_rows    = rows[rows["stage_role"].isin(EVAL_ROLES)]
        other_rows   = rows[~rows["stage_role"].isin(STUDENT_ROLES | TEACHER_ROLES | EVAL_ROLES)]

        # Tokens/duration
        student_tokens = student_rows["tokens_processed"].sum() if "tokens_processed" in rows.columns else np.nan
        student_dur    = student_rows["duration_seconds"].sum() if "duration_seconds" in rows.columns else np.nan

        # Energy kWh
        student_kwh = student_rows["energy_kwh"].sum() if "energy_kwh" in rows.columns else np.nan
        teacher_kwh = teacher_rows["energy_kwh"].sum() if "energy_kwh" in rows.columns else np.nan
        eval_kwh    = eval_rows["energy_kwh"].sum()    if "energy_kwh" in rows.columns else np.nan
        other_kwh   = other_rows["energy_kwh"].sum()   if "energy_kwh" in rows.columns else np.nan
        total_kwh_all = rows["energy_kwh"].sum()       if "energy_kwh" in rows.columns else np.nan

        # Energy Joules (effective)
        student_j = student_rows["energy_joules_eff"].sum() if "energy_joules_eff" in rows.columns else np.nan
        teacher_j = teacher_rows["energy_joules_eff"].sum() if "energy_joules_eff" in rows.columns else np.nan
        eval_j    = eval_rows["energy_joules_eff"].sum()    if "energy_joules_eff" in rows.columns else np.nan
        other_j   = other_rows["energy_joules_eff"].sum()   if "energy_joules_eff" in rows.columns else np.nan
        total_j_all = rows["energy_joules_eff"].sum()       if "energy_joules_eff" in rows.columns else np.nan

        # GPU/CPU (student-only by default)
        student_gpu_kwh = student_rows["gpu_energy_kwh"].sum() if "gpu_energy_kwh" in rows.columns else np.nan
        student_cpu_kwh = student_rows["cpu_energy_kwh"].sum() if "cpu_energy_kwh" in rows.columns else np.nan

        tokens_per_sec_student = _safe_div(student_tokens, student_dur)
        energy_j_per_token_total = _safe_div(total_j_all, student_tokens)
        energy_j_per_token_student = _safe_div(student_j, student_tokens)

        rec.update({
            "student_tokens": student_tokens,
            "student_duration_s": student_dur,
            "student_kwh": student_kwh,
            "teacher_kwh": teacher_kwh,
            "eval_kwh": eval_kwh,
            "other_kwh": other_kwh,
            "total_kwh_all": total_kwh_all,
            "student_energy_joules": student_j,
            "teacher_energy_joules": teacher_j,
            "eval_energy_joules": eval_j,
            "other_energy_joules": other_j,
            "total_energy_joules_all": total_j_all,
            "student_gpu_kwh": student_gpu_kwh,
            "student_cpu_kwh": student_cpu_kwh,
            "tokens_per_sec_student": tokens_per_sec_student,
            "energy_j_per_token_total": energy_j_per_token_total,
            "energy_j_per_token_student": energy_j_per_token_student,
        })
        recs.append(rec)

    return pd.DataFrame.from_records(recs)



# In[ ]:


# -------------------------------------------------------------------------
# Patch A: case-insensitive filter_cells (fixes 7B vs 7b issues)
# -------------------------------------------------------------------------
def filter_cells(
    df: pd.DataFrame,
    pipeline: str | list[str] | None = None,
    student_size: str | list[str] | None = None,
    dataset_choice: str | list[str] | None = None,
    run_id: str | list[str] | None = None,
    cell_id: str | list[str] | None = None,
    *,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    pipeline = _as_list(pipeline)
    student_size = _as_list(student_size)
    dataset_choice = _as_list(dataset_choice)
    run_id = _as_list(run_id)
    cell_id = _as_list(cell_id)

    def _apply(col: str, vals: list[str] | None):
        nonlocal out
        if vals is None or col not in out.columns:
            return
        if case_insensitive:
            out = out[_ci_mask(out[col], vals)]
        else:
            out = out[out[col].isin(vals)]

    _apply("pipeline", pipeline)
    _apply("student_size", student_size)
    _apply("dataset_choice", dataset_choice)
    _apply("run_id", run_id)
    _apply("cell_id", cell_id)

    print(f"[filter_cells] -> {len(out)} rows (from {len(df)} input rows)")
    return out


# In[ ]:


# -------------------------------------------------------------------------
# Patch B: stronger stage_role inference (teacher-side KD + caching patterns)
# -------------------------------------------------------------------------
STAGE_ROLE_OVERRIDES: dict[str, str] = {
    # Example manual overrides:
    # "logit_caching": "teacher_processing",
}

STAGE_ROLE_RULES: list[tuple[str, list[str]]] = [
    ("teacher_generation", [r"synthetic.*generation", r"teacher.*generation"]),
    ("teacher_processing", [r"logit", r"cache", r"teacher.*forward", r"teacher.*infer", r"distill.*teacher"]),
    ("data_preprocess", [r"preprocess", r"tokeni[sz]e", r"build_.*dataset", r"dataset_.*prep"]),
    ("evaluation", [r"\beval\b", r"gsm8k", r"mmlu", r"alpaca", r"ifeval", r"mt[-_]?bench"]),
]

def infer_stage_role_v2(row: pd.Series) -> str:
    name = _norm_key(row.get("stage_name", ""))
    pipeline = _norm_key(row.get("pipeline", ""))
    source = _norm_key(row.get("source", ""))

    if name in STAGE_ROLE_OVERRIDES:
        return STAGE_ROLE_OVERRIDES[name]

    for role, pats in STAGE_ROLE_RULES:
        for pat in pats:
            if re.search(pat, name):
                return role

    if pipeline in {"sft", "kd", "dpo", "true_sft", "synthetic_sft"}:
        if source == "summary":
            return "train_summary"
        return "student_train"

    if source == "summary":
        return "summary_only"
    return "other"

def retag_stage_roles(stage_df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    df = stage_df if inplace else stage_df.copy()
    df["stage_role"] = df.apply(infer_stage_role_v2, axis=1)
    return df


# In[ ]:


# --------------------------
# run/cell -> core grid (9-row style)
# --------------------------
def build_core_grid(
    cell_metrics_df: pd.DataFrame,
    primary_dataset: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collapse run-level rows into one row per (pipeline, student_size, dataset_choice).
    This is what you use for the 3×3 table.
    """
    df = cell_metrics_df.copy()

    if primary_dataset is not None and "dataset_choice" in df.columns:
        before = len(df)
        df = df[df["dataset_choice"].astype(str).str.lower() == str(primary_dataset).lower()]
        if verbose:
            print(f"[build_core_grid] dataset_choice={primary_dataset}: {before} -> {len(df)} rows")

    group_cols = [c for c in ["pipeline","student_size","dataset_choice"] if c in df.columns]
    grouped = df.groupby(group_cols, dropna=False)

    out = []
    for key, rows in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = dict(zip(group_cols, key))

        tokens = rows["student_tokens"].sum()
        dur_s  = rows["student_duration_s"].sum()

        # Sum energies across runs
        rec["runs_in_cell"] = len(rows)
        rec["total_student_tokens"] = tokens
        rec["total_student_duration_s"] = dur_s

        rec["student_kwh"] = rows["student_kwh"].sum()
        rec["teacher_kwh"] = rows["teacher_kwh"].sum()
        rec["eval_kwh"]    = rows["eval_kwh"].sum()
        rec["other_kwh"]   = rows["other_kwh"].sum()
        rec["total_kwh_all"] = rows["total_kwh_all"].sum()

        rec["student_energy_joules"] = rows["student_energy_joules"].sum()
        rec["teacher_energy_joules"] = rows["teacher_energy_joules"].sum()
        rec["eval_energy_joules"]    = rows["eval_energy_joules"].sum()
        rec["other_energy_joules"]   = rows["other_energy_joules"].sum()
        rec["total_energy_joules_all"] = rows["total_energy_joules_all"].sum()

        rec["student_gpu_kwh"] = rows["student_gpu_kwh"].sum()
        rec["student_cpu_kwh"] = rows["student_cpu_kwh"].sum()

        # Recompute derived metrics at the aggregated level
        tps = _safe_div(tokens, dur_s)
        jpt_total = _safe_div(rec["total_energy_joules_all"], tokens)
        jpt_student = _safe_div(rec["student_energy_joules"], tokens)

        # Provide both naming styles (plain + _agg)
        rec["tokens_per_sec_student"] = tps
        rec["tokens_per_sec_student_agg"] = tps

        rec["energy_j_per_token_total"] = jpt_total
        rec["energy_j_per_token_total_agg"] = jpt_total

        rec["energy_j_per_token_student"] = jpt_student
        rec["energy_j_per_token_student_agg"] = jpt_student

        out.append(rec)

    grid = pd.DataFrame(out)
    if verbose:
        print(f"[build_core_grid] rows={len(grid)} groups={group_cols}")
    return grid


# In[ ]:


# -------------------------------------------------------------------------
# Patch C: optional allocation of shared teacher stages to multiple student cells (model amortization)
# -------------------------------------------------------------------------
def allocate_shared_teacher_costs(
    stage_df: pd.DataFrame,
    allocation_rules: list[dict],
    *,
    energy_cols: Iterable[str] = (
        "energy_kwh",
        "energy_joules_eff",
        "gpu_energy_kwh",
        "cpu_energy_kwh",
        "duration_seconds",
        "tokens_processed",
    ),
) -> pd.DataFrame:
    if not allocation_rules:
        return stage_df

    base = stage_df.copy()
    if "stage_role" not in base.columns:
        raise ValueError("allocate_shared_teacher_costs expects 'stage_role' (run retag_stage_roles first).")

    allocated_rows = []
    for rule in allocation_rules:
        pat = rule.get("match_stage_name_regex")
        if not pat:
            continue

        role_match = rule.get("match_stage_role")
        target_pipes = rule.get("target_pipelines", [])
        target_sizes = rule.get("target_student_sizes", [])
        target_ds = rule.get("target_dataset_choice", None)
        mode = rule.get("mode", "amortize")
        weight_override = rule.get("weight", None)

        mask = base["stage_name"].astype(str).str.lower().str.contains(pat.lower(), regex=True, na=False)
        if role_match is not None:
            mask = mask & (base["stage_role"] == role_match)

        matched = base[mask].copy()
        if matched.empty:
            print(f"[allocate_shared_teacher_costs] No matches for rule pattern: {pat!r}")
            continue

        n_targets = max(1, len(target_pipes) * len(target_sizes))
        if weight_override is not None:
            weight = float(weight_override)
        else:
            weight = 1.0 if mode == "full" else 1.0 / float(n_targets)

        for _, r in matched.iterrows():
            for p in target_pipes:
                for s in target_sizes:
                    rr = r.copy()
                    rr["pipeline"] = p
                    rr["student_size"] = s
                    if target_ds is not None:
                        rr["dataset_choice"] = target_ds
                    rr["allocation_weight"] = weight
                    rr["allocated_from_stage_name"] = r.get("stage_name", None)
                    rr["is_allocated"] = True

                    for c in energy_cols:
                        if c in rr.index and pd.notna(rr[c]):
                            rr[c] = rr[c] * weight

                    allocated_rows.append(rr)

        print(
            f"[allocate_shared_teacher_costs] Matched {len(matched)} stage rows for {pat!r}; "
            f"created {len(matched) * n_targets} allocated rows (mode={mode}, weight={weight:g})."
        )

    if not allocated_rows:
        return base

    alloc_df = pd.DataFrame(allocated_rows)
    out = pd.concat([base, alloc_df], ignore_index=True)
    out["is_allocated"] = out.get("is_allocated", False).fillna(False)
    return out

def rebuild_aggregates(stage_df: pd.DataFrame, *, teacher_alloc_rules: list[dict] | None = None):
    """
    stage_df -> retag roles -> optional teacher allocation -> cell_metrics_df -> core_grid_df
    Returns: (stage_df2, cell_metrics_df2, core_grid_df2)
    """
    df2 = retag_stage_roles(stage_df, inplace=False)
    if teacher_alloc_rules:
        df2 = allocate_shared_teacher_costs(df2, teacher_alloc_rules)

    cell_df2 = aggregate_cell_metrics(df2)
    core_df2 = build_core_grid(cell_df2, primary_dataset=None, verbose=False)
    return df2, cell_df2, core_df2


# In[ ]:


# -------------------------------------------------------------------------
# Patch D: nicer totals + fractions (train-only vs all)
# -------------------------------------------------------------------------
def add_total_variants(core_df: pd.DataFrame) -> pd.DataFrame:
    out = core_df.copy()
    for col in ["student_kwh", "teacher_kwh", "eval_kwh", "other_kwh", "total_kwh_all"]:
        if col not in out.columns:
            out[col] = np.nan

    out["total_kwh_train_only"] = out["student_kwh"] + out["teacher_kwh"]
    out["teacher_frac_train_only"] = out["teacher_kwh"] / out["total_kwh_train_only"].replace({0: np.nan})
    out["eval_frac_of_total"] = out["eval_kwh"] / out["total_kwh_all"].replace({0: np.nan})
    return out


# In[ ]:


# -------------------------------------------------------------------------
# Patch E: improved Pareto plot with actual student-size colors + legend
# -------------------------------------------------------------------------
PIPELINE_DISPLAY = {"sft": "SFT", "kd": "KD", "true_sft": "Synthetic SFT", "synthetic_sft": "Synthetic SFT"}
PIPELINE_MARKERS = {"sft": "o", "kd": "s", "true_sft": "D", "synthetic_sft": "D"}

def plot_energy_quality_pareto_v2(
    df: pd.DataFrame,
    energy_col: str,
    quality_col: str,
    *,
    title: str | None = None,
    pipeline_col: str = "pipeline",
    student_size_col: str = "student_size",
    label_col: str | None = None,
    savepath: str | Path | None = None,
    figsize=(7, 5),
    annotate_pareto: bool = False,
):
    required = [energy_col, quality_col, "pareto_optimal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[plot_energy_quality_pareto_v2] Missing columns: {missing}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    student_vals = sorted(df[student_size_col].dropna().unique(), key=lambda x: _norm_key(x))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0","C1","C2","C3","C4","C5"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(student_vals)}

    for (p, s), sub in df.groupby([pipeline_col, student_size_col], dropna=False):
        if sub.empty:
            continue
        marker = PIPELINE_MARKERS.get(_norm_key(p), "o")
        color = color_map.get(s, None)

        non_p = sub[~sub["pareto_optimal"].fillna(False)]
        par = sub[sub["pareto_optimal"].fillna(False)]

        ax.scatter(non_p[energy_col], non_p[quality_col], marker=marker, c=color, s=40, alpha=0.45, edgecolors="none")
        ax.scatter(
            par[energy_col], par[quality_col],
            marker=marker, c=color, s=80, alpha=0.95,
            edgecolors="black", linewidths=1.2,
            label=f"{PIPELINE_DISPLAY.get(_norm_key(p), p)} • {s}",
        )

        if annotate_pareto and label_col and label_col in par.columns:
            for _, r in par.iterrows():
                ax.annotate(str(r[label_col]), (r[energy_col], r[quality_col]), xytext=(4,4),
                            textcoords="offset points", fontsize=8)

    ax.set_xlabel(energy_col)
    ax.set_ylabel(quality_col)
    ax.set_title(title or f"Energy vs Quality ({energy_col} vs {quality_col})")
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8, frameon=True)

    plt.tight_layout()
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        print(f"[plot_energy_quality_pareto_v2] Saved: {savepath}")
    plt.show()


# In[ ]:


# --------------------------
# Pareto frontier
# --------------------------
def mark_pareto_frontier(df: pd.DataFrame, energy_col: str, quality_col: str) -> pd.DataFrame:
    """
    Minimization in energy_col, maximization in quality_col.
    Marks pareto_optimal True/False.
    """
    out = df.copy()
    out["pareto_optimal"] = False

    sub = out[[energy_col, quality_col]].copy()
    sub = sub.dropna()
    idxs = sub.index.to_list()

    e = sub[energy_col].to_numpy()
    q = sub[quality_col].to_numpy()

    pareto = np.ones(len(sub), dtype=bool)
    for i in range(len(sub)):
        if not pareto[i]:
            continue
        dominates = (e <= e[i]) & (q >= q[i]) & ((e < e[i]) | (q > q[i]))
        if np.any(dominates):
            pareto[i] = False

    out.loc[idxs, "pareto_optimal"] = pareto
    return out


# In[ ]:


# --------------------------
# Eval merge helper
# --------------------------
def merge_eval_metrics(base_df: pd.DataFrame, eval_df: pd.DataFrame, on="run_id") -> pd.DataFrame:
    if eval_df is None or len(eval_df) == 0:
        return base_df
    return base_df.merge(eval_df, on=on, how="left")


# In[ ]:


# -------------------------------------------------------------------------
# Patch F: headline 9-row grid builder + LaTeX/CSV export wrapper
# -------------------------------------------------------------------------
def make_headline_grid(
    core_df: pd.DataFrame,
    *,
    dataset_choice: str | None = "tulu",
    pipelines: list[str] | None = None,
    student_sizes: list[str] | None = None,
    include_eval: bool = True,
) -> pd.DataFrame:
    df = core_df.copy()
    if dataset_choice is not None and "dataset_choice" in df.columns:
        df = df[df["dataset_choice"].astype(str).str.lower() == dataset_choice.lower()]

    pipelines = pipelines or ["sft", "kd", "true_sft"]
    student_sizes = student_sizes or ["1B", "7B", "13B"]
    df = filter_cells(df, pipeline=pipelines, student_size=student_sizes, case_insensitive=True)

    df = add_total_variants(df)
    df["headline_total_kwh"] = df["total_kwh_all"] if include_eval else df["total_kwh_train_only"]

    want = [
        "pipeline","student_size","dataset_choice",
        "total_student_tokens",
        "headline_total_kwh",
        "total_kwh_train_only",
        "teacher_kwh","teacher_frac_train_only",
        "tokens_per_sec_student_agg",
        "energy_j_per_token_total_agg","energy_j_per_token_student_agg",
        # optional eval cols if merged
        "gsm8k_acc","mmlu_acc","alpacaeval_winrate","ifeval_score",
    ]
    want = [c for c in want if c in df.columns]
    df = df[want].copy()

    if "pipeline" in df.columns:
        df["pipeline"] = df["pipeline"].map(lambda p: PIPELINE_DISPLAY.get(_norm_key(p), p))
    return df

def export_headline_tables(
    core_df: pd.DataFrame,
    *,
    out_dir: str | Path = "tables",
    dataset_choice: str | None = "tulu",
    include_eval: bool = True,
    basename: str = "energy_grid",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tbl = make_headline_grid(core_df, dataset_choice=dataset_choice, include_eval=include_eval)

    # save CSV
    csv_path = out_dir / f"{basename}.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"[export_headline_tables] Wrote CSV: {csv_path}")

    # save LaTeX (uses your existing helper if present)
    if "export_table_to_latex_and_csv" in globals():
        export_table_to_latex_and_csv(
            tbl,
            out_path_prefix=str(out_dir / basename),
            latex_caption=f"Energy/throughput summary ({dataset_choice})",
            latex_label=f"tab:{basename}",
            float_format="%.4g",
        )
    else:
        tex_path = out_dir / f"{basename}.tex"
        tex_path.write_text(tbl.to_latex(index=False, float_format="%.4g"), encoding="utf-8")
        print(f"[export_headline_tables] Wrote LaTeX: {tex_path}")

    return tbl


# In[ ]:


# -------------------------------------------------------------------------
# Optional convenience loaders
# -------------------------------------------------------------------------
def load_eval_metrics_csv(path: str | Path = "eval_metrics.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[load_eval_metrics_csv] Not found: {path} (returning empty df)")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[load_eval_metrics_csv] Loaded {len(df)} rows from {path}")
    return df

def load_overrides_csv(path: str | Path = "overrides.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[load_overrides_csv] Not found: {path} (returning empty df)")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[load_overrides_csv] Loaded {len(df)} rows from {path}")
    return df


# ---
# ## Example Commands

# In[ ]:


# Example: amortize synthetic tulu generation across 3 student sizes for Synthetic SFT
teacher_alloc_rules = [
    dict(
        match_stage_name_regex=r"synthetic_tulu_generation",
        match_stage_role="teacher_generation",
        target_pipelines=["true_sft"],
        target_student_sizes=["1B", "7B", "13B"],
        target_dataset_choice="tulu",
        mode="amortize",  # change to "full" for worst-case accounting
    )
]

stage_df_v2, cell_metrics_df_v2, core_grid_df_v2 = rebuild_aggregates(
    stage_df_all,
    teacher_alloc_rules=teacher_alloc_rules,
)

core_grid_df_v2 = add_total_variants(core_grid_df_v2)
display(core_grid_df_v2.head(20))


# In[ ]:


# Merge evaluation metrics (once I have them)
eval_df = load_eval_metrics_csv("eval_metrics.csv")

# Option A: merge on run_id (best if eval is per-run)
cell_with_eval = merge_eval_metrics(cell_metrics_df_v2, eval_df, on="run_id")

# Option B: merge on pipeline/student_size/dataset_choice (if eval is per cell)
# cell_with_eval = merge_eval_metrics(core_grid_df_v2, eval_df, on=["pipeline","student_size","dataset_choice"])


# In[ ]:


# Export the headline 9-row table (Latex + CSV)
headline_tbl = export_headline_tables(
    core_grid_df_v2,
    out_dir="tables",
    dataset_choice="tulu",   # swap to "math", "codeforces", etc.
    include_eval=True,
    basename="energy_grid_tulu",
)

display(headline_tbl)


# In[ ]:


# Pareto Frontier
# Mark pareto points (run-level or cell-level)
pareto_df = mark_pareto_frontier(
    cell_with_eval,  # or core_grid_df_v2 if it has quality metrics
    energy_col="total_kwh_all",
    quality_col="gsm8k_acc",
)

plot_energy_quality_pareto_v2(
    pareto_df,
    energy_col="total_kwh_all",
    quality_col="gsm8k_acc",
    title="GSM8K: Energy vs Quality (Pareto highlighted)",
    label_col="run_id",  # optional
    savepath="figures/pareto_gsm8k_kwh.pdf",
)

