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


# ## Create Dataframe
