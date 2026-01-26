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














