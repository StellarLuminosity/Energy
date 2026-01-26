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














