# Energy Analysis Interface

This guide is for the **interface helpers** added at the bottom of `interpret_results_interface_v2.py`.

## Typical workflow

### 1) Load one or more `stage_metrics.csv` files

```python
stage_df_raw = load_stage_metrics_csvs([
    "runs_a/stage_metrics.csv",
    "runs_b/stage_metrics.csv",
])
stage_df_all, stage_df_clean = prepare_stage_df(stage_df_raw)
```

### 2) Make a conservative analysis DF (recommended)

This helps avoid **double-counting** from `summary` / `snapshot` rows.

```python
stage_df = make_analysis_df(stage_df_all)  # defaults: drop summaries + snapshots + tokens==0
```

If you rely on summaries as fallback:

```python
stage_df = make_analysis_df(stage_df_all, exclude_sources=set())
```

### 3) (Optional) Override stage roles

If a stage is misclassified:

```python
stage_df = apply_stage_role_overrides(
    stage_df,
    overrides={"logit_caching": "teacher_processing"},
    match="contains",
)
```

### 4) Define custom “pipeline points” by stage names

Each entry creates **one run-level point** (one dot in plots / one row before core grid aggregation).

```python
cell_specs = {
  "sft_32b→1b_tulu": {
    "pipeline": "sft",
    "student_size": "1B",
    "dataset_choice": "tulu",
    "stage_names": [
      "sft_32b_to_1b_tulu_nosft",
      "sft_32b_to_1b_benchmark_eval",
    ],
  },

  "kd_32b→1b_tulu": {
    "pipeline": "kd",
    "student_size": "1B",
    "dataset_choice": "tulu",
    "stage_name_contains": [
      "kd_olmo2_32b_to_1b",
      "logit_caching",
      "benchmark_eval",
    ],
  },
}
```

Build the run-level table:

```python
cell_metrics_df = build_custom_cell_metrics(stage_df, cell_specs)
```

### 5) Build the 3×3 core grid (pipeline × student size)

```python
core_grid_df = build_core_grid(cell_metrics_df, primary_dataset="tulu")
```

Or in one call:

```python
cell_metrics_df, core_grid_df = build_core_grid_from_specs(stage_df, cell_specs, primary_dataset="tulu")
```

### 6) Merge evaluation metrics, plot, export LaTeX

Evaluation metrics can come from a separate CSV/JSON you maintain.

```python
eval_df = pd.DataFrame([
  {"run_id": "sft_32b→1b_tulu", "gsm8k_acc": 0.42, "mmlu_acc": 0.28},
  {"run_id": "kd_32b→1b_tulu",  "gsm8k_acc": 0.40, "mmlu_acc": 0.27},
])

cell_metrics_with_eval = merge_eval_metrics(cell_metrics_df, eval_df, on="run_id")

plot_energy_quality_pareto(
    cell_metrics_with_eval,
    energy_col="total_kwh_all",
    quality_col="gsm8k_acc",
    outpath="fig_energy_vs_gsm8k.pdf",
)

export_table_to_latex_and_csv(
    core_grid_df,
    latex_path="tables/core_grid.tex",
    csv_path="tables/core_grid.csv",
    float_format="%.3g",
)
```

## Notes / gotchas

- **Summaries and snapshots** often re-report totals; `make_analysis_df(...)` drops them by default.
- **Teacher-side stages**: ensure your stage names contain `logit`, `logprob`, `cache_logits`, etc., or override roles.
- This file is notebook-derived; if you want *import-safe* behavior, put plots/analysis execution behind
  `if __name__ == "__main__":` in your own scripts.
