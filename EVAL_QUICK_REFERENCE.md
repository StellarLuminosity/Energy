# Quick Reference: Evaluating Checkpoints

## Basic Usage

```bash
# Show help
./run_eval.sh

# Checkpoint directory
./run_eval.sh --model_path outputs/checkpoints/checkpoint_epoch0_step5000

# Final model file  
./run_eval.sh --model_path outputs/final_model/model.pt

# HuggingFace model
./run_eval.sh --model_name allenai/OLMo-2-0425-1B-SFT
```

## Quick Examples

### Evaluate latest checkpoint
```bash
LATEST=$(ls -t outputs/checkpoints/ | head -1)
./run_eval.sh --model_path outputs/checkpoints/$LATEST
```

### Evaluate all checkpoints
```bash
for ckpt in outputs/checkpoints/checkpoint_*; do
    ./run_eval.sh --model_path $ckpt
done
```

### Compare baseline vs trained
```bash
./run_eval.sh --model_name allenai/OLMo-2-0425-1B-SFT
./run_eval.sh --model_path outputs/final_model/model.pt
```

## Supported Formats

✅ **Distributed Checkpoint** (directory) - `checkpoint_epoch0_step5000/`
✅ **Single .pt File** - `final_model/model.pt`  
✅ **HuggingFace Model** - `allenai/OLMo-2-0425-1B-SFT`

## Direct Python (Alternative)

```bash
python src_simple/simple_eval.py --model_path outputs/checkpoints/checkpoint_epoch0_step5000
python src_simple/simple_eval.py --model_path outputs/final_model/model.pt
python src_simple/simple_eval.py --model_name allenai/OLMo-2-0425-1B-SFT
```

## Expected Output

```
======================================================================
MODEL EVALUATION
======================================================================
Loading test dataset...
Detected distributed checkpoint format
Loading distributed checkpoint from: outputs/checkpoints/checkpoint_epoch0_step5000
✓ Loaded checkpoint - Epoch: 0, Step: 5000
Checkpoint info: Epoch 0, Step 5000, Loss 2.3456

Evaluating model...
Evaluating: 100%|████████████████| 125/125 [00:15<00:00,  8.12it/s]

Model: outputs/checkpoints/checkpoint_epoch0_step5000
Dataset: allenai/tulu-v2-sft-mixture
Cross-Entropy Loss: 2.3456
Perplexity: 10.44
======================================================================
```

## OLMo benchmark script (AlpacaEval 2, GSM8K, MMLU, IFEval, MT-Bench-101)

Dependencies (install as needed):
- `lm-eval` for GSM8K/MMLU/IFEval (`pip install lm-eval`)
- `alpaca_eval` for AlpacaEval 2 (`pip install alpaca_eval`)
- `mt-bench-101` repo for MT-Bench-101 (https://github.com/mtbench101/mt-bench-101)

Usage:
```bash
# List available tasks
python distill_bench/data/olmo_benchmark.py --config <cfg.yaml> --tasks list --run-dir /tmp/bench --dry-run

# Run lm-eval tasks with a small sample limit
python distill_bench/data/olmo_benchmark.py --config <cfg.yaml> \
  --tasks gsm8k,mmlu,ifeval --max-samples 2 --run-dir /tmp/bench_lm

# Run AlpacaEval 2
python distill_bench/data/olmo_benchmark.py --config <cfg.yaml> \
  --tasks alpaca_eval --max-samples 5 --run-dir /tmp/bench_alpaca

# Run MT-Bench-101 subset
python distill_bench/data/olmo_benchmark.py --config <cfg.yaml> \
  --tasks mt_bench_101 --max-samples 4 --run-dir /tmp/bench_mt
```

Outputs:
- Per-task artifacts in the run dir (e.g., `lm_eval_<task>.json`, `alpaca_eval/alpaca_eval_results.json`, `mt_bench_101/scores.json`).
- Aggregated run summary at `<run_dir>/benchmark_summary.json`.
```
