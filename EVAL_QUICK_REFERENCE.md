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
