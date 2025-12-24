#!/bin/bash
# Evaluate a model on the test dataset

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--model_path PATH | --model_name NAME]"
    echo ""
    echo "Examples:"
    echo "  # Checkpoint directory:"
    echo "  $0 --model_path /scratch/klambert/model_log/singular/checkpoints/checkpoint_epoch0_step5000"
    echo ""
    echo "  # Final model file:"
    echo "  $0 --model_path outputs/final_model/model.pt"
    echo ""
    echo "  # HuggingFace model:"
    echo "  $0 --model_name allenai/OLMo-2-0425-1B-SFT"
    exit 1
fi

# Change to project directory
cd /home/klambert/projects/aip-craffel/klambert/SLMensembles || exit 1

# Run evaluation
python src_simple/simple_eval.py "$@"

