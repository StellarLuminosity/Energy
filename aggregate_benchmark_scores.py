import os
import json
import pandas as pd
from pathlib import Path

# Paths based on your setup
INPUT_DIR = "/scratch/klambert/model_log/olmo_benchmarks"
OUTPUT_DIR = "/home/klambert/projects/aip-craffel/klambert/Energy/benchmark_results"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "model_efficiency_summary.csv")

def aggregate_benchmarks():
    # Ensure the output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    all_model_data = []

    # Iterate through model folders in scratch
    for model_name in os.listdir(INPUT_DIR):
        model_path = os.path.join(INPUT_DIR, model_name)
        if not os.path.isdir(model_path):
            continue
            
        print(f"Processing: {model_name}...")
        
        metrics = []
        total_energy_kwh = 0.0
        
        # Walk through the model folder to find all relevant files
        for root, dirs, files in os.walk(model_path):
            for file in files:
                # 1. Extract Accuracy Metrics
                if file.startswith("task-") and file.endswith("-metrics.json"):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            data = json.load(f)
                            # Prioritize acc_norm for standard research reporting
                            score = data.get('acc_norm') or data.get('acc')
                            if score is not None:
                                metrics.append(score)
                    except Exception as e:
                        print(f"Error reading metric {file}: {e}")

                # 2. Extract Energy Data (CodeCarbon)
                if file == "emissions.csv":
                    try:
                        emissions_df = pd.read_csv(os.path.join(root, file))
                        # Summing energy_consumed in case the tracker ran in multiple stages
                        total_energy_kwh += emissions_df['energy_consumed'].sum()
                    except Exception as e:
                        print(f"Error reading emissions in {root}: {e}")

        # Aggregate results for this model
        if metrics:
            avg_acc = sum(metrics) / len(metrics)
            # Tagging models based on your naming convention
            is_distilled = "trained_adafactor" in model_name
            
            all_model_data.append({
                "Model_Name": model_name,
                "Type": "Distilled" if is_distilled else "Raw",
                "Mean_Accuracy": round(avg_acc, 4),
                "Total_Energy_kWh": round(total_energy_kwh, 6),
                "Tasks_Evaluated": len(metrics)
            })

    # Save results
    df = pd.DataFrame(all_model_data)
    df.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSuccess! Summary saved to: {SUMMARY_FILE}")
    return df

if __name__ == "__main__":
    summary_df = aggregate_benchmarks()
    # Display top performers by efficiency (Acc / Energy)
    summary_df['Efficiency_Score'] = summary_df['Mean_Accuracy'] / summary_df['Total_Energy_kWh']
    print(summary_df.sort_values(by="Efficiency_Score", ascending=False))