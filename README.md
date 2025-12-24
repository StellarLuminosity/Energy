# Distilling Large Language Models into Small LM Ensembles
This project investigates whether ensembles of small language models (LMs) can match the performance of large LMs through knowledge distillation. By sequentially training and aggregating multiple student models, we aim to create lightweight, highly parallelizable alternatives to monolithic large models, ideal for decentralized inference.

---

## Motivation

Running large LMs (e.g., 100B+ parameters) requires expensive, high-bandwidth multi-GPU systems. In contrast, small models can run on commodity hardware, but are less powerful. This project explores whether a large LM's capabilities can be **distilled into an ensemble** of smaller models using a carefully designed hybrid loss and KL divergence training loop.

--- 

## Project Structure

```
slm_ensembles/
├── main.py               # Orchestrates multi-round distillation training
├── train.py              # Implements DistillationTrainer with hybrid loss
├── ensemble.py           # ModelEnsemble class that averages logits across models
├── utils.py              # Logging, evaluation, and helper functions
├── config.example        # Configurable hyperparameters and paths
├── preprocess_dataset.py # Dataset formatting, tokenization, and filtering
├── train.sh              # SLURM script for launching jobs
├── SLM_ensembles.ipynb   # Visualization and analysis of logs
```

---

## Quickstart

### Requirements

* Python 3.10+
* PyTorch + CUDA
* [Transformers](https://github.com/huggingface/transformers)
* [TRL](https://github.com/huggingface/trl)
* Datasets: Hugging Face's `allenai/tulu-3-sft-mixture` (loaded as part of script)

Install dependencies (in virtual env):

```bash
pip install -r requirements.txt
```

Preprocess Dataset:

```bash
python preprocess_dataset.py
```

This applies chat template to conversation samples, tokenizes and masks tokens for assistant-only supervision, filters out truncated examples and saves the dataset to disk.

## Configuration
All paths, hyperparameters, and job-specific settings are defined in `config.py`:

* `teacher_model_name` / `student_model_name`
* `alpha` (KL vs. NTP blend)
* `steps_per_round`
* `total_rounds`
* Logging behavior and output paths

Before each run, ensure:

* Rename `config.py.example` to `config.py`
* Update the logging directories to match your system
* Set a unique `custom_run_name`, `description`, and if doing several runs, group them by the same `id_string`
* Update `train.sh` job name
* Clean old logs if needed

### 🏁 Start Training
```bash
python main.py
```

Or via SLURM:
```bash
sbatch train.sh
```

Real-time output to terminal:
```bash
srun -c 4 --gres=gpu:2 --partition a40 --mem=10GB --pty --time=16:00:00 bash
./train.sh
```

This launches iterative rounds of distillation. Each round:

* Trains a new student model
* Adds it to the ensemble
* Evaluates student, teacher, and ensemble on the test set
* Logs all metrics to CSV

You can change the number of rounds, learning rate, alpha, etc., in `config.py`.

---

## Evaluation

### Logging & Metrics

All logs are stored in `/csv_logs/YYYY-MM-DD/custom_filename.csv`, including:

* Hybrid loss
* KL divergence
* Next-token loss
* Eval loss and perplexity (for teacher, student, and ensemble)

### Visualization

Use `SLM_ensembles.py` to plot training/eval curves. Ensure to change the directory names in the script as well. Change variables such as `side_by_side` and `multiple` to control the number of files loaded as well as the logging. 

