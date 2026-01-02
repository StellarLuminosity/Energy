import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import os

from distill_bench.core.config_loader import load_config

# Load config
config = load_config('configs/experiments/kd_7b_to_1b.yaml')


def main():
    """Generate synthetic dataset using teacher model."""
    dataset = datasets.load_from_disk(config.dataset_path)
    dataloader = DataLoader(dataset["train"].select(range(100)), batch_size=2)

    ds = {"input_ids": [], "attention_mask": [], "labels": []}

    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.teacher_device,
    )
    teacher.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
    teacher.eval()

    print("\n=== GENERATING TEACHER LOGITS ===")
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(config.teacher_device)
        attention_mask = batch["attention_mask"].to(config.teacher_device)

        # Append the original inputs directly
        ds["input_ids"].extend(input_ids.cpu().unbind())
        ds["attention_mask"].extend(attention_mask.cpu().unbind())

        with torch.no_grad():
            generation_output = teacher.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.5,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_sequences = generation_output.sequences
            gen_only = generated_sequences[:, input_ids.shape[1] :]

            # pad = torch.full((gen_only.size(0), input_ids.size(1)), fill_value=-100, dtype=gen_only.dtype, device=gen_only.device)
            # labels = torch.cat([pad, gen_only], dim=1)

            # Create labels tensor with correct padding
            # The labels should be the same length as the input_ids + gen_only
            # We pad the original input_ids part with -100
            max_len = generated_sequences.shape[1]
            labels = torch.full((input_ids.size(0), max_len), fill_value=-100, dtype=torch.long, device=gen_only.device)
            
            # Fill the generated part with the actual tokens
            labels[:, input_ids.size(1):] = gen_only

            ds["labels"].extend(labels.cpu().unbind())

            del generation_output, generated_sequences, gen_only, labels, input_ids, attention_mask
            torch.cuda.empty_cache()

    dset = datasets.Dataset.from_dict(ds)
    dset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.makedirs(config.synthetic_dataset_path, exist_ok=True)
    dset.save_to_disk(config.synthetic_dataset_path)


if __name__ == "__main__":
    main()
