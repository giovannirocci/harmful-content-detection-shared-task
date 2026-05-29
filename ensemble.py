import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import os
import argparse

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from train import load_data, TextClassificationDataset, TASK_CONFIG


def main(model_paths, 
         val_split=0.2,
         max_length=256, 
         seed=42,
         output_dir="models/ensemble_results"):
    
    task_name = model_paths[0].split("/")[2]
    print(f"Loading models for task: {task_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    names = [path.split("/")[1] for path in model_paths]
    models = {name: AutoModelForSequenceClassification.from_pretrained(path) for name, path in zip(names, model_paths)}
    tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in zip(names, model_paths)}

    texts, labels, label_names = load_data(task_name)
    label_names = [str(name) for name in label_names]
    _, val_texts, _, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=seed, stratify=labels
    )
    print(f"Val: {len(val_texts)} | Labels: {label_names}")

    all_preds = []
    for name, model in models.items():
        print(f"Tokenizing validation data for model: {name}")
        tokenizer = tokenizers[name]
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
        dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        model.to(device)
        model.eval()

        model_preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                model_preds.extend(probs.tolist())
        all_preds.append(model_preds)

    # Average probabilities across models
    avg_probs = np.mean(all_preds, axis=0)
    pred_labels = np.argmax(avg_probs, axis=1)

    print("Ensemble Classification Report:")
    print(classification_report(val_labels, pred_labels, target_names=label_names, zero_division=0))

    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    with open(os.path.join(task_output_dir, "ensemble_results.txt"), "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Models: {', '.join(names)}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Validation Size: {len(val_texts)}\n")
        f.write(f"Classification Report:\n{classification_report(val_labels, pred_labels, target_names=label_names, zero_division=0)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Inference for Text Classification")
    parser.add_argument("--model_paths", "-mp", nargs="+", required=True, help="Paths to the trained models for ensembling")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="models/ensemble_results", help="Directory to save ensemble results")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(
        model_paths=args.model_paths,
        val_split=args.val_split,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=args.output_dir
    )
    