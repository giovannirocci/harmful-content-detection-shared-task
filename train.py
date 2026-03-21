"""
Fine-tuning an encoder model for harmful content classification (GermEval 2026).

Tasks:
  - c2a : Call to Action                  (binary:     FALSE / TRUE)
  - dbo : Democratic Basic Order attack   (4-class:    nothing / criticism / agitation / subversive)
  - def : Defamatory Offences             (binary:     FALSE / TRUE)
  - vio : Violence Detection              (6-class:    nothing / propensity / call2violence / support / glorification / other)
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "c2a": {
        "file": "GermEval2026/data/c2a/c2a_trial.csv",
        "text_col": "description",
        "label_col": "c2a",
        "labels": ["FALSE", "TRUE"],
    },
    "dbo": {
        "file": "GermEval2026/data/dbo/dbo_trial.csv",
        "text_col": "description",
        "label_col": "dbo",
        "labels": ["nothing", "criticism", "agitation", "subversive"],
    },
    "def": {
        "file": "GermEval2026/data/def/def_trial.csv",
        "text_col": "description",
        "label_col": "def",
        "labels": ["FALSE", "TRUE"],
    },
    "vio": {
        "file": "GermEval2026/data/vio/vio_trial.csv",
        "text_col": "description",
        "label_col": "vio",
        "labels": ["nothing", "propensity", "call2violence", "support", "glorification", "other"],
    },
}

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            **{k: v[idx] for k, v in self.encodings.items()},
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(task: str) -> tuple[list[str], list[int], list[str]]:
    """Load CSV, drop rows with missing labels, and encode labels as integers."""
    cfg = TASK_CONFIG[task]
    df = pd.read_csv(cfg["file"], sep=";", quotechar='"')
    df = df.dropna(subset=[cfg["text_col"], cfg["label_col"]])
    df = df[df[cfg["label_col"]].isin(cfg["labels"])]

    label2id = {lbl: i for i, lbl in enumerate(cfg["labels"])}
    texts = df[cfg["text_col"]].tolist()
    labels = df[cfg["label_col"]].map(label2id).tolist()
    return texts, labels, cfg["labels"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(label_names: list[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"macro_f1": macro_f1}
    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    task: str,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = "outputs",
    max_length: int = 128,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.15,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n=== Task: {task} | Model: {model_name} | Device: {device} ===")

    # --- Data ---
    texts, labels, label_names = load_data(task)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=seed, stratify=labels
    )
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Labels: {label_names}")

    # --- Tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label={i: lbl for i, lbl in enumerate(label_names)},
        label2id={lbl: i for i, lbl in enumerate(label_names)},
    )

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

    # --- Training args ---
    task_output_dir = os.path.join(output_dir, task)
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=os.path.join(task_output_dir, "logs"),
        logging_steps=50,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # --- Train ---
    trainer.train()

    # --- Evaluate ---
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, target_names=label_names, zero_division=0))

    # --- Save ---
    trainer.save_model(os.path.join(task_output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(task_output_dir, "best"))
    print(f"Model saved to {task_output_dir}/best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune encoder for GermEval 2026 tasks")
    parser.add_argument("--task", choices=list(TASK_CONFIG.keys()), required=True,
                        help="Classification task to train on")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model name or local path")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        task=args.task,
        model_name=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )
