# experiment_c_layer_ablation.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score
import os
import sys

# Standardized seed and device
SEED = 1337
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"
MODELS_TO_TEST = ["FacebookAI/xlm-roberta-base", "google-bert/bert-base-german-cased"]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier
from experiments.gbert_baseline_run import TicketDataset, train_test_split

def execute_ablation_run():
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    train_set, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    ablation_results = []

    for m_id in MODELS_TO_TEST:
        # Configuration: Freeze Bottom (Embeddings + 6 Layers) vs Top (Last 4 Layers Only Trainable)
        for freeze_config in ["bottom_half", "top_only"]:
            print(f"\n--- Testing Ablation: {m_id} | Config: {freeze_config} ---")
            
            tokenizer = AutoTokenizer.from_pretrained(m_id)
            model = TicketClassifier(m_id).to(DEVICE)
            
            # Access the underlying transformer backbone
            if hasattr(model, 'bert'):
                backbone = model.bert
            elif hasattr(model, 'roberta'):
                backbone = model.roberta
            else:
                backbone = model.encoder # For cases using generic AutoModel

            # Implement Freezing Logic
            if freeze_config == "bottom_half":
                # Hypothesis: Syntax is in the bottom; freeze it to preserve native structure
                for param in backbone.embeddings.parameters():
                    param.requires_grad = False
                for layer in backbone.encoder.layer[:6]:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            elif freeze_config == "top_only":
                # Hypothesis: Winning tickets for radicalization are only in high-level semantics
                for param in backbone.parameters():
                    param.requires_grad = False
                # Re-enable the last 4 layers
                for layer in backbone.encoder.layer[-4:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Training Loop (Shortened for ablation test)
            train_loader = DataLoader(TicketDataset(train_set, tokenizer, 128), batch_size=24, shuffle=True)
            val_loader = DataLoader(TicketDataset(val_set, tokenizer, 128), batch_size=24)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
            criterion = {t: torch.nn.CrossEntropyLoss() for t in tasks}

            for epoch in range(2): # 2 Epochs to see gradient stability
                model.train()
                for b in train_loader:
                    optimizer.zero_grad()
                    ids, mask = b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE)
                    preds = model(ids, mask)
                    loss = sum([criterion[t](preds[t], b['targets'][t].to(DEVICE)) for t in tasks])
                    loss.backward()
                    optimizer.step()

            # Final Evaluation
            model.eval()
            task_f1s = {}
            for t in tasks:
                preds, labels = [], []
                with torch.no_grad():
                    for b in val_loader:
                        out = model(b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE))
                        preds.extend(torch.argmax(out[t], dim=1).cpu().numpy())
                        labels.extend(b['targets'][t].numpy())
                task_f1s[f"{t}_f1"] = f1_score(labels, preds, average='macro')
            
            task_f1s["model"] = m_id
            task_f1s["config"] = freeze_config
            ablation_results.append(task_f1s)

    # Save quantitative findings
    df = pd.DataFrame(ablation_results)
    df.to_csv("experiment_c_ablation_results.csv", index=False)
    print("\n✅ Experiment C Complete. Results saved to experiment_c_ablation_results.csv")

if __name__ == "__main__":
    execute_ablation_run()