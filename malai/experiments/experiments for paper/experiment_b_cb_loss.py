# experiment_b_cb_loss.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import os
import sys

# Standardized setup
SEED = 1337
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"
MODEL_ID = "google-bert/bert-base-german-cased" 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier
from experiments.gbert_baseline_run import TicketDataset, train_test_split

# Implementation of Cui et al. (2019): Class-Balanced Loss Based on Effective Number of Samples
def get_cb_weights(labels_count, beta=0.9999):
    """
    Calculates weights based on the inverse of the effective number of samples.
    """
    effective_num = 1.0 - np.power(beta, labels_count)
    weights = (1.0 - beta) / np.array(effective_num)
    # Normalize weights to ensure loss scale remains consistent
    weights = weights / np.sum(weights) * len(labels_count)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def execute_cb_loss_experiment():
    tasks = ['c2a', 'dbo', 'vio', 'def']
    
    # 1. Load Data
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    
    train_df, val_df = train_test_split(master, test_size=0.15, random_state=SEED)

    # 2. Precise Counting Logic for CB Weights
    # This ensures the index of 'counts' matches the integer label expected by the model
    label_maps = {
        'c2a': {'false': 0, 'true': 1},
        'dbo': {'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3},
        'vio': {'nothing': 0, 'propensity': 1, 'prospensity': 1, 'call2violence': 2, 'support': 3, 'glorification': 4, 'other': 5},
        'def': {'false': 0, 'true': 1}
    }

    task_counts = {}
    for t in tasks:
        # Step 1: Standardize using your specific format to avoid nan errors
        standardized = train_df[t].astype(str).replace('nan', 'nothing').str.lower()
        
        # Step 2: Map to the model's integer IDs
        mapped_ids = standardized.map(label_maps[t])
        
        # Step 3: Count and sort by ID (0, 1, 2...)
        # We reindex with the known values to ensure we don't miss a class that has 0 samples
        unique_ids = sorted(list(set(label_maps[t].values())))
        counts = mapped_ids.value_counts().reindex(unique_ids, fill_value=0).sort_index().values
        task_counts[t] = counts
        print(f"Task {t.upper()} class counts: {counts}")

    # 3. Setup Model and CB Weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = TicketClassifier(MODEL_ID).to(DEVICE)
    
    # Generate weights using the Effective Number logic
    cb_weights = {t: get_cb_weights(task_counts[t]) for t in tasks}
    
    train_loader = DataLoader(TicketDataset(train_df, tokenizer, 128), batch_size=24, shuffle=True)
    val_loader = DataLoader(TicketDataset(val_df, tokenizer, 128), batch_size=24)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    # Apply weights to CrossEntropyLoss to combat class imbalance
    criterion = {t: torch.nn.CrossEntropyLoss(weight=cb_weights[t]) for t in tasks}

    # 4. Training Loop
    print(f"\n🚀 Starting Experiment B (CB Loss) with beta=0.9999")
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids, mask = batch['input_ids'].to(DEVICE), batch['attn_mask'].to(DEVICE)
            preds = model(ids, mask)
            
            # Weighted multi-task loss summation
            batch_loss = sum([criterion[t](preds[t], batch['targets'][t].to(DEVICE)) for t in tasks])
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")

    # 5. Evaluation and Reporting
    model.eval()
    final_metrics = {}
    for t in tasks:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['input_ids'].to(DEVICE), batch['attn_mask'].to(DEVICE))
                all_preds.extend(torch.argmax(out[t], dim=1).cpu().numpy())
                all_labels.extend(batch['targets'][t].numpy())
        
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        final_metrics[t] = report['macro avg']['f1-score']
        print(f"\n[CB-LOSS RESULT: {t.upper()}]")
        print(classification_report(all_labels, all_preds, zero_division=0))

    # Save results for comparison plot
    pd.DataFrame([final_metrics], index=["CB-Loss-Beta-0.9999"]).to_csv("experiment_b_cb_loss_results.csv")
    print("\n✅ Experiment B Complete. Results saved to experiment_b_cb_loss_results.csv")

if __name__ == "__main__":
    execute_cb_loss_experiment()