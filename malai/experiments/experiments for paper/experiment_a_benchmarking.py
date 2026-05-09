import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import sys

# Standardized seed and device
SEED = 1337
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"
MODELS = {
    "DistilBERT-Multilingual": "distilbert-base-multilingual-cased",
    "XLM-RoBERTa": "FacebookAI/xlm-roberta-base",
    "GBERT": "google-bert/bert-base-german-cased"
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier
from experiments.gbert_baseline_run import TicketDataset, train_test_split

def run_benchmarking():
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    _, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    results = []
    for name, m_id in MODELS.items():
        print(f"Benchmarking Initialization: {name}")
        tokenizer = AutoTokenizer.from_pretrained(m_id)
        model = TicketClassifier(m_id).to(DEVICE)
        
        # Load weights from baseline runs
        m_slug = m_id.split('/')[-1]
        weight_path = f"best_ticket_model_{m_slug}.bin"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        v_loader = DataLoader(TicketDataset(val_set, tokenizer, 128), batch_size=24)
        
        metrics = {"Model": name}
        for t in tasks:
            preds, labels = [], []
            with torch.no_grad():
                for b in v_loader:
                    out = model(b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE))
                    preds.extend(torch.argmax(out[t], dim=1).cpu().numpy())
                    labels.extend(b['targets'][t].numpy())
            
            metrics[f"{t}_f1"] = f1_score(labels, preds, average='macro')
            metrics[f"{t}_prec"] = precision_score(labels, preds, average='macro', zero_division=0)
            metrics[f"{t}_rec"] = recall_score(labels, preds, average='macro', zero_division=0)
        
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv("experiment_a_initialization_results.csv", index=False)
    print("Experiment A Complete: Results saved to experiment_a_initialization_results.csv")

if __name__ == "__main__":
    run_benchmarking()