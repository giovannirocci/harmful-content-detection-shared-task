import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration
SEED = 1337
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"

# The Models for Agreement Analysis
MODELS = {
    "XLM-R": "FacebookAI/xlm-roberta-base",
    "German-BERT": "google-bert/bert-base-german-cased"
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier
from experiments.gbert_baseline_run import TicketDataset

def run_agreement_analysis():
    tasks = ['c2a', 'dbo', 'vio', 'def']
    
    # 1. Prepare Validation Data
    dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    _, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    all_model_preds = {t: {} for t in tasks}
    true_labels = {t: [] for t in tasks}

    # 2. Generate Predictions from each Expert
    for name, m_id in MODELS.items():
        print(f"Generating predictions for Agreement Analysis: {name}")
        tokenizer = AutoTokenizer.from_pretrained(m_id)
        model = TicketClassifier(m_id).to(DEVICE)
        
        m_slug = m_id.split('/')[-1]
        weight_path = f"best_ticket_model_{m_slug}.bin"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        v_loader = DataLoader(TicketDataset(val_set, tokenizer, 128), batch_size=24, shuffle=False)
        
        for t in tasks:
            preds = []
            with torch.no_grad():
                for b in v_loader:
                    out = model(b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE))
                    preds.extend(torch.argmax(out[t], dim=1).cpu().numpy())
                    if name == "XLM-R": # Labels are consistent, only collect once
                        true_labels[t].extend(b['targets'][t].numpy())
            all_model_preds[t][name] = np.array(preds)

    # 3. Analyze Agreement vs. Performance
    metrics = []
    plt.figure(figsize=(20, 5))

    for idx, t in enumerate(tasks):
        preds_a = all_model_preds[t]["XLM-R"]
        preds_b = all_model_preds[t]["German-BERT"]
        labels = np.array(true_labels[t])

        # Calculate masks
        agree_mask = (preds_a == preds_b)
        disagree_mask = ~agree_mask

        # Metrics when they agree
        f1_agree = f1_score(labels[agree_mask], preds_a[agree_mask], average='macro', zero_division=0)
        # Metrics when they disagree (using XLM-R as proxy)
        f1_disagree = f1_score(labels[disagree_mask], preds_a[disagree_mask], average='macro', zero_division=0)
        
        agreement_rate = np.mean(agree_mask)
        
        metrics.append({
            "Task": t,
            "Agreement_Rate": agreement_rate,
            "F1_on_Agreement": f1_agree,
            "F1_on_Disagreement": f1_disagree
        })

        # Visualization: Agreement Confusion Matrix
        conf_matrix = confusion_matrix(preds_a, preds_b)
        plt.subplot(1, 4, idx+1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"{t.upper()} Agreement Map\n(XLM-R vs BERT)")
        plt.xlabel("BERT Preds")
        plt.ylabel("XLM-R Preds")

    # 4. Save Outputs
    plt.tight_layout()
    plt.savefig("experiment_e_agreement_heatmap.png")
    pd.DataFrame(metrics).to_csv("experiment_e_agreement_metrics.csv", index=False)
    
    print("\n✅ Experiment E Complete. Data saved to experiment_e_agreement_metrics.csv and heatmap saved.")

if __name__ == "__main__":
    run_agreement_analysis()