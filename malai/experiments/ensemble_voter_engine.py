import torch
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report

# Ensure reproducibility for the collaborative voting phase
SEED = 1337
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"

# The committee consists of a Multilingual Generalist and a Monolingual Specialist
MODELS = ["FacebookAI/xlm-roberta-base", "google-bert/bert-base-german-cased"]

# Path adjustment for modular access to models and baseline utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier
# Updated import to match the new gbert_baseline_run filename
from experiments.gbert_baseline_run import TicketDataset, train_test_split

def execute_ensemble_voting():
    """
    Implements a soft-voting ensemble to evaluate agreement dynamics 
    between multilingual (XLM-R) and monolingual (GBERT) backbones.
    """
    print("-" * 60)
    print("ENSEMBLE PHASE: Voter Agreement & Collaborative Intelligence")
    print("-" * 60)

    # 1. Prepare Validation Data
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    
    master['description'] = master['description'].fillna("")
    _, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    # 2. Probability aggregation from the expert bank
    ensemble_logits = {t: [] for t in tasks}
    true_labels = {t: [] for t in tasks}

    for m_id in MODELS:
        print(f"Extracting feature confidence from: {m_id}")
        tokenizer = AutoTokenizer.from_pretrained(m_id)
        m_slug = m_id.split('/')[-1]
        
        # Initialize the specific classifier and load pre-trained baseline weights
        model = TicketClassifier(m_id).to(DEVICE)
        weight_path = f"best_ticket_model_{m_slug}.bin"
        
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print(f"Successfully loaded weights: {weight_path}")
        else:
            print(f"⚠️ Warning: {weight_path} not found. Using initialized weights.")
            
        model.eval()

        v_loader = DataLoader(TicketDataset(val_set, tokenizer, 128), batch_size=24, shuffle=False)
        
        m_logits = {t: [] for t in tasks}
        with torch.no_grad():
            for batch in v_loader:
                preds = model(batch['input_ids'].to(DEVICE), batch['attn_mask'].to(DEVICE))
                for t in tasks:
                    # Apply Softmax to convert raw logits into probability distributions
                    m_logits[t].append(torch.softmax(preds[t], dim=1).cpu().numpy())
                    if m_id == MODELS[0]: # Collect true labels only once
                        true_labels[t].extend(batch['targets'][t].numpy())
        
        for t in tasks:
            ensemble_logits[t].append(np.concatenate(m_logits[t], axis=0))

    # 3. Collaborative Verdict (Soft Voting Mechanism)
    print("\nComputing Cross-Lingual Ensemble Verdicts...")
    final_f1s = {}
    for t in tasks:
        # Soft voting: Averaging probabilities allows the most 'certain' model to influence the outcome
        combined_probs = np.mean(ensemble_logits[t], axis=0)
        final_preds = np.argmax(combined_probs, axis=1)
        
        score = f1_score(true_labels[t], final_preds, average='macro')
        final_f1s[t] = score
        
        print(f"\n[COLLABORATIVE RESULT: {t.upper()}]")
        print(classification_report(true_labels[t], final_preds, zero_division=0))

    # 4. Results Export
    pd.DataFrame([final_f1s], index=["GBERT-XLMR-Ensemble"]).to_csv("ensemble_voter_results.csv")
    print("\n✅ Ensemble Phase Complete. Results saved to ensemble_voter_results.csv")

if __name__ == "__main__":
    execute_ensemble_voting()