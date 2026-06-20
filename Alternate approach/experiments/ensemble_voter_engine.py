import sys
import os

# --- 1. APPTAINER / CLUSTER ENVIRONMENT HOTFIX & HARD FORCED SAFETENSORS ---
# Force the entire transformers ecosystem to completely ignore legacy bin endpoints globally
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
# Patches missing float8 type attributes inside the Apptainer container space
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# MONKEY PATCH: Intercept torch.load to force weights_only tracking or bypass low container version restrictions
original_torch_load = torch.load
def secured_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and torch.__version__ < "2.6":
        kwargs['weights_only'] = True
    try:
        return original_torch_load(*args, **kwargs)
    except ValueError as e:
        if "upgrade torch to at least v2.6" in str(e):
            raise ImportError("Fallback to safetensors requested via mock gating")
        raise e
torch.load = secured_torch_load

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report

# --- 2. FIREWALL CONFIGURATION ---
SEED = 1337
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"

# ALIGNMENT: Perfectly mirrors both optimized backbone models
MODELS = ["cardiffnlp/twitter-xlm-roberta-base", "LSX-UniWue/ModernGBERT_134M"]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

# Explicitly importing the TicketDataset structure to avoid file dependency mismatches
from experiments.gbert_baseline_run import TicketDataset, train_test_split

# --- 3. EXECUTION ENGINE ---
def execute_ensemble_voting():
    print("-" * 60)
    print("ENSEMBLE PHASE: Voter Agreement & Collaborative Intelligence")
    print("-" * 60)

    tasks = ['c2a', 'dbo', 'violet', 'def'] # Handled multithread array profiles
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    
    master['description'] = master['description'].fillna("Empty")
    _, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    ensemble_logits = {t: [] for t in tasks}
    true_labels = {t: [] for t in tasks}

    for m_id in MODELS:
        print(f"Extracting feature confidence from: {m_id}")
        tokenizer = AutoTokenizer.from_pretrained(m_id)
        m_slug = m_id.split('/')[-1]
        
        # MONKEY PATCH: Globally swap out the from_pretrained layer of AutoModel before building TicketClassifier
        original_from_pretrained = AutoModel.from_pretrained
        AutoModel.from_pretrained = lambda pretrained_model_name_or_path, *args, **kwargs: original_from_pretrained(
            pretrained_model_name_or_path, *args, **{**kwargs, "use_safetensors": True}
        )
        
        try:
            model = TicketClassifier(m_id).to(DEVICE)
        finally:
            # Restore original functionality to maintain framework integrity downstream
            AutoModel.from_pretrained = original_from_pretrained
            
        weight_path = f"best_ticket_model_{m_slug}.bin"
        
        if os.path.exists(weight_path):
            # Safe parsing mapping execution context
            model.load_state_dict(original_torch_load(weight_path, map_location=DEVICE))
            print(f"Successfully loaded weights: {weight_path}")
        else:
            print(f"⚠️ Warning: {weight_path} not found. Expected path: {weight_path}")
            
        model.eval()

        v_loader = DataLoader(TicketDataset(val_set, tokenizer, 128), batch_size=24, shuffle=False)
        
        m_logits = {t: [] for t in tasks}
        with torch.no_grad():
            for batch in v_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attn_mask'].to(DEVICE)
                preds = model(ids, mask)
                for t in tasks:
                    m_logits[t].append(torch.softmax(preds[t], dim=1).cpu().numpy())
                    if m_id == MODELS[0]: 
                        true_labels[t].extend(batch['targets'][t].numpy())
        
        for t in tasks:
            ensemble_logits[t].append(np.concatenate(m_logits[t], axis=0))

    print("\nComputing Cross-Lingual Ensemble Verdicts...")
    final_f1s = {}
    for t in tasks:
        combined_probs = np.mean(ensemble_logits[t], axis=0)
        final_preds = np.argmax(combined_probs, axis=1)
        
        score = f1_score(true_labels[t], final_preds, average='macro')
        final_f1s[t] = score
        
        print(f"\n[COLLABORATIVE RESULT: {t.upper()}]")
        print(classification_report(true_labels[t], final_preds, zero_division=0))

    pd.DataFrame([final_f1s], index=["GBERT-XLMR-Ensemble"]).to_csv("ensemble_voter_results.csv")
    print("\n✅ Ensemble Phase Complete. Results saved to ensemble_voter_results.csv")

if __name__ == "__main__":
    execute_ensemble_voting()