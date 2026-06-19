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

# MONKEY PATCH: Intercept torch.load to force weights_only tracking or bypass the v2.6 gating rule if safetensors exist
original_torch_load = torch.load
def secured_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs and torch.__version__ < "2.6":
        kwargs['weights_only'] = True
    try:
        return original_torch_load(*args, **kwargs)
    except ValueError as e:
        if "upgrade torch to at least v2.6" in str(e):
            # Force fallbacks to safetensors parsing architectures implicitly
            raise ImportError("Fallback to safetensors requested via mock gating")
        raise e
torch.load = secured_torch_load

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# --- 2. CONFIG ---
SEED = 1337 
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS = ["cardiffnlp/twitter-xlm-roberta-base"] 
DATA_PATH = "../EDA/using-trial-data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3 
LR = 3e-5   
BATCH = 24  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

# --- 3. THEORETICAL LOSS ---
def get_cb_weights(labels_count, beta=0.9999):
    effective_num = 1.0 - np.power(beta, labels_count)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(labels_count)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# --- 4. DATASET ABSTRACTION (ALIGNED KEYS) ---
class HaystackDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.mapping = {
            'c2a': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1},
            'dbo': {'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3},
            'vio': {'nothing': 0, 'propensity': 1, 'prospensity': 1, 'call2violence': 2, 'support': 3, 'glorification': 4, 'other': 5},
            'def': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1}
        }

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data.iloc[i]
        tokens = self.tokenizer(str(item['description']), max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        targets = {}
        for task in ['c2a', 'dbo', 'vio', 'def']:
            val = str(item.get(task, 'nothing')).lower()
            targets[task] = torch.tensor(self.mapping[task].get(val, 0), dtype=torch.long)
            
        return {
            'input_ids': tokens['input_ids'].flatten(), 
            'attn_mask': tokens['attention_mask'].flatten(), 
            'targets': targets
        }

# --- 5. EXECUTION ENGINE ---
def run_experiment(m_name):
    print(f"\n🔍 INVESTIGATING INITIALIZATION: {m_name}")
    print("Integrating Native Mixed-Precision Engine & Localized Datasets")
    
    dfs = {t: pd.read_csv(f"{DATA_PATH}{t}_trial.csv", sep=';') for t in ['c2a', 'dbo', 'vio', 'def']}
    master = dfs['c2a']
    for t in ['dbo', 'vio', 'def']: 
        master = master.merge(dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    
    train, val = train_test_split(master, test_size=0.15, random_state=SEED)

    weights = {
        'c2a': get_cb_weights([15000, 1000]), 
        'dbo': get_cb_weights([14000, 1500, 400, 100]),
        'vio': get_cb_weights([15500, 100, 100, 100, 100, 100]),
        'def': get_cb_weights([15800, 200])
    }

    tokenizer = AutoTokenizer.from_pretrained(m_name)
    
    # Pre-fetching clean safetensors structure explicitly
    print("Pre-fetching clean safetensors structure...")
    AutoModel.from_pretrained(m_name, use_safetensors=True)
    
    # MONKEY PATCH: Globally swap out the from_pretrained layer of AutoModel before building TicketClassifier
    original_from_pretrained = AutoModel.from_pretrained
    AutoModel.from_pretrained = lambda pretrained_model_name_or_path, *args, **kwargs: original_from_pretrained(
        pretrained_model_name_or_path, *args, **{**kwargs, "use_safetensors": True}
    )
    
    try:
        model = TicketClassifier(m_name).to(DEVICE)
    finally:
        # Restore original functionality to maintain framework integrity downstream
        AutoModel.from_pretrained = original_from_pretrained
    
    loader = DataLoader(HaystackDataset(train, tokenizer), batch_size=BATCH, shuffle=True)
    v_loader = DataLoader(HaystackDataset(val, tokenizer), batch_size=BATCH)

    opt = AdamW(model.parameters(), lr=LR)
    crit = {t: torch.nn.CrossEntropyLoss(weight=weights[t]) for t in ['c2a', 'dbo', 'vio', 'def']}

    scaler = torch.cuda.amp.GradScaler()
    best_mean_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for b in tqdm(loader, desc=f"Epoch {epoch+1}"):
            opt.zero_grad()
            ids, mask = b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                preds = model(ids, mask)
                loss = sum([crit[t](preds[t], b['targets'][t].to(DEVICE)) for t in ['c2a', 'dbo', 'vio', 'def']])
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        all_p, all_l = {t: [] for t in ['c2a', 'dbo', 'vio', 'def']}, {t: [] for t in ['c2a', 'dbo', 'vio', 'def']}
        with torch.no_grad():
            for b in v_loader:
                ids, mask = b['input_ids'].to(DEVICE), b['attn_mask'].to(DEVICE)
                outputs = model(ids, mask)
                for t in ['c2a', 'dbo', 'vio', 'def']:
                    all_p[t].extend(torch.argmax(outputs[t], dim=1).cpu().numpy())
                    all_l[t].extend(b['targets'][t].numpy())
        
        current_f1s = [f1_score(all_l[t], all_p[t], average='macro') for t in ['c2a', 'dbo', 'vio', 'def']]
        mean_f1 = np.mean(current_f1s)
        print(f"Epoch {epoch+1} | Mean F1: {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            model_slug = m_name.split('/')[-1]
            torch.save(model.state_dict(), f"best_ticket_model_{model_slug}.bin")
            print(f"✔️ New best weights saved for {model_slug}")

    print(f"\nFinal Baseline Results for {m_name}:")
    for t in ['c2a', 'dbo', 'vio', 'def']:
        print(f"\nTask: {t.upper()}")
        print(classification_report(all_l[t], all_p[t], zero_division=0))

if __name__ == "__main__":
    for m in MODELS: 
        run_experiment(m)