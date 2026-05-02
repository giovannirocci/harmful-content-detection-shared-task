import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import os
import sys

# --- 1. CONFIG ---
SEED = 1337 
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS = ["FacebookAI/xlm-roberta-base"] 
DATA_PATH = "../EDA/using-trial-data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3 
LR = 3e-5   
BATCH = 24  

# Ensure modular access to local models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

# --- 2. THEORETICAL LOSS: Class-Balanced (CB) Logic ---
def get_cb_weights(labels_count, beta=0.9999):
    """
    Implements the Effective Number of Samples logic (Cui et al., 2019).
    Calculates weights to mitigate the impact of high-imbalance classes.
    """
    effective_num = 1.0 - np.power(beta, labels_count)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(labels_count)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# --- 3. DATASET ---
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
            
        return {'ids': tokens['input_ids'].squeeze(), 'mask': tokens['attention_mask'].squeeze(), 'targets': targets}

# --- 4. EXECUTION ENGINE ---
def run_experiment(m_name):
    print(f"\n🔍 INVESTIGATING INITIALIZATION: {m_name}")
    
    # Data Aggregation
    dfs = {t: pd.read_csv(f"{DATA_PATH}{t}_trial.csv", sep=';') for t in ['c2a', 'dbo', 'vio', 'def']}
    master = dfs['c2a']
    for t in ['dbo', 'vio', 'def']: 
        master = master.merge(dfs[t][['id', t]], on='id', how='outer')
    master['description'] = master['description'].fillna("")
    
    train, val = train_test_split(master, test_size=0.15, random_state=SEED)

    # Class-Balanced Weight Calculation
    weights = {
        'c2a': get_cb_weights([15000, 1000]), 
        'dbo': get_cb_weights([14000, 1500, 400, 100]),
        'vio': get_cb_weights([15500, 100, 100, 100, 100, 100]),
        'def': get_cb_weights([15800, 200])
    }

    tokenizer = AutoTokenizer.from_pretrained(m_name)
    model = TicketClassifier(m_name).to(DEVICE)
    
    loader = DataLoader(HaystackDataset(train, tokenizer), batch_size=BATCH, shuffle=True)
    v_loader = DataLoader(HaystackDataset(val, tokenizer), batch_size=BATCH)

    opt = AdamW(model.parameters(), lr=LR)
    crit = {t: torch.nn.CrossEntropyLoss(weight=weights[t]) for t in ['c2a', 'dbo', 'vio', 'def']}

    best_mean_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for b in tqdm(loader, desc=f"Epoch {epoch+1}"):
            opt.zero_grad()
            ids, mask = b['ids'].to(DEVICE), b['mask'].to(DEVICE)
            preds = model(ids, mask)
            loss = sum([crit[t](preds[t], b['targets'][t].to(DEVICE)) for t in ['c2a', 'dbo', 'vio', 'def']])
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        all_p, all_l = {t: [] for t in ['c2a', 'dbo', 'vio', 'def']}, {t: [] for t in ['c2a', 'dbo', 'vio', 'def']}
        with torch.no_grad():
            for b in v_loader:
                ids, mask = b['ids'].to(DEVICE), b['mask'].to(DEVICE)
                preds = model(ids, mask)
                for t in ['c2a', 'dbo', 'vio', 'def']:
                    all_p[t].extend(torch.argmax(preds[t], dim=1).cpu().numpy())
                    all_l[t].extend(b['targets'][t].numpy())
        
        current_f1s = [f1_score(all_l[t], all_p[t], average='macro') for t in ['c2a', 'dbo', 'vio', 'def']]
        mean_f1 = np.mean(current_f1s)
        print(f"Epoch {epoch+1} | Mean F1: {mean_f1:.4f}")

        # Save logic for Ensemble Phase
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            model_slug = m_name.split('/')[-1]
            torch.save(model.state_dict(), f"best_ticket_model_{model_slug}.bin")
            print(f"✔️ New best weights saved for {model_slug}")

    # Final Summary
    print(f"\nFinal Baseline Results for {m_name}:")
    for t in ['c2a', 'dbo', 'vio', 'def']:
        print(f"\nTask: {t.upper()}")
        print(classification_report(all_l[t], all_p[t], zero_division=0))

if __name__ == "__main__":
    for m in MODELS: 
        run_experiment(m)