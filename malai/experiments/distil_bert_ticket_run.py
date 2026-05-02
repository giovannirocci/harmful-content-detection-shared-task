import sys
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# --- 1. FIREWALL CONFIGURATION ---
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_ID = "distilbert-base-multilingual-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../EDA/using-trial-data/"
MAX_TOKENS = 128
BATCH_SIZE = 24  
EPOCHS = 3       
LR_RATE = 3e-5   

# Correctly link to the Ticket-Haystack specific model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

# --- 2. DATASET ABSTRACTION ---
class TicketDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            'c2a': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1},
            'dbo': {'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3},
            'vio': {'nothing': 0, 'propensity': 1, 'prospensity': 1, 'call2violence': 2, 'support': 3, 'glorification': 4, 'other': 5},
            'def': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1}
        }

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = self.tokenizer(
            str(row['description']), 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        target_dict = {}
        for task in ['c2a', 'dbo', 'vio', 'def']:
            raw_val = str(row.get(task, 'nothing')).lower()
            target_dict[task] = torch.tensor(self.label_map[task].get(raw_val, 0), dtype=torch.long)
            
        return {
            'input_ids': tokens['input_ids'].flatten(), 
            'attn_mask': tokens['attention_mask'].flatten(), 
            'targets': target_dict
        }

# --- 3. EXECUTION ENGINE ---
def execute_distilbert_study():
    print(f"\n{'#'*60}")
    print(f"EXPERIMENT A: Multilingual DistilBERT Initialization")
    print(f"Searching for 'Winning Tickets' in a Light Multilingual Space")
    print(f"{'#'*60}")
    
    # Data Aggregation
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    
    master['description'] = master['description'].fillna("Empty")
    train_set, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    # Standard Cross-Entropy for the Benchmark Phase
    # Note: Experiment B will introduce Class-Balanced (CB) weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = TicketClassifier(MODEL_ID).to(DEVICE)
    
    train_loader = DataLoader(TicketDataset(train_set, tokenizer, MAX_TOKENS), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TicketDataset(val_set, tokenizer, MAX_TOKENS), batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=LR_RATE)
    criterion = {t: nn.CrossEntropyLoss() for t in tasks}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attn_mask'].to(DEVICE)
            
            preds = model(ids, mask)
            
            # Summed Multi-Task Loss
            batch_loss = sum([criterion[t](preds[t], batch['targets'][t].to(DEVICE)) for t in tasks])
            
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        # Validation Step
        model.eval()
        val_store = {t: {'preds': [], 'labels': []} for t in tasks}
        
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attn_mask'].to(DEVICE)
                outputs = model(ids, mask)
                
                for t in tasks:
                    val_store[t]['preds'].extend(torch.argmax(outputs[t], dim=1).cpu().numpy())
                    val_store[t]['labels'].extend(batch['targets'][t].numpy())
        
        epoch_f1s = [f1_score(val_store[t]['labels'], val_store[t]['preds'], average='macro') for t in tasks]
        print(f"--- Epoch {epoch+1} Mean F1: {np.mean(epoch_f1s):.4f} ---")

    print(f"\nFinal Classification Analysis for {MODEL_ID}:")
    summary_f1 = {}
    for t in tasks:
        print(f"\n[TASK: {t.upper()}]")
        print(classification_report(val_store[t]['labels'], val_store[t]['preds'], zero_division=0))
        summary_f1[t] = f1_score(val_store[t]['labels'], val_store[t]['preds'], average='macro')
    
    # Save Results for History Tracking
    pd.DataFrame([summary_f1], index=[MODEL_ID]).to_csv("distilbert_baseline_results.csv")
    print(f"\n✅ Experiment A (DistilBERT) Results Saved.")

if __name__ == "__main__":
    execute_distilbert_study()