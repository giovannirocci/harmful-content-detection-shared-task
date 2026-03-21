import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# --- 1. CONFIG ---
MODEL_NAME = "answerdotai/ModernBERT-base" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated path to match: ~/GermanEval2026/src/GermEval2026/EDA/using-trial-data
DATA_DIR = "../GermEval2026/EDA/using-trial-data/"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5

# Import Blueprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.IntentAnalysisModel import IntentAnalysisModel

# --- 2. DATASET ---
class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.maps = {
            'c2a': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1},
            'dbo': {'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3},
            'vio': {'nothing': 0, 'propensity': 1, 'call2violence': 2, 'support': 3, 'glorification': 4, 'other': 5},
            'def': {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1}
        }

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(str(row['description']), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        labels = {}
        for t in self.maps:
            val = str(row.get(t, 'nothing')).lower()
            labels[t] = torch.tensor(self.maps[t].get(val, 0), dtype=torch.long)
            
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'labels': labels
        }

# --- 3. RUN ENGINE ---
def run_master_trial():
    print(f"🚀 Initializing IntentAnalysis Multi-Task Pipeline: {MODEL_NAME}")
    
    # A. DATA INGESTION & ROBUST MERGE
    tasks = ['c2a', 'dbo', 'vio', 'def']
    
    # Verify path exists before loading
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Data directory not found at: {os.path.abspath(DATA_DIR)}")

    dfs = {t: pd.read_csv(os.path.join(DATA_DIR, f"{t}_trial.csv"), sep=';') for t in tasks}
    
    master_df = dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master_df = master_df.merge(dfs[t][['id', t]], on='id', how='outer')
    
    master_df['description'] = master_df['description'].fillna("N/A")
    train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42)

    # B. DYNAMIC LOSS WEIGHTS
    weights = {
        'c2a': torch.tensor([1.0, 3.0]).to(DEVICE),
        'dbo': torch.tensor([1.0, 2.0, 4.0, 5.0]).to(DEVICE),
        'vio': torch.tensor([1.0, 5.0, 10.0, 10.0, 20.0, 20.0]).to(DEVICE),
        'def': torch.tensor([1.0, 5.0]).to(DEVICE)
    }

    # C. INIT MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = IntentAnalysisModel(MODEL_NAME).to(DEVICE)
    
    train_loader = DataLoader(MultiTaskDataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(val_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criteria = {t: nn.CrossEntropyLoss(weight=weights[t]) for t in tasks}

    # D. TRAIN & BEST-MODEL TRACKING
    best_avg_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask = batch['ids'].to(DEVICE), batch['mask'].to(DEVICE)
            outputs = model(ids, mask)
            
            batch_loss = sum([criteria[t](outputs[t], batch['labels'][t].to(DEVICE)) for t in tasks])
            
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # VALIDATION
        model.eval()
        val_res = {t: {'p': [], 'l': []} for t in tasks}
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['ids'].to(DEVICE), batch['mask'].to(DEVICE))
                for t in tasks:
                    val_res[t]['p'].extend(torch.argmax(outputs[t], dim=1).cpu().numpy())
                    val_res[t]['l'].extend(batch['labels'][t].numpy())
        
        current_f1s = [f1_score(val_res[t]['l'], val_res[t]['p'], average='macro') for t in tasks]
        mean_f1 = np.mean(current_f1s)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Mean Val F1: {mean_f1:.4f}")

        if mean_f1 > best_avg_f1:
            best_avg_f1 = mean_f1
            torch.save(model.state_dict(), f"best_intent_model_{MODEL_NAME.split('/')[-1]}.bin")
            print("⭐️ New Best Model Saved!")

    # E. FINAL GRAND REPORT
    print("\n" + "="*30 + "\nFINAL MULTI-TASK TRIAL REPORT\n" + "="*30)
    for t in tasks:
        print(f"\n--- Results for Subtask: {t.upper()} ---")
        print(classification_report(val_res[t]['l'], val_res[t]['p'], zero_division=0))

if __name__ == "__main__":
    run_master_trial()