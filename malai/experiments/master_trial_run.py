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
MODELS_TO_TEST = ["answerdotai/ModernBERT-base", "FacebookAI/xlm-roberta-base"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../GermEval2026/EDA/using-trial-data/"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5

# Import Blueprint from models folder
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
        labels = {t: torch.tensor(self.maps[t].get(str(row.get(t, 'nothing')).lower(), 0), dtype=torch.long) for t in self.maps}
        return {'ids': enc['input_ids'].flatten(), 'mask': enc['attention_mask'].flatten(), 'labels': labels}

# --- 3. TRAIN ENGINE ---
def train_model(model_name):
    print(f"\n{'='*60}\n🚀 STARTING EXPERIMENT: {model_name}\n{'='*60}")
    
    # A. Data Prep
    tasks = ['c2a', 'dbo', 'vio', 'def']
    dfs = {t: pd.read_csv(os.path.join(DATA_DIR, f"{t}_trial.csv"), sep=';') for t in tasks}
    master_df = dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master_df = master_df.merge(dfs[t][['id', t]], on='id', how='outer')
    master_df['description'] = master_df['description'].fillna("N/A")
    train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42)

    weights = {
        'c2a': torch.tensor([1.0, 3.0]).to(DEVICE),
        'dbo': torch.tensor([1.0, 2.0, 4.0, 5.0]).to(DEVICE),
        'vio': torch.tensor([1.0, 5.0, 10.0, 10.0, 20.0, 20.0]).to(DEVICE),
        'def': torch.tensor([1.0, 5.0]).to(DEVICE)
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = IntentAnalysisModel(model_name).to(DEVICE)
    train_loader = DataLoader(MultiTaskDataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(val_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criteria = {t: nn.CrossEntropyLoss(weight=weights[t]) for t in tasks}

    best_mean_f1 = 0
    final_val_reports = {}

    # B. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask = batch['ids'].to(DEVICE), batch['mask'].to(DEVICE)
            outputs = model(ids, mask)
            
            loss = sum([criteria[t](outputs[t], batch['labels'][t].to(DEVICE)) for t in tasks])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # C. Validation
        model.eval()
        val_res = {t: {'p': [], 'l': []} for t in tasks}
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['ids'].to(DEVICE), batch['mask'].to(DEVICE))
                for t in tasks:
                    val_res[t]['p'].extend(torch.argmax(outputs[t], dim=1).cpu().numpy())
                    val_res[t]['l'].extend(batch['labels'][t].numpy())
        
        current_f1s = {t: f1_score(val_res[t]['l'], val_res[t]['p'], average='macro') for t in tasks}
        mean_f1 = np.mean(list(current_f1s.values()))
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Mean Val F1: {mean_f1:.4f}")

        # D. Save Best State & Detailed Reports
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            final_val_reports = val_res # Capture the state of the best run
            model_slug = model_name.split('/')[-1]
            torch.save(model.state_dict(), f"best_intent_model_{model_slug}.bin")
            print(f"⭐️ New Best Model Saved for {model_slug}!")

    # E. Print detailed report for this specific model before returning
    print(f"\nDetailed Reports for {model_name}:")
    for t in tasks:
        print(f"\n--- {t.upper()} ---")
        print(classification_report(final_val_reports[t]['l'], final_val_reports[t]['p'], zero_division=0))
    
    # Return best scores for the final summary table
    return {t: f1_score(final_val_reports[t]['l'], final_val_reports[t]['p'], average='macro') for t in tasks}

if __name__ == "__main__":
    final_comparison = {}
    for m_name in MODELS_TO_TEST:
        final_comparison[m_name] = train_model(m_name)

    # F. FINAL SUMMARY TABLE
    print("\n" + "🏆" * 20)
    print("      FINAL BACKBONE COMPARISON (Best Macro-F1)")
    print("🏆" * 20)
    
    comp_df = pd.DataFrame(final_comparison).T
    print(comp_df.to_string())
    
    # Export results to CSV for your research paper
    comp_df.to_csv("backbone_comparison_results.csv")
    print("\nResults exported to 'backbone_comparison_results.csv'")