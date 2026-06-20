import sys
import os
import torch

# --- 1. APPTAINER / CLUSTER ENVIRONMENT HOTFIX & HARD FORCED SAFETENSORS ---
# Force the entire transformers ecosystem to completely ignore legacy bin endpoints globally
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# --- 2. FIREWALL CONFIGURATION ---
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_ID = "LSX-UniWue/ModernGBERT_134M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MAINTAINED: Reverted data paths back to your local environment file structures
DATA_PATH = "../EDA/using-trial-data/"
MAX_TOKENS = 128
BATCH_SIZE = 24  
EPOCHS = 3       
LR_RATE = 2e-5   

# Ensure the model directory is in the path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

# --- 3. DATASET ABSTRACTION ---
class TicketDataset(Dataset):
    """
    Standard multi-task dataset loader for German political discourse analysis.
    """
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

# --- 4. EXECUTION ENGINE (WITH INTEGRATED FP16 ENGINE) ---
def execute_gbert_baseline():
    """
    Executes training and evaluation for the monolingual GBERT baseline.
    """
    print("-" * 60)
    print(f"Linguistic Benchmark: {MODEL_ID}")
    print("Integrating Native Mixed-Precision Engine & Localized Datasets")
    print("-" * 60)
    
    tasks = ['c2a', 'dbo', 'vio', 'def']
    raw_dfs = {t: pd.read_csv(os.path.join(DATA_PATH, f"{t}_trial.csv"), sep=';') for t in tasks}
    master = raw_dfs['c2a']
    for t in ['dbo', 'vio', 'def']:
        master = master.merge(raw_dfs[t][['id', t]], on='id', how='outer')
    
    master['description'] = master['description'].fillna("Empty")
    train_set, val_set = train_test_split(master, test_size=0.15, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Pre-fetching clean safetensors structure explicitly
    print("Pre-fetching clean safetensors structure...")
    AutoModel.from_pretrained(MODEL_ID, use_safetensors=True)
    
    # MONKEY PATCH: Intercept the from_pretrained layer of AutoModel before building TicketClassifier
    original_from_pretrained = AutoModel.from_pretrained
    AutoModel.from_pretrained = lambda pretrained_model_name_or_path, *args, **kwargs: original_from_pretrained(
        pretrained_model_name_or_path, *args, **{**kwargs, "use_safetensors": True}
    )
    
    try:
        model = TicketClassifier(MODEL_ID).to(DEVICE)
    finally:
        # Restore original functionality to maintain framework integrity downstream
        AutoModel.from_pretrained = original_from_pretrained
    
    train_loader = DataLoader(TicketDataset(train_set, tokenizer, MAX_TOKENS), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TicketDataset(val_set, tokenizer, MAX_TOKENS), batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=LR_RATE)
    criterion = {t: torch.nn.CrossEntropyLoss() for t in tasks}

    # Native PyTorch Gradient Scaler for FP16 Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attn_mask'].to(DEVICE)
            
            # Autocast forward operations to FP16 half-precision context
            with torch.cuda.amp.autocast():
                preds = model(ids, mask)
                batch_loss = sum([criterion[t](preds[t], batch['targets'][t].to(DEVICE)) for t in tasks])
            
            # Scaler backpropagation sequence
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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

    print(f"\nFinal Performance Metrics for {MODEL_ID}:")
    summary_f1 = {}
    for t in tasks:
        print(f"\n[SUBTASK: {t.upper()}]")
        print(classification_report(val_store[t]['labels'], val_store[t]['preds'], zero_division=0))
        summary_f1[t] = f1_score(val_store[t]['labels'], val_store[t]['preds'], average='macro')
    
    # Save Results for History Tracking
    model_slug = MODEL_ID.split('/')[-1]
    pd.DataFrame([summary_f1], index=[MODEL_ID]).to_csv("gbert_baseline_results.csv")
    
    # Saving model weights for the Ensemble Phase
    torch.save(model.state_dict(), f"best_ticket_model_{model_slug}.bin")
    print(f"\n✅ Optimization cycle complete for {MODEL_ID}. Weights generated: best_ticket_model_{model_slug}.bin")

if __name__ == "__main__":
    execute_gbert_baseline()