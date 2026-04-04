import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# 1. CONFIGURATION
MODEL_NAME = "answerdotai/ModernBERT-base" 
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8         # Increased slightly because we now use a scheduler
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "../../EDA/using-trial-data/dbo_trial.csv"

# 2. ARCHITECTURE
class DBOClassifier(nn.Module):
    def __init__(self, model_name):
        super(DBOClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Increased dropout slightly for better generalization
            nn.Linear(256, 4) 
        )

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        return self.head(out.last_hidden_state[:, 0, :])

# 3. DATASET
class DBODataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3}

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(str(row['description']), max_length=self.max_len, 
                                  padding='max_length', truncation=True, return_tensors='pt')
        return {
            'ids': encoding['input_ids'].flatten(),
            'mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.label_map.get(str(row['dbo']).lower(), 0), dtype=torch.long)
        }

# 4. MATURE TRAINING LOOP
def train_subtask_2():
    df = pd.read_csv(DATA_PATH, sep=';')
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DBOClassifier(MODEL_NAME).to(DEVICE)
    
    # --- DYNAMIC CLASS WEIGHTING ---
    # Calculates weight = total_samples / (n_classes * n_samples_in_class)
    labels = train_df['dbo'].str.lower().map({'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3}).values
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    # Ensure we have weights for all 4 classes even if some are missing in trial
    full_weights = np.ones(4)
    for i, label_idx in enumerate(np.unique(labels)):
        full_weights[label_idx] = weights[i]
    
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(DEVICE)
    print(f"Computed Dynamic Weights: {full_weights}")

    train_loader = DataLoader(DBODataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DBODataset(val_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- SCHEDULER ---
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    best_macro_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(batch['ids'].to(DEVICE), batch['mask'].to(DEVICE))
            loss = criterion(outputs, batch['label'].to(DEVICE))
            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['ids'].to(DEVICE), batch['mask'].to(DEVICE))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} | Val Macro-F1: {current_f1:.4f}")

        # --- SAVE BEST MODEL ---
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            torch.save(model.state_dict(), 'subtask2_dbo_best_model.bin')
            print("--> New Best Model Saved!")

if __name__ == "__main__":
    train_subtask_2()