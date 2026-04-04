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
EPOCHS = 8         
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path logic: Up 2 levels to GermEval2026/ then into EDA/
DATA_PATH = "../../EDA/using-trial-data/vio_trial.csv"

# 2. SUBTASK 3 MULTI-CLASS ARCHITECTURE (6 CLASSES)
class VIOClassifier(nn.Module):
    def __init__(self, model_name):
        super(VIOClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        # 6 Output Classes: nothing, propensity, call2Violence, support, glorification, other
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6) 
        )

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        return self.head(out.last_hidden_state[:, 0, :])

# 3. DATASET CLASS
class VIODataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Mapping for 6 violence categories
        self.label_map = {
            'nothing': 0, 
            'propensity': 1, 
            'call2violence': 2, 
            'support': 3,
            'glorification': 4,
            'other': 5
        }

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(str(row['description']), max_length=self.max_len, 
                                  padding='max_length', truncation=True, return_tensors='pt')
        
        # Normalize labels to lowercase for robust matching
        label_str = str(row['vio']).lower()
        return {
            'ids': encoding['input_ids'].flatten(),
            'mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        }

# 4. TRAINING LOOP WITH SCHEDULER & BEST-MODEL TRACKING
def train_subtask_3():
    print(f"Starting Training for Subtask 3 (Violence) on {DEVICE}...")
    
    df = pd.read_csv(DATA_PATH, sep=';')
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VIOClassifier(MODEL_NAME).to(DEVICE)
    
    # Calculate Dynamic Class Weights to handle the "Extreme Imbalance"
    labels_train = train_df['vio'].str.lower().map({
        'nothing': 0, 'propensity': 1, 'call2violence': 2, 
        'support': 3, 'glorification': 4, 'other': 5
    }).values
    
    # Filter out any NaNs if trial data is messy
    valid_mask = ~np.isnan(labels_train)
    labels_train = labels_train[valid_mask].astype(int)
    
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)
    full_weights = np.ones(6)
    for i, label_idx in enumerate(np.unique(labels_train)):
        full_weights[label_idx] = weights[i]
    
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(DEVICE)
    print(f"Dynamic weights applied: {full_weights}")

    train_loader = DataLoader(VIODataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VIODataset(val_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=len(train_loader) * EPOCHS
    )

    best_macro_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(batch['ids'].to(DEVICE), batch['mask'].to(DEVICE))
            loss = criterion(outputs, batch['label'].to(DEVICE))
            loss.backward()
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

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            torch.save(model.state_dict(), 'subtask3_vio_best_model.bin')
            print("--> Best Model Saved!")

    # Final Report using the best model logic
    print("\n--- Final Subtask 3 Report ---")
    print(classification_report(
        all_labels, all_preds, labels=[0,1,2,3,4,5],
        target_names=['Nothing', 'Propensity', 'Call2Vio', 'Support', 'Glorify', 'Other'],
        zero_division=0
    ))

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        train_subtask_3()
    else:
        print(f"Error: vio_trial.csv not found at {DATA_PATH}")