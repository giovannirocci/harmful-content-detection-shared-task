import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# 1. EDA-BASED CONFIGURATION
MODEL_NAME = "answerdotai/ModernBERT-base" 
MAX_LEN = 128      # Optimized based on your Text Length Distribution EDA
BATCH_SIZE = 16
EPOCHS = 5         
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to data relative to: src/GermEval2026/subtask 1/using-trial-data/
# Up 1 level: subtask 1/
# Up 2 levels: GermEval2026/
DATA_PATH = "../../EDA/using-trial-data/c2a_trial.csv"

# 2. SUBTASK 1 SPECIALIZED ARCHITECTURE
class C2AClassifier(nn.Module):
    def __init__(self, model_name):
        super(C2AClassifier, self).__init__()
        # Shared Encoder (The native German "Brain")
        self.backbone = AutoModel.from_pretrained(model_name)
        h_size = 768 
        self.head = nn.Sequential(
            nn.Linear(h_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2) # TRUE (Call to Action) vs FALSE
        )

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        # Use CLS token as the aggregate representation
        vec = out.last_hidden_state[:, 0, :]
        return self.head(vec)

# 3. DATASET CLASS
class C2ADataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Label Mapping for Subtask 1
        self.label_map = {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1}

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row['description']), 
            max_length=self.max_len, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'ids': encoding['input_ids'].flatten(),
            'mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.label_map.get(str(row['c2a']), 0), dtype=torch.long)
        }

# 4. FULL TRAINING & EVALUATION LOOP
def train_subtask_1():
    print(f"Starting Training for Subtask 1 (Call2Action) on {DEVICE}...")
    print(f"Loading data from: {DATA_PATH}")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Split 80/20 to validate results internally
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = C2AClassifier(MODEL_NAME).to(DEVICE)
    
    train_dataset = C2ADataset(train_df, tokenizer, MAX_LEN)
    val_dataset = C2ADataset(val_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Weighted Loss: 3.0 weight for 'TRUE' (C2A) based on your Label Distribution EDA
    class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    final_val_preds = []
    final_val_labels = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            ids, mask = batch['ids'].to(DEVICE), batch['mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                ids, mask = batch['ids'].to(DEVICE), batch['mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(ids, mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(labels.cpu().numpy())
        
        val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        # Save results from the last epoch for the final report
        if epoch == EPOCHS - 1:
            final_val_preds = all_val_preds
            final_val_labels = all_val_labels

    # --- FINAL VALIDATION RESULTS ---
    print("\n--- Final Validation Results (Subtask 1: C2A) ---")
    print(classification_report(
        final_val_labels, 
        final_val_preds, 
        labels=[0, 1], 
        target_names=['No C2A', 'C2A'],
        zero_division=0
    ))

    final_score = f1_score(final_val_labels, final_val_preds, average='macro')
    print(f"OFFICIAL COMPETITION METRIC (Macro-F1): {final_score:.4f}")
    
    torch.save(model.state_dict(), 'subtask1_c2a_model.bin')
    print("Model Training Complete and Saved.")

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        train_subtask_1()
    else:
        print(f"Error: c2a_trial.csv not found at {DATA_PATH}")