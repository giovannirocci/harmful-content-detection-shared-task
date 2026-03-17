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

# Path logic: Up 2 levels to GermEval2026/ then into EDA/
DATA_PATH = "../../EDA/using-trial-data/dbo_trial.csv"

# 2. SUBTASK 2 MULTI-CLASS ARCHITECTURE
class DBOClassifier(nn.Module):
    def __init__(self, model_name):
        super(DBOClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h_size = 768 
        
        # 4 Output Classes: nothing, criticism, agitation, subversive
        self.head = nn.Sequential(
            nn.Linear(h_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4) 
        )

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        vec = out.last_hidden_state[:, 0, :]
        return self.head(vec)

# 3. DATASET CLASS (Multi-class mapping)
class DBODataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Mapping for Subtask 2 classes
        self.label_map = {
            'nothing': 0, 
            'criticism': 1, 
            'agitation': 2, 
            'subversive': 3
        }

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
            # Use .get() with default 0 to handle any missing labels in trial data
            'label': torch.tensor(self.label_map.get(str(row['dbo']).lower(), 0), dtype=torch.long)
        }

# 4. TRAINING & EVALUATION LOOP
def train_subtask_2():
    print(f"Starting Training for Subtask 2 (DBO) on {DEVICE}...")
    print(f"Loading data from: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, sep=';')
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DBOClassifier(MODEL_NAME).to(DEVICE)
    
    train_dataset = DBODataset(train_df, tokenizer, MAX_LEN)
    val_dataset = DBODataset(val_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # WEIGHTED LOSS: Fixed based on your Label Distribution EDA
    # Severity increases from Nothing -> Subversive. 
    # We give highest weights to agitation and subversive.
    class_weights = torch.tensor([1.0, 2.0, 4.0, 5.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    final_val_preds, final_val_labels = [], []

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
        all_val_preds, all_val_labels = [], []
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
        
        if epoch == EPOCHS - 1:
            final_val_preds, final_val_labels = all_val_preds, all_val_labels

    # --- FINAL VALIDATION RESULTS ---
    print("\n--- Final Validation Results (Subtask 2: DBO) ---")
    print(classification_report(
        final_val_labels, 
        final_val_preds, 
        labels=[0, 1, 2, 3], 
        target_names=['Nothing', 'Criticism', 'Agitation', 'Subversive'],
        zero_division=0
    ))

    final_score = f1_score(final_val_labels, final_val_preds, average='macro')
    print(f"OFFICIAL COMPETITION METRIC (Macro-F1): {final_score:.4f}")
    
    torch.save(model.state_dict(), 'subtask2_dbo_model.bin')
    print("Model Training Complete and Saved.")

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        train_subtask_2()
    else:
        print(f"Error: dbo_trial.csv not found at {DATA_PATH}")