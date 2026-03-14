import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report

# 1. EDA-BASED CONFIGURATION
MODEL_NAME = "answerdotai/ModernBERT-base" 
MAX_LEN = 128      # Covers ~100% of data based on your EDA word-count density
BATCH_SIZE = 16
EPOCHS = 5         # Increased slightly to give the model more time to learn scarce cases
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. SUBTASK 4 SPECIALIZED ARCHITECTURE
class DefamationClassifier(nn.Module):
    def __init__(self, model_name):
        super(DefamationClassifier, self).__init__()
        # Shared Encoder (The native German "Brain")
        self.backbone = AutoModel.from_pretrained(model_name)
        h_size = 768 

        # Subtask 4 specialized binary head
        self.head = nn.Sequential(
            nn.Linear(h_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2) # Output: 2 classes (Defamation vs Not)
        )

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        # Pooling: taking the CLS token which represents the whole tweet
        vec = out.last_hidden_state[:, 0, :]
        return self.head(vec)

# 3. DATASET CLASS (Specific to Subtask 4)
class DefamationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Mapping: Subtask 4 labels are true/false strings in the CSV
        self.label_map = {'false': 0, 'true': 1, 'FALSE': 0, 'TRUE': 1}

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            str(row['description']), 
            max_length=self.max_len, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'ids': encoding['input_ids'].flatten(),
            'mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.label_map.get(str(row['def']), 0), dtype=torch.long)
        }

# 4. FULL TRAINING & EVALUATION LOOP
def train_subtask_4():
    print(f"Starting Training for Subtask 4 (Defamation) on {DEVICE}...")
    
    # Load the trial data
    df = pd.read_csv('def_trial.csv', sep=';')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DefamationClassifier(MODEL_NAME).to(DEVICE)
    
    # DataLoader
    dataset = DefamationDataset(df, tokenizer, MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # LOSS & OPTIMIZER
    # Weighted Loss: 5x weight on 'True' cases to boost Macro-F1 (Imbalance Solution)
    class_weights = torch.tensor([1.0, 5.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            ids, mask = batch['ids'].to(DEVICE), batch['mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # For Metric Tracking
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate Macro-F1 after each epoch (Your primary metric)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} | Macro-F1: {macro_f1:.4f}")

    # 5. SAVE MODEL
    torch.save(model.state_dict(), 'subtask4_defamation_model.bin')
    print("Model Training Complete and Saved.")

if __name__ == "__main__":
    if os.path.exists('def_trial.csv'):
        train_subtask_4()
    else:
        print("Error: def_trial.csv not found in the current directory.")