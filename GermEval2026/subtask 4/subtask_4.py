import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

# 1. EDA-BASED CONFIGURATION
# Using ModernGBERT for native German understanding
MODEL_NAME = "answerdotai/ModernBERT-base" 
# max_len=128 covers almost 100% of tweets based on our word count density plot
MAX_LEN = 128 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. THE MULTI-HEAD ARCHITECTURE
class MultiTaskGermanClassifier(nn.Module):
    def __init__(self, model_name):
        super(MultiTaskGermanClassifier, self).__init__()
        # The shared encoder ("The Brain")
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = 768 

        # Task-Specific Heads
        self.head_c2a = nn.Linear(hidden_size, 2)  # Binary
        self.head_dbo = nn.Linear(hidden_size, 4)  # Multi-class
        self.head_vio = nn.Linear(hidden_size, 6)  # Multi-class
        self.head_def = nn.Linear(hidden_size, 2)  # Subtask 4 (Defamation)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token as the aggregated representation of the tweet
        sentence_vec = outputs.last_hidden_state[:, 0, :]

        # Simultaneous predictions
        return (self.head_c2a(sentence_vec), 
                self.head_dbo(sentence_vec), 
                self.head_vio(sentence_vec), 
                self.head_def(sentence_vec))

# 3. EDA-BASED WEIGHTED LOSS
# Solves the "Significant Class Imbalance" found in our Label Distribution EDA
def get_weighted_loss():
    # Weighting the 'TRUE' or harmful classes 5x to optimize for Macro-F1
    weight_binary = torch.tensor([1.0, 5.0]).to(DEVICE)
    # Applying weights specifically for Subtask 4 (Defamation) as requested
    return {
        "c2a": nn.CrossEntropyLoss(),
        "dbo": nn.CrossEntropyLoss(),
        "vio": nn.CrossEntropyLoss(),
        "def": nn.CrossEntropyLoss(weight=weight_binary)
    }

# 4. DATA PLUMBING: MASTER DATASET CLASS
class GermEvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['description'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Mapping labels to integers
        self.c2a = df['c2a'].map({'FALSE': 0, 'TRUE': 1}).fillna(0).astype(int).tolist()
        self.dbo = df['dbo'].map({'nothing': 0, 'criticism': 1, 'agitation': 2, 'subversive': 3}).fillna(0).astype(int).tolist()
        self.vio = df['vio'].map({'nothing': 0, 'propensity': 1, 'call2Violence': 2, 'support': 3, 'glorification': 4, 'other': 5}).fillna(0).astype(int).tolist()
        self.def_labels = df['def'].map({'FALSE': 0, 'TRUE': 1}).fillna(0).astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'c2a': torch.tensor(self.c2a[item], dtype=torch.long),
            'dbo': torch.tensor(self.dbo[item], dtype=torch.long),
            'vio': torch.tensor(self.vio[item], dtype=torch.long),
            'def': torch.tensor(self.def_labels[item], dtype=torch.long)
        }

# 5. EXECUTION BLOCK
if __name__ == "__main__":
    # Load Master CSV from Phase 1
    if os.path.exists('master_train_data.csv'):
        master_df = pd.read_csv('master_train_data.csv', sep=';')
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = MultiTaskGermanClassifier(MODEL_NAME).to(DEVICE)
        
        # Create Dataset and DataLoader
        dataset = GermEvalDataset(master_df, tokenizer, MAX_LEN)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        print(f"Success: Model built and loaded with {len(dataset)} tweets.")
        print(f"Subtask 4 Weighting applied. Ready for Macro-F1 optimization.")
    else:
        print("Please run the Phase 1 Data Plumbing script first to create 'master_train_data.csv'.")