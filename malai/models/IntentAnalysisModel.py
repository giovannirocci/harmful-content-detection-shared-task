import torch
import torch.nn as nn
from transformers import AutoModel

class IntentAnalysisModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h_size = 768 
        
        # Professional heads with ReLU and Dropout for better generalization
        def create_head(output_classes):
            return nn.Sequential(
                nn.Linear(h_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_classes)
            )

        self.c2a_head = create_head(2)  # Subtask 1
        self.dbo_head = create_head(4)  # Subtask 2
        self.vio_head = create_head(6)  # Subtask 3
        self.def_head = create_head(2)  # Subtask 4

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        vec = out.last_hidden_state[:, 0, :] # CLS Pooling
        
        return {
            'c2a': self.c2a_head(vec),
            'dbo': self.dbo_head(vec),
            'vio': self.vio_head(vec),
            'def': self.def_head(vec)
        }