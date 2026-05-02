import torch
import torch.nn as nn
from transformers import AutoModel

class TicketClassifier(nn.Module):
    """
    Search for the 'Winning Ticket' sub-networks for German Extremist patterns.
    Designed for Cross-Lingual Transfer evaluation.
    """
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        
        d_model = self.encoder.config.hidden_size
        
        
        def build_ticket_head(out_dims):
            return nn.Sequential(
                nn.Linear(d_model, 384),
                nn.Tanh(), 
                nn.Dropout(0.2),
                nn.Linear(384, out_dims)
            )

    # Cross-lingual task heads
        self.subtask_1 = build_ticket_head(2) # C2A
        self.subtask_2 = build_ticket_head(4) # DBO
        self.subtask_3 = build_ticket_head(6) # VIO
        self.subtask_4 = build_ticket_head(2) # DEF

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Using a pooled output (mean) instead of [CLS] for different signal extraction
        pooler = torch.mean(outputs.last_hidden_state, dim=1)
        
        return {
            'c2a': self.subtask_1(pooler),
            'dbo': self.subtask_2(pooler),
            'vio': self.subtask_3(pooler),
            'def': self.subtask_4(pooler)
        }