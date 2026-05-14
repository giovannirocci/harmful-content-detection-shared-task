import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


# First version of the ensemble code (skeleton)
class EnsembleModel(torch.nn.Module):
    def __init__(self, model1, model2, hidden_size=768):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = torch.nn.Linear(hidden_size * 2, 2)

    def forward(self, input_ids1, input_ids2):
        outputs1 = self.model1(input_ids=input_ids1).logits
        outputs2 = self.model2(input_ids=input_ids2).logits
        concatenated = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(concatenated)
        return logits