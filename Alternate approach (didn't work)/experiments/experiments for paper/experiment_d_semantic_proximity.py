import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

# Configuration
SEED = 1337
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Representative terms for German political discourse and extremist dog-whistles
TERMS = [
    "Demokratie", "Freiheit", "Grundgesetz",              # Neutral/Positive
    "Volksverräter", "Lügenpresse", "Remigration",       # High-Risk/Dog-Whistles
    "Gewalt", "Widerstand", "Umsturz",                    # Action-Oriented
    "PEGIDA", "Merkel", "Abschiebung",                   # Context-Specific
    "Frieden", "Heimat", "Patriot"                        # Contested Semantics
]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.TicketClassifier import TicketClassifier

def run_semantic_proximity_analysis():
    # Comparison between the Multilingual (XLMR) and Native (BERT) backbones
    models_to_compare = {
        "XLM-R": "FacebookAI/xlm-roberta-base",
        "German-BERT": "google-bert/bert-base-german-cased"
    }

    plt.figure(figsize=(16, 8))

    for i, (name, m_id) in enumerate(models_to_compare.items()):
        print(f"Generating Semantic Map for: {name}")
        tokenizer = AutoTokenizer.from_pretrained(m_id)
        model = TicketClassifier(m_id).to(DEVICE)
        
        # Load weights from previous runs
        m_slug = m_id.split('/')[-1]
        weight_path = f"best_ticket_model_{m_slug}.bin"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        
        model.eval()
        embeddings = []
        
        # Determine correct attribute for the transformer backbone
        backbone = getattr(model, 'bert', None) or getattr(model, 'roberta', None) or getattr(model, 'encoder', None)

        for term in TERMS:
            inputs = tokenizer(term, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                # Extract the [CLS] token embedding as the semantic representation
                outputs = backbone(**inputs)
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)

        # Dimensionality reduction via t-SNE
        tsne = TSNE(n_components=2, perplexity=min(5, len(TERMS)-1), random_state=SEED, init='pca')
        reduced = tsne.fit_transform(np.array(embeddings))

        # Plotting
        plt.subplot(1, 2, i+1)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
        for j, term in enumerate(TERMS):
            plt.annotate(term, (reduced[j, 0], reduced[j, 1]), fontsize=9)
        plt.title(f"Semantic Space: {name}")
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("experiment_d_semantic_proximity_map.png")
    print("✅ Experiment D Complete. Map saved as experiment_d_semantic_proximity_map.png")

if __name__ == "__main__":
    run_semantic_proximity_analysis()