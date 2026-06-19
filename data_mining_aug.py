import os
import argparse
import pandas as pd
import numpy as np
import torch
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from train import TASK_CONFIG


def extract_gold_sentences(dataset, label, label_column, text_column='description'):
    gold_sentences = dataset[dataset[label_column] == label][text_column].tolist()
    return gold_sentences


def get_detailed_instruct(query):
    return f"Instruct: Given a social media post, find sentences that are most similar to it.\nQuery: '{query}'"


def main(args):
    # Load datasets
    original = pd.read_csv(args.dataset_path, sep=';')
    pool = load_dataset(args.mining_source, split='train').to_pandas()

    # Load sentence embedding model
    model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True)

    label_column = TASK_CONFIG[args.task_name]['label_col']
    text_column = TASK_CONFIG[args.task_name]['text_col']
    labels = TASK_CONFIG[args.task_name]['labels']
    

    # Prepare pool sentences and their embeddings
    pool_hateful = pool[pool['labels'] == 1]['text'].tolist()

    os.makedirs(args.cache_dir, exist_ok=True)
    pool_embeddings_path = os.path.join(args.cache_dir, f'{args.mining_source.split("/")[-1]}_pool_embeddings.npy')
    if os.path.exists(pool_embeddings_path) and not args.force_recompute:
        pool_embeddings = np.load(pool_embeddings_path)
    else:
        pool_texts = ["Passage: " + sentence for sentence in pool_hateful]
        pool_embeddings = model.encode(pool_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        np.save(pool_embeddings_path, pool_embeddings)

    # Build FAISS index
    dimension = pool_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(pool_embeddings)

    augmented_rows = []
    seen_indices = set()
    for label in labels:
        if label in ['nothing', False]: 
            continue
        gold = extract_gold_sentences(original, label, label_column, text_column)
        gold_prefixed = [get_detailed_instruct(sentence) for sentence in gold]
        gold_embs = model.encode(gold_prefixed, normalize_embeddings=True)
        for prototype in gold_embs:
            query = np.asarray(prototype, dtype=np.float32).reshape(1, -1)
            scores, indices = index.search(query, k=args.k_neighbors)
            for score, idx in zip(scores[0], indices[0]):
                if score < args.similarity_threshold:
                    continue
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
                augmented_rows.append({
                    'id': f"{label}_aug_{len(seen_indices)}",
                    text_column: pool_hateful[idx],
                    label_column: label
                })

    # Save augmented data
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.drop_duplicates(subset=[text_column], inplace=True)
    augmented_df.to_csv(args.output_file, sep=';', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data mining augmentation for social media posts")
    parser.add_argument('--dataset_path', '-d', type=str, required=True, help='Path to the original dataset CSV file')
    parser.add_argument('--task_name', '-t', type=str, required=True, help='Short name of the task (e.g., "def", "c2a", "dbo")')
    parser.add_argument('--mining_source', '-m', type=str, default='manueltonneau/german-hate-speech-superset', help='Hugging Face dataset to mine from')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to save the augmented CSV file')
    parser.add_argument('--cache_dir', '-c', type=str, default='_embeddings_cache', help='Directory to cache embeddings')
    parser.add_argument('--k_neighbors', '-k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--similarity_threshold', '-s', type=float, default=0.75, help='Minimum cosine similarity to include a neighbor')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation of embeddings even if cached')

    args = parser.parse_args()
    main(args)
    