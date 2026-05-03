import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm

from gensim.models.fasttext import load_facebook_vectors


def augment_text(text, model, top_n=10, replace_prob=0.3):
    words = text.split()
    augmented = []
    for word in words:
        if random.random() < replace_prob:
            try:
                similar = model.most_similar(word, topn=top_n)
                augmented.append(random.choice(similar)[0])
            except KeyError:
                augmented.append(word)
        else:
            augmented.append(word)
    return ' '.join(augmented)


def augment_samples(input_file, output_file):
    df = pd.read_csv(input_file, sep=';')
    text_col = 'description'
    label_col = df.columns[-1]

    model = load_facebook_vectors('models/embedding/cc.de.300.bin')

    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if args.positives_only and row[label_col] in [False, 'nothing']:
            continue
        augmented_text = augment_text(row[text_col], model)
        augmented_rows.append({
            'id': f"{row['id']}_aug",
            text_col: augmented_text,
            label_col: row[label_col]
        })
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_file, sep=';', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment samples in a CSV file")
    parser.add_argument('--input_file','-i', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to save the augmented CSV file')
    parser.add_argument('--positives_only', '-p', action='store_true', help='Augment only positive samples')

    args = parser.parse_args()
    augment_samples(args.input_file, args.output_file)