import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

import nlpaug.augmenter.word as naw


def augment_samples(input_file, output_file):
    df = pd.read_csv(input_file, sep=';')
    text_col = 'description'
    label_col = df.columns[-1]
    
    aug = naw.SynonymAug(aug_src='wordnet', lang='deu')

    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        augmented_text = aug.augment(row[text_col])
        augmented_rows.append({
            'id': f"{row['id']}_aug",
            text_col: augmented_text,
            label_col: row[label_col]
        })
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_file, sep=';', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment positive samples in a CSV file")
    parser.add_argument('--input_file','-i', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to save the augmented CSV file')

    args = parser.parse_args()
    augment_samples(args.input_file, args.output_file)