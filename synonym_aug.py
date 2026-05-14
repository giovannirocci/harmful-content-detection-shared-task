import pandas as pd
import random
import wn
import argparse
import spacy
from tqdm import tqdm


wn.download("odenet:1.4")
nlp = spacy.load("de_core_news_sm")
random.seed(42)


def augment_text(text):
    doc = nlp(text)
    
    augmented = []
    for token in doc:
        if token.pos_ == 'NOUN':
            synsets = wn.synsets(token.text, pos='n')
        elif token.pos_ == 'VERB':
            synsets = wn.synsets(token.text, pos='v')
        else:
            synsets = []

        if synsets:
            synonyms = set()
            for lemma in synsets[0].lemmas():
                synonyms.add(lemma)
            augmented.append(random.choice(list(synonyms)))
        else:
            augmented.append(token.text)
    return ' '.join(augmented)


def augment_samples(input_file, output_file, only_positives=False):
    df = pd.read_csv(input_file, sep=';')

    text_col = 'description'
    label_col = df.columns[-1]

    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if only_positives and row[label_col] in [False, "nothing"]:
            continue
        augmented_text = augment_text(row[text_col])
        augmented_rows.append({
            'id': f"{row['id']}_aug",
            text_col: augmented_text,
            label_col: row[label_col]
        })
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_file, sep=';', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment samples in a CSV file using synonyms")
    parser.add_argument('--input_file','-i', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to save the augmented CSV file')
    parser.add_argument('--positives_only', '-p', action='store_true', help='Augment only positive samples')

    args = parser.parse_args()
    augment_samples(args.input_file, args.output_file, args.positives_only)