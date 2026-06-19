import pandas as pd
import argparse


def test_no_duplicates_in_augmented_data(args):
    original = pd.read_csv(args.original, sep=';')
    augmented = pd.read_csv(args.augmented, sep=';')

    text_column = args.text_column
    original_texts = set(original[text_column].tolist())
    augmented_texts = set(augmented[text_column].tolist())

    duplicates = original_texts.intersection(augmented_texts)
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicates between original and augmented datasets: {duplicates}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original","-o", default="GermEval2026/data/def/def_train_renamed.csv")
    parser.add_argument("--augmented", "-a", default="augmented/mining_def.csv")
    parser.add_argument("--text_column", "-t", default="description")
    args = parser.parse_args()
    
    test_no_duplicates_in_augmented_data(args)
    print("Test passed: No duplicates found between original and augmented datasets.")