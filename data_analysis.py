import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse

# Exploratory Data Analysis for GermEval 2026 Subtasks

parser = argparse.ArgumentParser()
parser.add_argument('--c2a', default='GermEval2026/data/c2a/c2a_train_26.csv', help='Path to Call2Action CSV file')
parser.add_argument('--dbo', default='GermEval2026/data/dbo/dbo_train_26.csv', help='Path to DBO CSV file')
parser.add_argument('--vio', default='GermEval2026/data/vio/vio_train_26.csv', help='Path to Violence CSV file')
parser.add_argument('--dff', default='GermEval2026/data/def/def_train.csv', help='Path to Defamation CSV file')
parser.add_argument('--output_dir', default='plots', help='Directory to save analysis results')
args = parser.parse_args()

# 1. Load Datasets
c2a = pd.read_csv(args.c2a, sep=';')
dbo = pd.read_csv(args.dbo, sep=';')
vio = pd.read_csv(args.vio, sep=';')
def_df = pd.read_csv(args.dff, sep=';')

datasets = {'Call2Action': c2a, 'DBO': dbo, 'Violence': vio, 'Defamation': def_df}

# --- Analysis 1: Label Distribution ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
for i, (name, df) in enumerate(datasets.items()):
    label_col = df.columns[-1]
    counts = df[label_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Label Distribution: {name}')
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{args.output_dir}/label_distribution.png')

# --- Analysis 2: ID Overlap ---
overlap_matrix = pd.DataFrame(index=datasets.keys(), columns=datasets.keys())
mask = np.triu(np.ones(overlap_matrix.shape, dtype=bool), k=1)
for n1, df1 in datasets.items():
    for n2, df2 in datasets.items():
        overlap = len(set(df1['id']) & set(df2['id']))
        overlap_matrix.loc[n1, n2] = overlap
plt.figure(figsize=(10, 8))
sns.heatmap(overlap_matrix.astype(int), annot=True, fmt='d', cmap='Blues', mask=mask)
plt.title('ID Overlap between Subtasks')
plt.savefig(f'{args.output_dir}/id_overlap.png')

# --- Analysis 3: Text Length (Word Count) ---
plt.figure(figsize=(12, 6))
for name, df in datasets.items():
    df['word_count'] = df['description'].str.split().str.len()
    sns.kdeplot(df['word_count'], label=name, fill=True)
plt.title('Word Count Distribution across Subtasks')
plt.legend()
plt.savefig(f'{args.output_dir}/text_length_distribution.png')

# --- Analysis 4: Top Hashtags ---
def extract_hashtags(text):
    return re.findall(r'#\w+', str(text))
hashtags_all = [tag for df in datasets.values() for text in df['description'] for tag in extract_hashtags(text)]
hashtag_counts = pd.Series(hashtags_all).value_counts().head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=hashtag_counts.values, y=hashtag_counts.index, palette='magma')
plt.title('Top 15 Most Frequent Hashtags')
plt.savefig(f'{args.output_dir}/top_hashtags.png')
