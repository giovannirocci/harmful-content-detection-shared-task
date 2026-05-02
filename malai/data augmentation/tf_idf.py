import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Setup Paths 
base_path = r'C:\Users\Samsa\.vscode\coding\germeval-3\malai\EDA\using-trial-data'
output_path = os.path.join(base_path, 'augmented_data')
if not os.path.exists(output_path): os.makedirs(output_path)

# We will focus on VIO as the example
df = pd.read_csv(os.path.join(base_path, 'vio_trial.csv'), sep=';')

# 3. TF-IDF Importance Analysis
# We calculate which words are 'unique' to the dataset
vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['description'].astype(str))
feature_names = vectorizer.get_feature_names_out()

def augment_by_tfidf(text, threshold=0.1):
    """
    Preserves high-TF-IDF words (signal) and modifies low-TF-IDF words (noise).
    """
    words = str(text).split()
    if len(words) < 5: return text # Don't shrink tiny tweets
    
    # Get word scores
    scores = {word.lower(): 0 for word in words}
    try:
        # Find this specific document's scores
        doc_idx = df[df['description'] == text].index[0]
        feature_index = tfidf_matrix[doc_idx, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc_idx, x] for x in feature_index])
        for idx, score in tfidf_scores:
            scores[feature_names[idx]] = score
    except:
        pass

    # Logic: If word score is low, we occasionally 'mask' it to create a variant
    augmented_words = [
        w if scores.get(w.lower(), 0) > threshold or np.random.random() > 0.3 
        else "[...]" 
        for w in words
    ]
    return " ".join(augmented_words)

# 4. Generate Augmented Samples
print("Starting TF-IDF Augmentation...")
minority_df = df[df['vio'] != 'nothing'].copy() # Focus on the extremist classes
minority_df['original_description'] = minority_df['description']
minority_df['description'] = minority_df['description'].apply(augment_by_tfidf)
minority_df['is_augmented'] = True

final_aug_ds = pd.concat([df, minority_df]).drop_duplicates(subset=['description'])
final_aug_ds.to_csv(os.path.join(output_path, 'AUGMENTED_VIO_DATA.csv'), index=False, sep=';')

# 5. REPORTs
print("\n" + "="*30)
print("AUGMENTATION REPORT")
print("="*30)
print(f"Original Row Count: {len(df)}")
print(f"New Augmented Count: {len(final_aug_ds)}")
print(f"Increase in Signal: {len(minority_df)} new variants created.")
print("-" * 30)
print("SAMPLE COMPARISON:")
print(f"ORIGINAL: {minority_df['original_description'].iloc[0][:100]}...")
print(f"AUGMENTED: {minority_df['description'].iloc[0][:100]}...")
print("="*30)