import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter

# 1. Define Paths
base_path = r'C:\Users\Samsa\.vscode\coding\germeval-3\malai\EDA\using-trial-data'
def get_path(filename):
    return os.path.join(base_path, filename)

# 2. Load Datasets
c2a = pd.read_csv(get_path('c2a_trial.csv'), sep=';')
dbo = pd.read_csv(get_path('dbo_trial.csv'), sep=';')
vio = pd.read_csv(get_path('vio_trial.csv'), sep=';')
def_df = pd.read_csv(get_path('def_trial.csv'), sep=';')

full_corpus = pd.concat([c2a, dbo, vio, def_df])['description'].astype(str)

# 3. NEW ANALYSIS 1: Cross-Lingual Semantic Overlap (The "Ticket" Search)
# Theoretical Grounding: The Lottery Ticket Hypothesis
# We check how many 'Universal Radicalization' concepts appear in German text
def analyze_semantic_overlap(texts):
    # These represent universal extremist/radical tropes often found cross-lingually
    universal_concepts = {
        'Antisemitism/Identity': ['jude', 'zion', 'globalist', 'elite', 'volk'],
        'Violence/Action': ['krieg', 'kampf', 'terror', 'vernichtung', 'angriff'],
        'State/Order': ['regime', 'diktatur', 'system', 'verräter', 'widerstand']
    }
    
    results = {}
    text_blob = " ".join(texts).lower()
    
    for category, keywords in universal_concepts.items():
        count = sum(1 for word in keywords if word in text_blob)
        results[category] = count / len(keywords) # Percentage of concept coverage
        
    return results

print("Running Analysis 1: Cross-Lingual Semantic Overlap...")
overlap_data = analyze_semantic_overlap(full_corpus)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(overlap_data.keys()), y=list(overlap_data.values()), hue=list(overlap_data.keys()), palette='viridis', legend=False)
plt.title('Analysis 1: Universal Radicalization Concept Coverage')
plt.ylabel('Concept Hit Rate (0.0 - 1.0)')
plt.xlabel('Thematic Category')
plt.savefig('semantic_concept_overlap.png')
print("✅ semantic_concept_overlap.png saved.")

# 4. ANALYSIS 2: Global vs. Local Radicalization Signals
international_tags = ['#hate', '#freedom', '#police', '#stop', '#war', '#globalism', '#truth']
local_german_tags = ['#merkelmussweg', '#pegida', '#afd', '#deutschland', '#migration', '#volksverräter']

def count_tags(text, tag_list):
    count = 0
    text = text.lower()
    for tag in tag_list:
        if tag in text: count += 1
    return count

full_df = pd.concat([c2a, dbo, vio, def_df])
full_df['intl_count'] = full_df['description'].apply(lambda x: count_tags(str(x), international_tags))
full_df['local_count'] = full_df['description'].apply(lambda x: count_tags(str(x), local_german_tags))

plt.figure(figsize=(10, 6))
tag_data = full_df[['intl_count', 'local_count']].sum()
sns.barplot(x=tag_data.index, y=tag_data.values, hue=tag_data.index, palette='coolwarm', legend=False)
plt.title('Analysis 2: Global vs. Local Radicalization Signals')
plt.ylabel('Total Occurrences')
plt.savefig('global_local_signals.png')

# 5. ANALYSIS 3: Lexical Diversity
def get_lexical_diversity(text):
    words = str(text).lower().split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)

diversity_results = {}
for name, df in {'C2A': c2a, 'DBO': dbo, 'VIO': vio, 'DEF': def_df}.items():
    diversity_results[name] = df['description'].apply(get_lexical_diversity).mean()

plt.figure(figsize=(10, 6))
labels = list(diversity_results.keys())
values = list(diversity_results.values())
sns.barplot(x=labels, y=values, hue=labels, palette='plasma', legend=False)
plt.title('Analysis 3: Lexical Diversity (Complexity of the Haystack)')
plt.ylabel('Mean Type-Token Ratio')
plt.savefig('lexical_diversity.png')

# 6. ANALYSIS 4: Winning Ticket Patterns
def get_stems(texts):
    all_words = " ".join(texts).lower().split()
    stems = [w[:4] for w in all_words if len(w) > 4 and w.isalpha()]
    return Counter(stems).most_common(15)

top_stems = get_stems(full_corpus)
stems_df = pd.DataFrame(top_stems, columns=['Stem', 'Count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=stems_df, x='Count', y='Stem', hue='Stem', palette='rocket', legend=False)
plt.title('Analysis 4: Dominant Linguistic Stems (Winning Ticket Patterns)')
plt.savefig('winning_ticket_stems.png')

print("✅ Group Project EDA Complete. All 4 charts generated without external library dependencies.")