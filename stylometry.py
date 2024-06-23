import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load datasets
df1 = pd.read_csv('sentiments_anti_env_corpus.csv')
df2 = pd.read_csv('sentiments_anti_vacc_corpus.csv')
df3 = pd.read_csv('sentiments_pro_env_corpus.csv')
df4 = pd.read_csv('sentiments_pro_vacc_corpus.csv')

# Combine datasets into a single DataFrame
df1['label'] = 'Anti-Environment'
df2['label'] = 'Anti-Vaccine'
df3['label'] = 'Pro-Environment'
df4['label'] = 'Pro-Vaccine'

combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Replace ellipsis and None with empty strings
combined_df['cleaned'] = combined_df['cleaned'].apply(lambda x: "" if x is ... or x is None else x)

# Ensure all entries are strings
combined_df['cleaned'] = combined_df['cleaned'].astype(str)

# Define functions for stylometric features
def type_token_ratio(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    types = set(tokens)
    return len(types) / len(tokens)

def hapax_legomena(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    freq = Counter(tokens)
    hapaxes = [word for word, count in freq.items() if count == 1]
    return len(hapaxes)

def yules_k(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    freq = Counter(tokens)
    N = len(tokens)
    M = sum([count * count for count in freq.values()])
    if N == 0:
        return 0
    K = 10000 * (M - N) / (N * N)
    return K

# Apply functions to calculate features
combined_df['ttr'] = combined_df['cleaned'].apply(type_token_ratio)
combined_df['hapax_legomena'] = combined_df['cleaned'].apply(hapax_legomena)
combined_df['yules_k'] = combined_df['cleaned'].apply(yules_k)

# Compare average linguistic diversity metrics between bots and humans
print(combined_df.groupby('classification')[['ttr', 'hapax_legomena', 'yules_k']].mean())

# Vectorize text data to get character n-grams
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
X = vectorizer.fit_transform(combined_df['cleaned']).toarray()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA results
plt.figure(figsize=(10, 7))
for classification, marker, color in zip(['human', 'bot'], ['o', '^'], ['blue', 'red']):
    plt.scatter(X_pca[combined_df['classification'] == classification, 0], 
                X_pca[combined_df['classification'] == classification, 1], 
                marker=marker, 
                color=color, 
                alpha=0.7, 
                label=classification)

plt.legend()
plt.title('PCA of Stylometric Features (Bots vs. Humans)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Stylometric analysis for each dataset separately
for label, df in [('Anti-Environment', df1), ('Anti-Vaccine', df2), ('Pro-Environment', df3), ('Pro-Vaccine', df4)]:
    df['cleaned'] = df['cleaned'].apply(lambda x: "" if x is ... or x is None else x)
    df['cleaned'] = df['cleaned'].astype(str)
    df['ttr'] = df['cleaned'].apply(type_token_ratio)
    df['hapax_legomena'] = df['cleaned'].apply(hapax_legomena)
    df['yules_k'] = df['cleaned'].apply(yules_k)
    
    # Compare average linguistic diversity metrics between bots and humans for each dataset
    print(f'{label} Dataset:')
    print(df.groupby('classification')[['ttr', 'hapax_legomena', 'yules_k']].mean())
    
    # Vectorize text data to get character n-grams
    X = vectorizer.fit_transform(df['cleaned']).toarray()

    # Standardize features
    X_scaled = scaler.fit_transform(X)

    # PCA for dimensionality reduction
    X_pca = pca.fit_transform(X_scaled)

    # Plot the PCA results
    plt.figure(figsize=(10, 7))
    for classification, marker, color in zip(['human', 'bot'], ['o', '^'], ['blue', 'red']):
        plt.scatter(X_pca[df['classification'] == classification, 0], 
                    X_pca[df['classification'] == classification, 1], 
                    marker=marker, 
                    color=color, 
                    alpha=0.7, 
                    label=classification)

    plt.legend()
    plt.title(f'PCA of Stylometric Features ({label} Dataset)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


'''lexical diversity'''
import pandas as pd
import numpy as np
import spacy
import seaborn as sns
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Function to extract stylometric features
def extract_stylometric_features(df, text_column):
    features = []
    
    for text in df[text_column].astype(str):  # Convert to string type
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        num_tokens = len(tokens)
        num_sentences = len(sentences)
        
        if num_tokens > 0:
            avg_word_length = np.mean([len(token) for token in tokens])
            lexical_diversity = len(set(tokens)) / num_tokens
        else:
            avg_word_length = 0
            lexical_diversity = 0
            
        if num_sentences > 0:
            avg_sentence_length = num_tokens / num_sentences
        else:
            avg_sentence_length = 0

        features.append({
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': avg_sentence_length,
        })

    return pd.DataFrame(features)

# Load datasets
df1 = pd.read_csv('sentiments_anti_env_corpus.csv', dtype=str)
df2 = pd.read_csv('sentiments_anti_vacc_corpus.csv', dtype=str)
df3 = pd.read_csv('sentiments_pro_env_corpus.csv', dtype=str)
df4 = pd.read_csv('sentiments_pro_vacc_corpus.csv', dtype=str)

# Extract features for each dataset
datasets = {
    'Anti-Environment': df1,
    'Anti-Vaccine': df2,
    'Pro-Environment': df3,
    'Pro-Vaccine': df4
}

stylometric_features = []

for label, df in datasets.items():
    if 'filter_token' in df.columns:
        text_column = 'filter_token'
    elif 'cleaned' in df.columns:
        text_column = 'cleaned'
    else:
        raise ValueError("DataFrame must contain either 'filter_token' or 'cleaned' column.")
    
    features = extract_stylometric_features(df, text_column)
    features['Dataset'] = label
    features['classification'] = df['classification']  # Ensure classification is included
    stylometric_features.append(features)

# Combine all features into a single DataFrame
combined_features = pd.concat(stylometric_features, ignore_index=True)

# Convert 'classification' column to categorical type
combined_features['classification'] = combined_features['classification'].astype('category')

# Visualize stylometric features for bot and human separately
plt.figure(figsize=(12, 8))

# Average Word Length
plt.subplot(3, 1, 1)
sns.boxplot(x='Dataset', y='avg_word_length', hue='classification', data=combined_features)
plt.title('Average Word Length')

# Lexical Diversity
plt.subplot(3, 1, 2)
sns.boxplot(x='Dataset', y='lexical_diversity', hue='classification', data=combined_features)
plt.title('Lexical Diversity')

# Average Sentence Length
plt.subplot(3, 1, 3)
sns.boxplot(x='Dataset', y='avg_sentence_length', hue='classification', data=combined_features)
plt.title('Average Sentence Length')

plt.tight_layout()
plt.show()

# Creating tabular results 
# Create and display summary tables for each dataset
summary_tables = {}

for label, df in datasets.items():
    # Extract features
    if 'filter_token' in df.columns:
        text_column = 'filter_token'
    elif 'cleaned' in df.columns:
        text_column = 'cleaned'
    else:
        raise ValueError("DataFrame must contain either 'filter_token' or 'cleaned' column.")
    
    features = extract_stylometric_features(df, text_column)
    features['classification'] = df['classification'].astype('category')

    # Group by classification and calculate mean for each feature
    summary_table = features.groupby('classification').mean()
    summary_tables[label] = summary_table

    # Print the summary table
    print(f'Summary Table - {label}')
    print(summary_table)
    print('\n')

# Optional: Save summary tables to CSV files
for label, summary_table in summary_tables.items():
    summary_table.to_csv(f'summary_table_{label.replace(" ", "_").lower()}.csv')
