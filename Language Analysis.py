'''The Language Similarity'''
# We have employed the spacy and meduim range model to undertake the language similarity in between bots and Human
# Since meduim model is quite a heavy, we have to resort to calculation individual similarity of bot and human driven content. 

'''1.1. Anti-Environment'''
import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Function to calculate average similarity between bot and human texts in a dataset
def calculate_similarity(df):
    # Drop rows with NaN in 'cleaned' column
    df = df.dropna(subset=['cleaned'])
    
    # Ensure all texts are strings
    df['cleaned'] = df['cleaned'].astype(str)
    
    bot_texts = df[df['classification'] == 'bot']['cleaned'].tolist()
    human_texts = df[df['classification'] == 'human']['cleaned'].tolist()

    bot_docs = [nlp(text) for text in bot_texts]
    human_docs = [nlp(text) for text in human_texts]

    similarity_scores = []
    for bot_doc in bot_docs:
        for human_doc in human_docs:
            similarity_scores.append(bot_doc.similarity(human_doc))
    
    avg_similarity = np.mean(similarity_scores)
    return avg_similarity

# Load a single dataset
df1 = pd.read_csv('sentiments_anti_env_corpus.csv')

# Calculate similarity for the dataset
avg_similarity = calculate_similarity(df1)
print(f'Average Similarity for Anti-Environment: {avg_similarity}')

'''1.2. Anti-Vaccine'''
# Load a single dataset
df2 = pd.read_csv('sentiments_anti_vacc_corpus.csv')

# Calculate similarity for the dataset
avg_similarity = calculate_similarity(df2)
print(f'Average Similarity for Anti-Vaccine: {avg_similarity}')

# Visualize the result
similarity_df = pd.DataFrame([{'Dataset': 'Anti-Vaccine', 'Average Similarity': avg_similarity}])

plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Average Similarity', data=similarity_df)
plt.title('Average Language Similarity between Bot and Human Texts in Anti-Vaccine Dataset')
plt.xlabel('Dataset')
plt.ylabel('Average Similarity')
plt.ylim(0, 1)  # Similarity scores range from 0 to 1
plt.show()

'''1.3 Pro Environment'''
# Load pro-environment dataset
df3 = pd.read_csv('sentiments_pro_env_corpus.csv')

# Calculate similarity for the dataset
avg_similarity = calculate_similarity(df3)
print(f'Average Similarity for Pro-Environment: {avg_similarity}')

# Visualize the result
similarity_df = pd.DataFrame([{'Dataset': 'Pro-Environment', 'Average Similarity': avg_similarity}])

plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Average Similarity', data=similarity_df)
plt.title('Average Language Similarity between Bot and Human Texts in Pro-Environment Dataset')
plt.xlabel('Dataset')
plt.ylabel('Average Similarity')
plt.ylim(0, 1)  # Similarity scores range from 0 to 1
plt.show()

'''1.4. Pro Vaccine
# Load pro-vaccine dataset
df4 = pd.read_csv('sentiments_pro_vacc_corpus.csv')

# Calculate similarity for the dataset
avg_similarity = calculate_similarity(df4)
print(f'Average Similarity for Pro-Vaccine: {avg_similarity}')

# Visualize the result
similarity_df = pd.DataFrame([{'Dataset': 'Pro-Vaccine', 'Average Similarity': avg_similarity}])

plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Average Similarity', data=similarity_df)
plt.title('Average Language Similarity between Bot and Human Texts in Pro-Vaccine Dataset')
plt.xlabel('Dataset')
plt.ylabel('Average Similarity')
plt.ylim(0, 1)  # Similarity scores range from 0 to 1
plt.show()
