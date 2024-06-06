'''The Language Similarity'''
# We have employed the spacy and meduim range model to undertake the language similarity in between bots and Human
# Since meduim model is quite a heavy, we have to resort to calculation individual similarity of bot and human driven content. 

import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Define function for similarity calculation
def calculate_cosine_similarity(df):
    # Determine which column to use for text data
    if 'filter_token' in df.columns:
        text_column = 'filter_token'
    elif 'cleaned' in df.columns:
        text_column = 'cleaned'
    else:
        raise ValueError("DataFrame must contain either 'filter_token' or 'cleaned' column.")

    # Clean the text column to ensure all entries are strings
    df[text_column] = df[text_column].astype(str).fillna('')

    # Separate bot and human texts
    bot_texts = df[df['classification'] == 'bot'][text_column].tolist()
    human_texts = df[df['classification'] == 'human'][text_column].tolist()

    # Create spaCy documents and vectors
    bot_vectors = [nlp(text).vector for text in bot_texts]
    human_vectors = [nlp(text).vector for text in human_texts]

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(bot_vectors, human_vectors)

    # Calculate average similarity
    avg_similarity = np.mean(similarity_matrix)

    return avg_similarity, similarity_matrix

# Load datasets
df1 = pd.read_csv('sentiments_anti_env_corpus.csv')
df2 = pd.read_csv('sentiments_anti_vacc_corpus.csv')
df3 = pd.read_csv('sentiments_pro_env_corpus.csv')
df4 = pd.read_csv('sentiments_pro_vacc_corpus.csv')

# Calculate similarities for each dataset
datasets = {
    'Anti-Environment': df1,
    'Anti-Vaccine': df2,
    'Pro-Environment': df3,
    'Pro-Vaccine': df4
}

similarities = {}
similarity_matrices = {}

for label, df in datasets.items():
    avg_similarity, similarity_matrix = calculate_cosine_similarity(df)
    similarities[label] = avg_similarity
    similarity_matrices[label] = similarity_matrix

    # Optionally, save the similarity matrix to a CSV file
    sim_df = pd.DataFrame(similarity_matrix, index=[f'Bot_{i}' for i in range(len(similarity_matrix))],
                          columns=[f'Human_{j}' for j in range(len(similarity_matrix[0]))])
    sim_df.to_csv(f'language_similarity_analysis_cosine_{label}.csv')

    print(f'Average Similarity for {label}: {avg_similarity}')

# Convert similarities to lists for Plotly
datasets_list = list(similarities.keys())
avg_similarities = list(similarities.values())

# Define bar width
bar_width = 0.4  # Adjust this value to change the width of the bars

# Create bar chart
fig = go.Figure(data=[
    go.Bar(name='Average Similarity', x=datasets_list, y=avg_similarities, marker_color='indianred', width=[bar_width]*len(datasets_list))
])

# Update layout
fig.update_layout(
    title='Average Language Similarity between Bot and Human Texts in Each Dataset',
    xaxis_title='Dataset',
    yaxis_title='Average Similarity',
    yaxis=dict(range=[0, 1]),  # Similarity scores range from 0 to 1
    template='plotly_white'
)

# Show plot
fig.show()
