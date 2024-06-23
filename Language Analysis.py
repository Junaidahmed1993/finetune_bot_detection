'''1. The Language Similarity'''
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


'''Results 
Average Similarity for Anti-Environment: 0.6665716171264648
Average Similarity for Anti-Vaccine: 0.7359713912010193
Average Similarity for Pro-Environment: 0.5394230484962463
Average Similarity for Pro-Vaccine: 0.5021994709968567
'''
'''TextStat: Langueg Difficulty Analysis'''
# Load the datasets 
# We have loaded our datasets iteratively. 
import pandas as pd
df = pd.read_csv('Datasets.csv')
import numpy as np

'''1. Flesh Reading Ease'''
# Replace NaN values with empty strings
df['cleaned'].fillna('', inplace=True)
# Apply the function
df['flesch_rT'] = df['cleaned'].apply(textstat.flesch_reading_ease)
# Labels score
def labelize_flesch_rT(score):
    if score > 150:
        return 'Very Easy'
    elif 120 <= score <= 149:
        return 'Easy'
    elif 80 <= score < 120:
        return 'Fairly Easy'
    elif 50 <= score < 79:
        return 'Easy'  
    elif 20 <= score < 49:
        return 'Difficult'
    elif 0 <= score < 19:
        return 'Confusing'
    elif score < 0:
        return 'Highly Confusing'
    else:
        return 'Invalid Score'  
    
# Apply the labelize_score function to each Flesch Reading Ease score
df['FleschLabel'] = df['flesch_rT'].apply(labelize_flesch_rT)

'''2. Flesh Kinndard'''
df['flesch_Kin']=df['cleaned'].apply(textstat.flesch_kincaid_grade)
def labelize_flesch_Kin_score(score):
    if 90.0 <= score <= 100.0:
        return '5th Grade'
    elif 80.0 <= score < 90.0:
        return '6th Grade'
    elif 70.0 <= score < 80.0:
        return '7th Grade'
    elif 60.0 <= score < 70.0:
        return '8th & 9th Grade'
    elif 50.0 <= score < 60.0:
        return '10th to 12th Grade'
    elif 30.0 <= score < 50.0:
        return 'College'
    elif 10.0 <= score < 30.0:
        return 'College Graduate'
    elif 0.0 <= score < 10.0:
        return 'Professional'
    elif score > 100:
        return '1st Grade'
    else:
        return 'Invalid Score'

df['flesch_Kin_label'] = df['flesch_Kin'].apply(labelize_flesch_Kin_score)

'''3. gunning Fog'''
df['GuningFog'] =df['cleaned'].apply(textstat.gunning_fog)
def custom_gunning_fog_label(score):
    if score <= 5:
        return 'Basic Understanding'
    elif 5 < score <= 10:
        return 'Some High School'
    elif 10 < score <= 15:
        return 'High School'
    elif 15 < score <= 20:
        return 'College Level'
    elif 20 < score <= 50:
        return 'Advanced College'
    elif 50 < score <= 100:
        return 'Graduate Level'
    elif 100 < score <= 200:
        return 'Postgraduate Level'
    elif 200 < score <= 400:
        return 'Professional Level'
    else:  
        return 'Extremely Advanced'

df['GuningFog_label'] = df['GuningFog'].apply(custom_gunning_fog_label)

'''4. Automated Reading Index'''
df['ARI'] = df['cleaned'].apply(textstat.automated_readability_index)
def define_ari_labels(score):
    if score < 0:
        return 'Pre-Kindergarten'
    elif 0 <= score < 10:
        return 'Elementary'
    elif 10 <= score < 50:
        return 'Middle School'
    elif 50 <= score < 100:
        return 'High School'
    elif 100 <= score < 150:
        return 'College'
    elif 150 <= score < 200:
        return 'Graduate'
    elif 200 <= score < 400:
        return 'Advanced Graduate'
#     else:
#         return 'Undefined'

# Apply the new label definition function to each ARI score
df['ARI_label'] = df['ARI'].apply(define_ari_labels)

'''5. Lexicon Count'''
df['LexiCount'] = df['cleaned'].apply(textstat.lexicon_count)
# Define a function to labelize lexicon counts
def labelize_lexicon_count(count):
    if count < 5:
        return "Low Word Count"
    elif 6 <= count <= 15:
        return "Moderate Word Count"
    elif 16 <= count <= 25:
        return "High Word Count"
    else:
        return "Very High Word Count"

# Apply the labelize_lexicon_count function to each lexicon count and store the results in a new column
df['LexiCount_label'] = df['LexiCount'].apply(labelize_lexicon_count)

'''6. Sentence Count'''
df['SentCount'] = df['cleaned'].apply(textstat.sentence_count)
def labelize_sentence_count(count):
    if count == 1:
        return 'One Sentence'
    else:
        return 'Multiple Sentences'

# Apply the labelize_sentence_count function to each sentence count and store the results in a new column
df['SentCount_label'] = df['SentCount'].apply(labelize_sentence_count)

'''7. Letter Count'''
df['LetterCount'] = df['cleaned'].apply(textstat.letter_count)
def labelize_letter_count(count):
    if count == 0:
        return 'No Letters'
    elif 0 < count <= 50:
        return 'Low Letters'
    elif 51 < count <= 150:
        return 'Moderate Letters'
    elif 151 < count <= 300:
        return 'High Letters'
    else:
        return 'Very High Letters'

'''SAVING THE DATA'''
# Apply the labelize_letter_count function to each letter count and store the results in a new column
df['LetterCount_label'] = df['LetterCount'].apply(labelize_letter_count)

# Now let's save all those score into new file so that we can use it while visulizing the results
data = df[['classification', 'flesch_rT', 'flesch_Kin', 'GuningFog',
       'ARI', 'LexiCount', 'SentCount', 'LetterCount', 'FleschLabel',
       'flesch_Kin_label', 'GuningFog_label', 'ARI_label', 'LexiCount_label',
       'SentCount_label', 'LetterCount_label']] 
data.to_csv('each_dataset.csv', index=False)

'''LETs MERGE THE DATASETS'''
import pandas as pd

# Load the datasets
pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Load the sentiment files
pro_env_sentiments = pd.read_csv('sentiments_pro_env_corpus.csv')
anti_vacc_sentiments = pd.read_csv('sentiments_anti_vacc_corpus.csv')
anti_env_sentiments = pd.read_csv('sentiments_anti_env_corpus.csv')

# Merge datasets with sentiments to add the 'classification' column
pro_env_merged = pd.merge(pro_env, pro_env_sentiments[['classification']], left_index=True, right_index=True, how='left')
anti_vacc_merged = pd.merge(anti_vacc, anti_vacc_sentiments[['classification']], left_index=True, right_index=True, how='left')
anti_env_merged = pd.merge(anti_env, anti_env_sentiments[['classification']], left_index=True, right_index=True, how='left')

pro_env_merged.to_csv('ProEnvLangAna.csv', index=False)
anti_vacc_merged.to_csv('AntiVaccLangAna.csv', index=False)
anti_env_merged.to_csv('AntiEnvLangAna.csv', index=False)

'''VISULIZATIONS'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification_x'] == 'bot']
pro_env_human = pro_env[pro_env['classification_x'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification_x'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification_x'] == 'human']
anti_env_bot = anti_env[anti_env['classification_x'] == 'bot']
anti_env_human = anti_env[anti_env['classification_x'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification_x'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification_x'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(pro_env_bot['FleschLabel']).union(set(pro_env_human['FleschLabel']))
labels_pro_vacc = set(pro_vacc_bot['FleschLabel']).union(set(pro_vacc_human['FleschLabel']))
labels_anti_env = set(anti_env_bot['FleschLabel']).union(set(anti_env_human['FleschLabel']))
labels_anti_vacc = set(anti_vacc_bot['FleschLabel']).union(set(anti_vacc_human['FleschLabel']))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['FleschLabel']:
        label_counts[label] += 1
    return label_counts

# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(20, 6))

plt.bar(x, [pro_env_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('FleschLabel')
plt.ylabel('Count')
plt.title('FleschLabel Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()

'''KINNARD Visuals'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification'] == 'bot']
pro_env_human = pro_env[pro_env['classification'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification'] == 'human']
anti_env_bot = anti_env[anti_env['classification'] == 'bot']
anti_env_human = anti_env[anti_env['classification'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(pro_env_bot['flesch_Kin_label']).union(set(pro_env_human['flesch_Kin_label']))
labels_pro_vacc = set(pro_vacc_bot['flesch_Kin_label']).union(set(pro_vacc_human['flesch_Kin_label']))
labels_anti_env = set(anti_env_bot['flesch_Kin_label']).union(set(anti_env_human['flesch_Kin_label']))
labels_anti_vacc = set(anti_vacc_bot['flesch_Kin_label']).union(set(anti_vacc_human['flesch_Kin_label']))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['flesch_Kin_label']:
        label_counts[label] += 1
    return label_counts

# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(20, 6))

plt.bar(x, [pro_env_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('Flesch Kinnard')
plt.ylabel('Count')
plt.title('Flesch Kinnard Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()

'''Gunning Fog Visulas'''
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification'] == 'bot']
pro_env_human = pro_env[pro_env['classification'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification'] == 'human']
anti_env_bot = anti_env[anti_env['classification'] == 'bot']
anti_env_human = anti_env[anti_env['classification'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(pro_env_bot['GuningFog_label']).union(set(pro_env_human['GuningFog_label']))
labels_pro_vacc = set(pro_vacc_bot['GuningFog_label']).union(set(pro_vacc_human['GuningFog_label']))
labels_anti_env = set(anti_env_bot['GuningFog_label']).union(set(anti_env_human['GuningFog_label']))
labels_anti_vacc = set(anti_vacc_bot['GuningFog_label']).union(set(anti_vacc_human['GuningFog_label']))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['GuningFog_label']:
        label_counts[label] += 1
    return label_counts

# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(20, 6))

plt.bar(x, [pro_env_bot_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('GuningFog_label')
plt.ylabel('Count')
plt.title('GuningFog_label Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()

'''ARI Visulas'''
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification'] == 'bot']
pro_env_human = pro_env[pro_env['classification'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification'] == 'human']
anti_env_bot = anti_env[anti_env['classification'] == 'bot']
anti_env_human = anti_env[anti_env['classification'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(map(str, pro_env_bot['ARI_label'])).union(set(map(str, pro_env_human['ARI_label'])))
labels_pro_vacc = set(map(str, pro_vacc_bot['ARI_label'])).union(set(map(str, pro_vacc_human['ARI_label'])))
labels_anti_env = set(map(str, anti_env_bot['ARI_label'])).union(set(map(str, anti_env_human['ARI_label'])))
labels_anti_vacc = set(map(str, anti_vacc_bot['ARI_label'])).union(set(map(str, anti_vacc_human['ARI_label'])))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['ARI_label']:
        if pd.notna(label):  # Check if the value is not NaN
            label_counts[label] += 1
    return label_counts


# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(20, 6))

plt.bar(x, [pro_env_bot_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts.get(label, 0) for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts.get(label, 0) for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('ARI_label')
plt.ylabel('Count')
plt.title('ARI_label Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()

'''lexicon Count Visulas'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification'] == 'bot']
pro_env_human = pro_env[pro_env['classification'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification'] == 'human']
anti_env_bot = anti_env[anti_env['classification'] == 'bot']
anti_env_human = anti_env[anti_env['classification'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(pro_env_bot['LexiCount_label']).union(set(pro_env_human['LexiCount_label']))
labels_pro_vacc = set(pro_vacc_bot['LexiCount_label']).union(set(pro_vacc_human['LexiCount_label']))
labels_anti_env = set(anti_env_bot['LexiCount_label']).union(set(anti_env_human['LexiCount_label']))
labels_anti_vacc = set(anti_vacc_bot['LexiCount_label']).union(set(anti_vacc_human['LexiCount_label']))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['LexiCount_label']:
        label_counts[label] += 1
    return label_counts

# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(18, 6))

plt.bar(x, [pro_env_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('Lexicon Count')
plt.ylabel('Count')
plt.title('Lexicon Count Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()

'''letter count visuals'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pro_env = pd.read_csv('ProEnvLangAna.csv')
pro_vacc = pd.read_csv('ProVaccLangAna.csv')
anti_env = pd.read_csv('AntiEnvLangAna.csv')
anti_vacc = pd.read_csv('AntiVaccLangAna.csv')

# Filter data into separate datasets: bot-generated and human-generated for each dataset
pro_env_bot = pro_env[pro_env['classification'] == 'bot']
pro_env_human = pro_env[pro_env['classification'] == 'human']
pro_vacc_bot = pro_vacc[pro_vacc['classification'] == 'bot']
pro_vacc_human = pro_vacc[pro_vacc['classification'] == 'human']
anti_env_bot = anti_env[anti_env['classification'] == 'bot']
anti_env_human = anti_env[anti_env['classification'] == 'human']
anti_vacc_bot = anti_vacc[anti_vacc['classification'] == 'bot']
anti_vacc_human = anti_vacc[anti_vacc['classification'] == 'human']

# Get unique labels from bot and human data for each dataset
labels_pro_env = set(pro_env_bot['LetterCount_label']).union(set(pro_env_human['LetterCount_label']))
labels_pro_vacc = set(pro_vacc_bot['LetterCount_label']).union(set(pro_vacc_human['LetterCount_label']))
labels_anti_env = set(anti_env_bot['LetterCount_label']).union(set(anti_env_human['LetterCount_label']))
labels_anti_vacc = set(anti_vacc_bot['LetterCount_label']).union(set(anti_vacc_human['LetterCount_label']))

# Create dictionaries to count label occurrences
def count_labels(data, labels):
    label_counts = {label: 0 for label in labels}
    for label in data['LetterCount_label']:
        label_counts[label] += 1
    return label_counts

# Count labels for each dataset and classification
pro_env_bot_counts = count_labels(pro_env_bot, labels_pro_env)
pro_env_human_counts = count_labels(pro_env_human, labels_pro_env)
pro_vacc_bot_counts = count_labels(pro_vacc_bot, labels_pro_vacc)
pro_vacc_human_counts = count_labels(pro_vacc_human, labels_pro_vacc)
anti_env_bot_counts = count_labels(anti_env_bot, labels_anti_env)
anti_env_human_counts = count_labels(anti_env_human, labels_anti_env)
anti_vacc_bot_counts = count_labels(anti_vacc_bot, labels_anti_vacc)
anti_vacc_human_counts = count_labels(anti_vacc_human, labels_anti_vacc)

# Combine counts from all datasets
all_labels = sorted(set(labels_pro_env) | set(labels_pro_vacc) | set(labels_anti_env) | set(labels_anti_vacc))

# Plotting bar chart for label counts
x = range(len(all_labels))
width = 0.1

plt.figure(figsize=(18, 6))

plt.bar(x, [pro_env_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Env)', color='tab:blue')
plt.bar([i + width for i in x], [pro_env_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Env)', color='tab:orange')
plt.bar([i + 2*width for i in x], [pro_vacc_bot_counts[label] for label in all_labels], width=width, label='Pro-Bot (Vacc)', color='tab:green')
plt.bar([i + 3*width for i in x], [pro_vacc_human_counts[label] for label in all_labels], width=width, label='Pro-Human (Vacc)', color='tab:red')
plt.bar([i + 4*width for i in x], [anti_env_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Env)', color='tab:purple')
plt.bar([i + 5*width for i in x], [anti_env_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Env)', color='tab:brown')
plt.bar([i + 6*width for i in x], [anti_vacc_bot_counts[label] for label in all_labels], width=width, label='Anti-Bot (Vacc)', color='tab:pink')
plt.bar([i + 7*width for i in x], [anti_vacc_human_counts[label] for label in all_labels], width=width, label='Anti-Human (Vacc)', color='tab:olive')

plt.xlabel('Letter Count')
plt.ylabel('Count')
plt.title('Letter Count Distribution - Bot vs Human (Env vs Vacc)')
plt.xticks([i + 3*width for i in x], all_labels)
plt.legend()
plt.show()
