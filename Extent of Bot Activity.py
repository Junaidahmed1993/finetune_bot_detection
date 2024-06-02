'''Note'''
# So, following code is about calculating the percentage of the bot and human activity in each of datasets.
# We have first calculated the percentage of each bot and human in each dataset and than visulize it using plotty

'''1. Anti-Enviornment'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('anti_env_corpus1.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")

'''2. Anti-Vaccine'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('anti_vaccine_corpus1.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")


'''3. Pro-environment'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('pro_env_corpus.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")

'''4. Pro vaccine'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('pro_vacc_corpus.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")

'''5. Environmemnt'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('env_dataset.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")

'''6. Vaccine'''
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('vacc_dataset.csv')

# Count the occurrences of 'bot' and 'human'
bot_count = (df['classification'] == 'bot').sum()
human_count = (df['classification'] == 'human').sum()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the percentage of 'bot' and 'human' classifications
bot_percentage = (bot_count / total_samples) * 100
human_percentage = (human_count / total_samples) * 100

print(f"Bot Percentage: {bot_percentage:.2f}%")
print(f"Human Percentage: {human_percentage:.2f}%")


#############################################################################################################
# Now, following code visulize the results based frequency in percentage of the bot vs human calculated using above code.

'''Visulization'''
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data
categories = ['Pro-Vaccine', 'Pro Environment', 'Anti vaccine', 'Anti Environment', 'Vaccine Dataset Combined', 'Environment Dataset Combined']
bot_percentages = [67.15, 70.77, 34.32, 26.63, 44.63, 50.11]
human_percentages = [32.85, 29.23, 65.68, 73.37, 55.37, 49.89]

# Create subplots
fig = make_subplots(rows=2, cols=3, subplot_titles=categories)

# Bar width
bar_width = 0.3  # Adjust the bar width

# Plotting
for i, category in enumerate(categories):
    row = (i // 3) + 1
    col = (i % 3) + 1
    
    fig.add_trace(go.Bar(
        x=['Bot'],
        y=[bot_percentages[i]],
        name='Bot',
        marker_color='red',
        width=[bar_width],
        text=[f'{bot_percentages[i]:.2f}%'],
        textposition='outside',
        showlegend=False  # Remove legend entry
    ), row=row, col=col)
    
    fig.add_trace(go.Bar(
        x=['Human'],
        y=[human_percentages[i]],
        name='Human',
        marker_color='blue',
        width=[bar_width],
        text=[f'{human_percentages[i]:.2f}%'],
        textposition='outside',
        showlegend=False  # Remove legend entry
    ), row=row, col=col)
    
    fig.update_yaxes(range=[0, 100], title_text='Percentage', row=row, col=col)

# Update layout
fig.update_layout(
    height=800, 
    width=1200, 
    title_text='Bot and Human Percentages per Category',
)

# Show plot
fig.show()
