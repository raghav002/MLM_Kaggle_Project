import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the dataset (assuming train.csv is the file name)
df = pd.read_csv('train.csv')

# Drop rows with missing keywords (if any)
df = df.dropna(subset=['keyword'])

# Group by keyword and target, and count the number of occurrences
grouped = df.groupby(['keyword', 'target']).size().reset_index(name='count')

# Get the total count of tweets for each keyword
keyword_total = df.groupby('keyword').size().reset_index(name='total')

# Merge the grouped counts with the total counts for each keyword
merged = pd.merge(grouped, keyword_total, on='keyword')

# Calculate the percentage of tweets for target 0 and target 1
merged['percentage'] = (merged['count'] / merged['total']) * 100

# Get unique keywords and split them into 3 roughly equal-sized groups
keywords = merged['keyword'].unique()
n_keywords = len(keywords)
n_plots = 3
keywords_per_plot = math.ceil(n_keywords / n_plots)

# Create a single subplot
fig, ax = plt.subplots(figsize=(14, 8))

# Get the keywords for the first subset
keyword_subset = keywords[:keywords_per_plot]

# Filter the merged DataFrame for the first subset of keywords
subset = merged[merged['keyword'].isin(keyword_subset)]

# Plot the first subset
sns.barplot(data=subset, x='keyword', y='percentage', hue='target', ax=ax)

# Customize the subplot
ax.set_xlabel('Keyword')
ax.set_ylabel('Percentage of Tweets')
ax.set_title('Keyword by Target Analysis (Part 1)')
ax.tick_params(axis='x', rotation=90)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()