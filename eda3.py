import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
import string

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('train.csv')

# Define a function to preprocess the text (lowercase, remove punctuation, stopwords)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    return words

# Preprocess the tweets
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Separate disaster and non-disaster tweets
disaster_tweets = df[df['target'] == 1]['cleaned_text']
non_disaster_tweets = df[df['target'] == 0]['cleaned_text']

# Flatten the lists of words
disaster_words = [word for tweet in disaster_tweets for word in tweet]
non_disaster_words = [word for tweet in non_disaster_tweets for word in tweet]

# 1. Most common words in non-disaster tweets (top 20)
non_disaster_common_words = Counter(non_disaster_words).most_common(20)
print("Top 20 most common words in non-disaster tweets:")
for word, freq in non_disaster_common_words:
    print(f"{word}: {freq}")

# 2. Most common words in disaster tweets (top 20)
disaster_common_words = Counter(disaster_words).most_common(20)
print("\nTop 20 most common words in disaster tweets:")
for word, freq in disaster_common_words:
    print(f"{word}: {freq}")

# Define a function to get n-grams from the tweets
def get_ngrams(texts, n):
    ngram_list = []
    for tweet in texts:
        ngram_list.extend(list(ngrams(tweet, n)))
    return ngram_list

# 3. Most common 2-word phrases (bigrams) for both disaster and non-disaster tweets
disaster_bigrams = get_ngrams(disaster_tweets, 2)
non_disaster_bigrams = get_ngrams(non_disaster_tweets, 2)

disaster_common_bigrams = Counter(disaster_bigrams).most_common(10)
non_disaster_common_bigrams = Counter(non_disaster_bigrams).most_common(10)

print("\nTop 10 most common 2-word phrases in disaster tweets:")
for bigram, freq in disaster_common_bigrams:
    print(f"{' '.join(bigram)}: {freq}")

print("\nTop 10 most common 2-word phrases in non-disaster tweets:")
for bigram, freq in non_disaster_common_bigrams:
    print(f"{' '.join(bigram)}: {freq}")

# 4. Most common 3-word phrases (trigrams) for both disaster and non-disaster tweets
disaster_trigrams = get_ngrams(disaster_tweets, 3)
non_disaster_trigrams = get_ngrams(non_disaster_tweets, 3)

disaster_common_trigrams = Counter(disaster_trigrams).most_common(10)
non_disaster_common_trigrams = Counter(non_disaster_trigrams).most_common(10)

print("\nTop 10 most common 3-word phrases in disaster tweets:")
for trigram, freq in disaster_common_trigrams:
    print(f"{' '.join(trigram)}: {freq}")

print("\nTop 10 most common 3-word phrases in non-disaster tweets:")
for trigram, freq in non_disaster_common_trigrams:
    print(f"{' '.join(trigram)}: {freq}")