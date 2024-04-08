import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Maxim/OneDrive/Desktop/Exeter/Year 2/COM2013 - Data Science Group Project 2/data.csv")

df.info()

rating_counts = df['RATING'].value_counts().sort_index()

plt.bar(rating_counts.index, rating_counts.values)

# Set labels and title
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Ratings')
plt.xticks(range(1, 11, 1))

# Show the plot
plt.show()

df['Review_Length'] = df['REVIEW'].apply(len)

# Compute the average length of reviews
average_review_length = df['Review_Length'].mean()
print(f"Average Review Length: {average_review_length:.2f} characters")

plt.figure(figsize=(10, 6))
plt.hist(df['Review_Length'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Number of Words in Review')
plt.xlabel('Review Length (characters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

max_length_index = df['Review_Length'].idxmax()
print(df.loc[max_length_index])

min_length_index = df['Review_Length'].idxmin()
print(df.loc[min_length_index])

from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

all_reviews = ' '.join(df['REVIEW'].astype(str))
tokens = word_tokenize(all_reviews)

# Remove stopwords (common words like 'the', 'and', etc.)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

# Calculate word frequencies
freq_dist = FreqDist(filtered_tokens)

top_n = 20
plt.figure(figsize=(12, 6))
freq_dist.plot(top_n, title=f'Top {top_n} Most Common Words in Movie Reviews', cumulative=False)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
