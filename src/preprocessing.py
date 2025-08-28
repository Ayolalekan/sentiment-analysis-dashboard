import pandas as pd

# Load dataset
df = pd.read_csv('data/amazon_reviews.csv')

# Keep only relevant columns
df = df[['reviews.text', 'reviews.rating', 'brand', 'categories', 'name']]

# Drop rows with missing text or rating
df = df.dropna(subset=['reviews.text', 'reviews.rating'])

# Convert ratings to integers
df['reviews.rating'] = df['reviews.rating'].astype(int)

# Map ratings to sentiment
def map_sentiment(rating):
    if rating in [1, 2]:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:  # 4 or 5
        return "positive"

df['sentiment'] = df['reviews.rating'].apply(map_sentiment)

# Show sample
print("Dataset with sentiment labels:")
print(df[['reviews.text', 'reviews.rating', 'sentiment']].head())

# Sentiment distribution
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())
