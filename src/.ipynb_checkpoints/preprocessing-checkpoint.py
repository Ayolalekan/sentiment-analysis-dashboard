import pandas as pd

# Load dataset
df = pd.read_csv('data/amazon_reviews.csv')

# Keep only relevant columns
df = df[['reviews.text', 'reviews.rating', 'brand', 'categories', 'name']]

# Drop rows with missing text or rating
df = df.dropna(subset=['reviews.text', 'reviews.rating'])

# Convert ratings to integers
df['reviews.rating'] = df['reviews.rating'].astype(int)

# Quick look
print("Cleaned dataset:")
print(df.head())

# Check rating distribution
print("\nRating distribution:")
print(df['reviews.rating'].value_counts())
