import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load preprocessed dataset
df = pd.read_csv('data/amazon_reviews.csv')

# Keep only text + sentiment from earlier preprocessing
df = df[['reviews.text', 'reviews.rating']].dropna()
df['reviews.rating'] = df['reviews.rating'].astype(int)

# Map sentiment
def map_sentiment(rating):
    if rating in [1, 2]:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"
df['sentiment'] = df['reviews.rating'].apply(map_sentiment)

# Setup for cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df['clean_text'] = df['reviews.text'].apply(clean_text)

# Show before vs after
print("Before cleaning:")
print(df['reviews.text'].iloc[0])
print("\nAfter cleaning:")
print(df['clean_text'].iloc[0])


# Save cleaned dataset
df.to_csv('data/cleaned_reviews.csv', index=False)

print("\n✅ Cleaned dataset saved to data/cleaned_reviews.csv")
print("Cleaned data sample:")
print(df[['clean_text', 'sentiment']].head())



# Save cleaned dataset
df.to_csv('data/cleaned_reviews.csv', index=False)

print("\n✅ Cleaned dataset saved to data/cleaned_reviews.csv")
print("Sample of cleaned dataset:")
print(df[['clean_text', 'sentiment']].head())

