from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --- Paths & checks ---
data_path = Path("data/cleaned_reviews.csv")
assert data_path.exists(), f"Missing file: {data_path} (did you save cleaned_reviews.csv?)"

# --- Load cleaned data ---
df = pd.read_csv(data_path)
df = df[['clean_text', 'sentiment']].dropna()
# drop empty strings (just in case)
df = df[df['clean_text'].astype(str).str.strip().astype(bool)]

print(f"Rows after cleaning: {len(df):,}")
print(df['sentiment'].value_counts())

# --- Train/test split ---
X = df['clean_text'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

# --- TF-IDF vectorizer ---
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams for better signal
    min_df=5,             # ignore very rare terms
    max_df=0.9,           # ignore super-common terms
    max_features=50000,   # cap vocabulary for memory/speed
    lowercase=False       # we already lowercased during cleaning
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print("\nTF-IDF shapes:")
print("  X_train_tfidf:", X_train_tfidf.shape)
print("  X_test_tfidf :", X_test_tfidf.shape)

# peek at a few feature names so you know it worked
feature_names = vectorizer.get_feature_names_out()
print("\nSample features:", list(feature_names[:20]))

# --- Save vectorizer for reuse ---
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("\n✅ Saved vectorizer -> models/tfidf_vectorizer.joblib")
