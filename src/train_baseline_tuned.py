import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import os

# --- Load cleaned dataset ---
data_path = Path("data/cleaned_reviews.csv")
df = pd.read_csv(data_path)
df = df[['clean_text', 'sentiment']].dropna()
df = df[df['clean_text'].astype(str).str.strip().astype(bool)]

X = df['clean_text'].values
y = df['sentiment'].values

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF vectorizer (tuned) ---
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),   # unigrams + bigrams + trigrams
    min_df=5,
    max_df=0.9,
    max_features=50000,
    lowercase=False
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# --- Handle class imbalance ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
print("After SMOTE :", dict(pd.Series(y_train_res).value_counts()))

# --- Logistic Regression with hyperparameter tuning ---
# Trying a stronger regularization (smaller C)
clf = LogisticRegression(
    max_iter=1000,
    class_weight=None,   # SMOTE already balanced classes
    C=0.5,               # smaller C = stronger regularization
    n_jobs=-1
)
clf.fit(X_train_res, y_train_res)

# --- Evaluate ---
y_pred = clf.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model & vectorizer ---
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/logreg_tuned.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer_tuned.joblib")
print("\n✅ Saved model -> models/logreg_tuned.joblib")
print("✅ Saved vectorizer -> models/tfidf_vectorizer_tuned.joblib")
