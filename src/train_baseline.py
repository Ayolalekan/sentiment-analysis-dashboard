import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- Load cleaned dataset ---
data_path = Path("data/cleaned_reviews.csv")
df = pd.read_csv(data_path)
df = df[['clean_text', 'sentiment']].dropna()
df = df[df['clean_text'].astype(str).str.strip().astype(bool)]

X = df['clean_text'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Load saved vectorizer ---
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# --- Train baseline Logistic Regression ---
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # handle imbalance
    n_jobs=-1
)
clf.fit(X_train_tfidf, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model ---
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/logreg_baseline.joblib")
print("\n✅ Saved model -> models/logreg_baseline.joblib")
