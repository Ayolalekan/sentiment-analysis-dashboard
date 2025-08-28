import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

# --- Load saved TF-IDF vectorizer ---
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# --- Handle class imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

print("Before SMOTE:", dict(pd.Series(y_train).value_counts()))
print("After SMOTE :", dict(pd.Series(y_train_res).value_counts()))

# --- Train Logistic Regression ---
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)
clf.fit(X_train_res, y_train_res)

# --- Evaluate on original test set ---
y_pred = clf.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model ---
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/logreg_baseline_smote.joblib")
print("\n✅ Saved model -> models/logreg_baseline_smote.joblib")
