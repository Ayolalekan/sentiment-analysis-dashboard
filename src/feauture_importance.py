import joblib
import pandas as pd
import numpy as np

# --- Load model and vectorizer ---
clf = joblib.load("models/logreg_tuned.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer_tuned.joblib")

# --- Get feature names and coefficients ---
feature_names = np.array(vectorizer.get_feature_names_out())
coef = clf.coef_   # shape = (n_classes, n_features)
classes = clf.classes_

# --- Show top 20 features per class ---
top_n = 20
for i, class_label in enumerate(classes):
    print(f"\nTop {top_n} features for class '{class_label}':")
    top_idx = np.argsort(coef[i])[-top_n:]   # largest coefficients
    top_features = feature_names[top_idx]
    top_scores   = coef[i][top_idx]
    for f, s in zip(top_features[::-1], top_scores[::-1]):  # descending
        print(f"{f:20} -> {s:.4f}")
