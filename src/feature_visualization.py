import joblib
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Load model and vectorizer ---
clf = joblib.load("models/logreg_tuned.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer_tuned.joblib")

feature_names = np.array(vectorizer.get_feature_names_out())
coef = clf.coef_   # shape = (n_classes, n_features)
classes = clf.classes_

top_n = 20  # top features to plot

for i, class_label in enumerate(classes):
    # --- Top features ---
    top_idx = np.argsort(coef[i])[-top_n:]
    top_features = feature_names[top_idx][::-1]  # descending
    top_scores = coef[i][top_idx][::-1]

    # --- Bar chart ---
    plt.figure(figsize=(10,6))
    plt.barh(top_features, top_scores, color='skyblue')
    plt.xlabel("Coefficient weight")
    plt.title(f"Top {top_n} features for class '{class_label}'")
    plt.tight_layout()
    plt.savefig(f"visuals/top_features_{class_label}.png")
    plt.show()

    # --- Word Cloud ---
    word_scores = {feature: score for feature, score in zip(top_features, top_scores)}
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(word_scores)
    
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {class_label}")
    plt.tight_layout()
    plt.savefig(f"visuals/wordcloud_{class_label}.png")
    plt.show()
