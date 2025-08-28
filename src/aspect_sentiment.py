import pandas as pd
from joblib import load
import ast

df = pd.read_csv("data/reviews_with_aspects.csv")
df['aspects_found'] = df['aspects_found'].apply(ast.literal_eval)  # <-- Fix here

model = load("models/logreg_tuned.joblib")
vectorizer = load("models/tfidf_vectorizer_tuned.joblib")

aspect_sentiments = []

for idx, row in df.iterrows():
    review_text = row['clean_text']
    sentiment_pred = model.predict(vectorizer.transform([review_text]))[0]
    
    for aspect in row['aspects_found']:  
        aspect_sentiments.append({
            'review_index': idx,
            'aspect': aspect,
            'aspect_sentiment': sentiment_pred
        })

df_aspect_sent = pd.DataFrame(aspect_sentiments)
df_aspect_sent.to_csv("data/aspect_level_sentiment.csv", index=False)
print("✅ Fixed: Aspect-level sentiment saved to data/aspect_level_sentiment.csv")
