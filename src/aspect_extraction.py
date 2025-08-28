import pandas as pd
from joblib import load

# ------------------------------
# Step 1: Load cleaned dataset
# ------------------------------
df = pd.read_csv("data/cleaned_reviews.csv")

# ------------------------------
# Step 2: Define aspects to track
# ------------------------------
aspects = ['delivery', 'packaging', 'price', 'quality', 'battery', 'design', 'performance', 'support']

# ------------------------------
# Step 3: Detect aspects in each review
# ------------------------------
def extract_aspects(review):
    if not isinstance(review, str):
        return []  # no aspects if review is missing
    review_words = review.split()
    found = [aspect for aspect in aspects if aspect in review_words]
    return found


df['aspects_found'] = df['clean_text'].apply(extract_aspects)

# Keep only reviews that mention at least one aspect
df_aspect = df[df['aspects_found'].str.len() > 0]

# Preview
print(df_aspect[['clean_text', 'aspects_found', 'sentiment']].head(10))

# ------------------------------
# Optional Step 4: Save for next phase
# ------------------------------
df_aspect.to_csv("data/reviews_with_aspects.csv", index=False)
