import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_reviews.csv")

# Number of example reviews per sentiment
n_examples = 5

for sentiment in ['positive', 'neutral', 'negative']:
    print(f"\n--- {sentiment.upper()} REVIEWS ---\n")
    subset = df[df['sentiment'] == sentiment]['clean_text']
    examples = subset.sample(n=n_examples, random_state=42)  # random sample
    for i, review in enumerate(examples, 1):
        print(f"{i}. {review}\n")
