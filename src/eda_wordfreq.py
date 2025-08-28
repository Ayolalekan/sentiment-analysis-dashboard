import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_reviews.csv")

# Function to get top words for a given sentiment
def plot_top_words(sentiment_label, top_n=20):
    subset = df[df['sentiment'] == sentiment_label]
    
    # Keep only strings
    texts = subset['clean_text'].dropna().astype(str)
    
    all_words = " ".join(texts).split()
    from collections import Counter
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)

    # Plot
    words, counts = zip(*top_words)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.barh(words[::-1], counts[::-1], color='skyblue')
    plt.title(f"Top {top_n} words for {sentiment_label} reviews")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"visuals/top_words_{sentiment_label}.png")
    plt.show()


# Example usage
for sentiment in ['positive', 'neutral', 'negative']:
    plot_top_words(sentiment)
