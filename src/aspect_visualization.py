import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Step 1: Load aspect-level sentiment
# ----------------------
df = pd.read_csv("data/aspect_level_sentiment.csv")

# ----------------------
# Step 2: Aggregate sentiment counts per aspect
# ----------------------
agg = df.groupby(['aspect', 'aspect_sentiment']).size().unstack(fill_value=0)

# ----------------------
# Step 3: Calculate percentages (optional)
# ----------------------
agg_pct = agg.div(agg.sum(axis=1), axis=0) * 100

# ----------------------
# Step 4: Plot stacked bar chart
# ----------------------
plt.figure(figsize=(12,6))
agg_pct.plot(kind='bar', stacked=True, colormap='Set2')
plt.ylabel('Percentage of reviews')
plt.title('Aspect-Level Sentiment Distribution')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()
