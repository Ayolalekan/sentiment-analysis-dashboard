import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -----------------------
# Load data
# -----------------------
df_aspects = pd.read_csv("data/aspect_level_sentiment.csv")

# Example top words and features (replace with your actual values or load from CSV)
top_words_positive = ['love', 'great', 'easy', 'perfect', 'excellent']
top_words_neutral = ['ok', 'decent', 'rate', 'sure', 'good']
top_words_negative = ['returned', 'slow', 'disappointed', 'waste', 'fault']

top_features_positive = ['love', 'great', 'easy', 'perfect', 'excellent']
top_features_negative = ['returned', 'returning', 'didnt', 'slow', 'disappointed']
feature_importance_pos = [5.3886, 5.0993, 3.8161, 3.6190, 3.2152]
feature_importance_neg = [5.1441, 4.2791, 4.2720, 4.2560, 3.9533]

# -----------------------
# Page title
# -----------------------
st.title("📊 Amazon Product Reviews Sentiment Dashboard")
st.markdown("Explore overall and aspect-level sentiment insights interactively.")

# -----------------------
# Overall sentiment
# -----------------------
st.subheader("Overall Sentiment Distribution")
sentiment_counts = df_aspects['aspect_sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# -----------------------
# Aspect-level sentiment
# -----------------------
st.subheader("Aspect-Level Sentiment Distribution")

agg = df_aspects.groupby(['aspect', 'aspect_sentiment']).size().unstack(fill_value=0)
agg_pct = agg.div(agg.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12,6))
agg_pct.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)
plt.ylabel('Percentage of reviews')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# -----------------------
# Filter and view reviews by aspect
# -----------------------
st.subheader("🔍 Explore Reviews by Aspect")
selected_aspect = st.selectbox("Choose an aspect", df_aspects['aspect'].unique())
aspect_reviews = df_aspects[df_aspects['aspect'] == selected_aspect]
st.write(aspect_reviews[['review_index','aspect_sentiment']].head(10))

# -----------------------
# Word clouds per sentiment
# -----------------------
st.subheader("🌐 Word Clouds per Sentiment")
for sentiment in df_aspects['aspect_sentiment'].unique():
    st.markdown(f"**{sentiment.upper()}**")
    text = " ".join(df_aspects[df_aspects['aspect_sentiment']==sentiment]['aspect'].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10,4))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

# -----------------------
# Top Words per Sentiment
# -----------------------
st.subheader("🔥 Top Words per Sentiment")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Positive**")
    st.write(", ".join(top_words_positive))
with col2:
    st.markdown("**Neutral**")
    st.write(", ".join(top_words_neutral))
with col3:
    st.markdown("**Negative**")
    st.write(", ".join(top_words_negative))

# -----------------------
# Top Features per Sentiment
# -----------------------
st.subheader("💡 Top Features Driving Sentiment")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Positive Features**")
    st.write(", ".join(top_features_positive))
    fig_pos, ax_pos = plt.subplots()
    ax_pos.bar(top_features_positive, feature_importance_pos, color='green')
    plt.xticks(rotation=45)
    plt.title("Top Features - Positive")
    st.pyplot(fig_pos)
with col2:
    st.markdown("**Negative Features**")
    st.write(", ".join(top_features_negative))
    fig_neg, ax_neg = plt.subplots()
    ax_neg.bar(top_features_negative, feature_importance_neg, color='red')
    plt.xticks(rotation=45)
    plt.title("Top Features - Negative")
    st.pyplot(fig_neg)
