import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Restaurant Review Dashboard", layout="wide")

st.title("🍽️ AI Restaurant Review Analytics Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(synthetic_restaurant_reviews_500.csv)

# -----------------------------
# SENTIMENT FUNCTION
# -----------------------------
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Review"].apply(get_sentiment)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filter Data")

city_filter = st.sidebar.multiselect(
    "Select City",
    options=df["City"].unique(),
    default=df["City"].unique()
)

cuisine_filter = st.sidebar.multiselect(
    "Select Cuisine",
    options=df["Cuisine"].unique(),
    default=df["Cuisine"].unique()
)

filtered_df = df[
    (df["City"].isin(city_filter)) &
    (df["Cuisine"].isin(cuisine_filter))
]

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Restaurants", len(filtered_df))
col2.metric("Average Rating", round(filtered_df["Rating"].mean(), 2))
col3.metric("Total Votes", filtered_df["Votes"].sum())

# -----------------------------
# SENTIMENT PIE CHART
# -----------------------------
st.subheader("Customer Sentiment Distribution")

sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig1 = px.pie(sentiment_counts, names="Sentiment", values="Count")
st.plotly_chart(fig1)

# -----------------------------
# CITY VS RATING
# -----------------------------
st.subheader("Average Rating by City")

city_rating = filtered_df.groupby("City")["Rating"].mean().reset_index()
fig2 = px.bar(city_rating, x="City", y="Rating")
st.plotly_chart(fig2)

# -----------------------------
# CUISINE POPULARITY
# -----------------------------
st.subheader("Cuisine Popularity")

cuisine_votes = filtered_df.groupby("Cuisine")["Votes"].sum().reset_index()
fig3 = px.bar(cuisine_votes, x="Cuisine", y="Votes")
st.plotly_chart(fig3)

# -----------------------------
# WORD CLOUD
# -----------------------------
st.subheader("Word Cloud of Reviews")

text = " ".join(filtered_df["Review"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# -----------------------------
# SHOW DATA
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df)
