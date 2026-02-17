import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(page_title="VoiceOfDine", layout="wide")

st.title("🍽️ VoiceOfDine")
st.subheader("Restaurant Review Intelligence System")
st.caption("Turning customer feedback into actionable business insights")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("synthetic_restaurant_reviews_500.csv")

# -------------------------
# SENTIMENT FUNCTION (NLP)
# -------------------------
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Review"].apply(get_sentiment)

# -------------------------
# RESTAURANT LOGIN (SELECTOR)
# -------------------------
st.sidebar.title("🔐 Restaurant Login")

restaurant_list = df["Restaurant_Name"].unique()

selected_restaurant = st.sidebar.selectbox(
    "Select Your Restaurant",
    restaurant_list
)

restaurant_df = df[df["Restaurant_Name"] == selected_restaurant]

# -------------------------
# PERFORMANCE SUMMARY
# -------------------------
st.subheader(f"📊 Performance Dashboard: {selected_restaurant}")

col1, col2, col3 = st.columns(3)

col1.metric("Average Rating", round(restaurant_df["Rating"].mean(), 2))
col2.metric("Total Votes", int(restaurant_df["Votes"].sum()))
col3.metric("Total Reviews", len(restaurant_df))

# -------------------------
# BUSINESS HEALTH SCORE
# -------------------------
avg_rating = restaurant_df["Rating"].mean()

if avg_rating >= 4:
    health = "🟢 Excellent"
elif avg_rating >= 3:
    health = "🟡 Needs Improvement"
else:
    health = "🔴 At Risk"

st.subheader("🏥 Business Health Status")
st.write(health)

# -------------------------
# SENTIMENT PIE CHART
# -------------------------
st.subheader("😊 Customer Sentiment Analysis")

sentiment_counts = restaurant_df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig1 = px.pie(sentiment_counts, names="Sentiment", values="Count")
st.plotly_chart(fig1)

# -------------------------
# WORD CLOUD (CUSTOMER VOICE)
# -------------------------
st.subheader("🗣️ Customer Voice Word Cloud")

text = " ".join(restaurant_df["Review"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# -------------------------
# COMMON COMPLAINTS (NLP INSIGHT)
# -------------------------
st.subheader("⚠️ Customer Complaints Detected")

negative_reviews = restaurant_df[restaurant_df["Sentiment"] == "Negative"]

if len(negative_reviews) > 0:
    st.write(negative_reviews["Review"].head(5))
else:
    st.success("No major complaints detected!")

# -------------------------
# STRENGTH DETECTION
# -------------------------
st.subheader("💪 What Customers Like")

positive_reviews = restaurant_df[restaurant_df["Sentiment"] == "Positive"]

if len(positive_reviews) > 0:
    st.write(positive_reviews["Review"].head(5))
else:
    st.info("Not enough positive feedback yet.")

# -------------------------
# COST VS RATING INSIGHT
# -------------------------
st.subheader("💰 Pricing vs Rating Insight")

fig2 = px.scatter(
    restaurant_df,
    x="Cost_for_Two",
    y="Rating",
    size="Votes",
    title="Does Pricing Affect Ratings?"
)
st.plotly_chart(fig2)

# -------------------------
# AI BUSINESS SUGGESTIONS
# -------------------------
st.subheader("🤖 AI Improvement Suggestions")

reviews_text = " ".join(restaurant_df["Review"]).lower()

if avg_rating < 3:
    st.warning("Your rating is low. Focus on improving food quality and service.")

if "slow" in reviews_text:
    st.info("Customers mention slow service. Improve service speed.")

if "expensive" in reviews_text:
    st.info("Customers feel pricing is high. Review your pricing strategy.")

if "rude" in reviews_text:
    st.info("Train staff to improve customer interaction.")

if "amazing" in reviews_text or "delicious" in reviews_text:
    st.success("Customers love your food quality. Maintain consistency!")

# -------------------------
# SHOW DATA
# -------------------------
st.subheader("📄 Review Data")
st.dataframe(restaurant_df)
