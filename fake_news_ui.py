import streamlit as st
import pandas as pd
import joblib
import string
import requests
from xgboost import XGBClassifier

stop_words = set("""a about above after again against all am an and any are aren't as at be because been before being below between both 
but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had 
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd 
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once 
only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such 
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this 
those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's 
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours 
yourself yourselves""".split())

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0].max()
    return pred, proba

model = joblib.load("fake_news_model_fixed.pkl")
vectorizer = joblib.load("tfidf_vectorizer_fixed.pkl")

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Check", "📁 Batch Upload", "🧼 News Cleaner", "🛰️ Live News"])

with tab1:
    st.subheader("Enter a news article to check:")
    user_input = st.text_area("News text")
    if st.button("Check"):
        if not user_input.strip():
            st.warning("Please enter some news text.")
        else:
            label, confidence = predict_news(user_input)
            label_text = "✅ Real News" if label == 1 else "🚫 Fake News"
            st.success(f"{label_text} (Confidence: {confidence*100:.2f}%)")

with tab2:
    st.subheader("Upload a CSV file with news to analyze")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            df["clean_text"] = df["text"].apply(clean_text)
            X = vectorizer.transform(df["clean_text"])
            df["Prediction"] = model.predict(X)
            df["Confidence"] = model.predict_proba(X).max(axis=1)
            df["Prediction"] = df["Prediction"].map({0: "Fake", 1: "Real"})
            st.dataframe(df[["text", "Prediction", "Confidence"]])

with tab3:
    st.subheader("Paste raw news and clean it")
    raw_text = st.text_area("Raw news")
    if st.button("Clean Text"):
        cleaned = clean_text(raw_text)
        st.text_area("Cleaned text", cleaned, height=200)

with tab4:
    st.subheader("Live News Headlines from NewsAPI")
    api_key = st.text_input("Enter your NewsAPI Key", type="password")
    if api_key:
        try:
            res = requests.get(f"https://newsapi.org/v2/top-headlines?country=in&pageSize=5&apiKey={api_key}")
            data = res.json()
            if data["status"] == "ok":
                for article in data["articles"]:
                    headline = article["title"]
                    st.markdown(f"#### 🗞️ {headline}")
                    if st.button(f"Analyze: {headline}", key=headline):
                        label, confidence = predict_news(headline)
                        label_text = "✅ Real News" if label == 1 else "🚫 Fake News"
                        st.info(f"{label_text} (Confidence: {confidence*100:.2f}%)")
            else:
                st.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to fetch news: {e}")
    else:
        st.info("🔑 Enter API key above to begin.")
