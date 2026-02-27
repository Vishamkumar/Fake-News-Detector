import streamlit as st
import joblib
from transformers import pipeline

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 AI Fake News Detector for Students")
st.write("Check if a news article is REAL or FAKE + AI explanation and summary.")

# ==============================
# LOAD ML MODEL
# ==============================

@st.cache_resource
def load_ml_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_ml_model()

# ==============================
# LOAD LOCAL LLM (NO TOKEN NEEDED)
# ==============================

@st.cache_resource
def load_llm():
    generator = pipeline(
        "text-generation",          # FIXED pipeline type
        model="google/flan-t5-small",
        max_new_tokens=150
    )
    return generator

generator = load_llm()

# ==============================
# FUNCTIONS
# ==============================

def predict_news(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]


def credibility_analysis(text):
    prompt = f"Check if this news is credible and explain why:\n{text}"
    result = generator(prompt)
    return result[0]["generated_text"]


def summarize_news(text):
    prompt = f"Summarize this news article:\n{text}"
    result = generator(prompt)
    return result[0]["generated_text"]


# ==============================
# USER INPUT UI
# ==============================

article = st.text_area("Paste News Article Here", height=200)

if st.button("Analyze News"):

    if article.strip() == "":
        st.warning("Please paste a news article.")
    else:
        with st.spinner("Analyzing news..."):

            # ML Prediction
            prediction = predict_news(article)

            # AI Analysis
            explanation = credibility_analysis(article)
            summary = summarize_news(article)

        # ==============================
        # RESULTS DISPLAY
        # ==============================

        st.subheader("Prediction Result")

        if prediction == "FAKE":
            st.error("🚨 This news is predicted as FAKE")
        else:
            st.success("✅ This news is predicted as REAL")

        st.subheader("AI Credibility Analysis")
        st.write(explanation)

        st.subheader("News Summary")
        st.write(summary)