import joblib
import requests

# ==============================
# LOAD TRAINED ML MODEL
# ==============================

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# ==============================
# ML PREDICTION FUNCTION
# ==============================

def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


# ==============================
# LLM ANALYSIS (Hugging Face API)
# Uses free open model (no approval needed)
# ==============================

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# ⚠️ Replace with your Hugging Face token
headers = {
    "Authorization": "Bearer YOUR_HF_ACCESS_TOKEN_HERE"
}


def query_llm(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# ==============================
# GENERATE CREDIBILITY EXPLANATION
# ==============================

def credibility_analysis(text):
    prompt = f"""
Check if this news seems credible or suspicious and explain why:

{text}
"""
    result = query_llm(prompt)

    try:
        return result[0]["generated_text"]
    except:
        return "LLM analysis unavailable."


# ==============================
# GENERATE SUMMARY
# ==============================

def summarize_news(text):
    prompt = f"Summarize this news article:\n{text}"
    result = query_llm(prompt)

    try:
        return result[0]["generated_text"]
    except:
        return "Summary unavailable."


# ==============================
# MAIN PROGRAM
# ==============================

if __name__ == "__main__":

    print("\nPaste News Article:\n")
    article = input()

    # ML Prediction
    prediction = predict_news(article)

    print("\n==========================")
    print("Fake News Prediction:", prediction)

    # AI Credibility Check
    print("\nCredibility Analysis:")
    print(credibility_analysis(article))

    # Summary
    print("\nSummary:")
    print(summarize_news(article))

    print("\n==========================")