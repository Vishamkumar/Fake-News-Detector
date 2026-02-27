import joblib

# Load trained model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def detect_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = model.decision_function(vector)[0]

    credibility_score = round((abs(confidence) / 2) * 100, 2)

    return prediction, credibility_score

if __name__ == "__main__":
    article = input("Paste News Article:\n")

    result, score = detect_news(article)

    print("\nNews Status:", result)
    print("Credibility Score:", score, "%")