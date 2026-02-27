import pandas as pd
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="hinge")

nltk.download('stopwords')

# Load datasets
true_news = pd.read_csv("data/True.csv", encoding="latin1")
fake_news = pd.read_csv("data/Fake.csv", encoding="latin1")

# Balance dataset
min_count = min(true_news.shape[0], fake_news.shape[0])

true_news = true_news.sample(min_count, random_state=42)
fake_news = fake_news.sample(min_count, random_state=42)

# Add labels
true_news["label"] = "REAL"
fake_news["label"] = "FAKE"

# Combine datasets
data = pd.concat([true_news, fake_news], axis=0)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Some datasets use "title + text", some only "text"
# If title exists, combine it with text
if "title" in data.columns:
    data["content"] = data["title"] + " " + data["text"]
else:
    data["content"] = data["text"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["content"], data["label"], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)

# Model
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="hinge")
model.fit(X_train_vec, y_train)

# Save model
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully")

X_test_vec = vectorizer.transform(X_test)
pred = model.predict(X_test_vec)

print("Model Accuracy:", accuracy_score(y_test, pred))