# Fake News Detector with AI Summarization

## Overview
This project is an AI-powered **Fake News Detection and Summarization System** designed for students and general users.  
It analyzes news articles, predicts whether they are **REAL** or **FAKE**, and generates concise summaries.  
Additionally, the system integrates **Machine Learning (ML)** with **Large Language Models (LLMs)** like Meta Llama for credibility analysis.

---

## Features
- Predicts fake or real news using a **Passive Aggressive Classifier** trained on labeled datasets.  
- Generates **summaries** of articles for faster understanding.  
- Integrates **Meta Llama / FLAN-T5** LLM for AI-based credibility scoring.  
- **Web interface** built with **Streamlit** for easy interaction.  
- Can be deployed on **Hugging Face Spaces** for web access.

---

## System Approach
1. **Data Collection:**  
   - News datasets were collected from [Kaggle](https://www.kaggle.com) and categorized into `True` and `Fake` news CSV files.
2. **Data Preprocessing:**  
   - Combined `title` and `text` columns.  
   - Cleaned and shuffled the data for training.
3. **Model Training:**  
   - Machine Learning model trained with `TfidfVectorizer` + `PassiveAggressiveClassifier`.  
   - Model accuracy achieved ~99.4% on the test set.
4. **LLM Integration:**  
   - Meta Llama / FLAN-T5 is used for AI-based summarization and credibility scoring.
5. **Web App:**  
   - Streamlit app allows users to input news and get predictions + summaries in real-time.

---

## System Requirements
- **Python 3.10+**  
- **RAM:** 8 GB+ (recommended for LLM inference)  
- **Disk Space:** 10 GB+  
- **Operating System:** Windows / Linux / MacOS

---

## Libraries Used
- **pandas**: For data loading, cleaning, and manipulation.  
- **numpy**: For numerical operations on datasets.  
- **scikit-learn**: For ML models and preprocessing (`TfidfVectorizer`, `PassiveAggressiveClassifier`).  
- **nltk**: For text tokenization and stopwords removal.  
- **transformers**: For loading LLMs from Hugging Face (Meta Llama, FLAN-T5).  
- **streamlit**: To build a user-friendly web interface.  
- **joblib**: To save and load ML models efficiently.  

---

## Project Files
- `train_model.py` → Train the ML model on True / Fake dataset.  
- `predict_and_analyze.py` → Make predictions and run LLM summarization.  
- `summarizer.py` → Standalone summarization script.  
- `app.py` → Streamlit app for web interface.  
- `True.csv` / `Fake.csv` → Dataset files.  

---

## Deployment
1. **Local Deployment (VS Code / Terminal)**  
   ```bash
   python -m streamlit run app.py
