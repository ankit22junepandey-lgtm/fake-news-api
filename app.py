from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

# Load the saved models + vectorizer
bundle = joblib.load("news_models.pkl")
vectorizer = bundle["vectorizer"]
LR = bundle["LR"]
DT = bundle["DT"]
GB = bundle["GB"]
RF = bundle["RF"]

# FastAPI app
app = FastAPI(title="Fake News Detection API")

# Input schema
class NewsInput(BaseModel):
    text: str

# Text cleaning — MUST match the cleaning used during training
def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Output label
def output_label(n):
    return "Real News" if n == 1 else "Fake News"

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Fake News Detection API",
        "usage": "POST to /predict/ with {\"text\": \"your news article here\"}"
    }

# Prediction endpoint — Logistic Regression only
@app.post("/predict/")
def predict(news: NewsInput):
    cleaned = simple_clean(news.text)
    new_xv_test = vectorizer.transform([cleaned])
    pred_LR = LR.predict(new_xv_test)[0]
    return {"prediction": output_label(pred_LR)}
