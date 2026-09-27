from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

# Load models
bundle = joblib.load("news_models.pkl")
vectorizer = bundle["vectorizer"]
LR = bundle["LR"]
DT = bundle["DT"]
GB = bundle["GB"]
RF = bundle["RF"]

app = FastAPI(title="Fake News Detection API")

class NewsInput(BaseModel):
    text: str

# Must match the cleaning used during training
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

def output_label(n):
    return "Real News" if n == 1 else "Fake News"

@app.get("/")
def home():
    return {"message": "Welcome to Fake News Detection API! Use /predict to test."}

@app.post("/predict/")
def predict(news: NewsInput):
    cleaned = simple_clean(news.text)          # ← the fix
    new_xv_test = vectorizer.transform([cleaned])

    pred_LR = LR.predict(new_xv_test)[0]
    pred_DT = DT.predict(new_xv_test)[0]
    pred_GB = GB.predict(new_xv_test)[0]
    pred_RF = RF.predict(new_xv_test)[0]

    return {
        "Logistic Regression": output_label(pred_LR),
        "Decision Tree": output_label(pred_DT),
        "Gradient Boosting": output_label(pred_GB),
        "Random Forest": output_label(pred_RF)
    }
