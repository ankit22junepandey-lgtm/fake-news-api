from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("news_models.pkl")

# Initialize FastAPI app
app = FastAPI()

# Input schema
class NewsInput(BaseModel):
    text: str

# Root endpoint (just to check if API is live)
@app.get("/")
def home():
    return {"message": "✅ Fake News API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict_news(payload: NewsInput):
    text = payload.text.strip()
    if not text:
        return {"error": "⚠️ Please enter some news text"}

    prediction = model.predict([text])[0]
    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    return {"prediction": result}
