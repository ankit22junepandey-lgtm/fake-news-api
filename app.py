from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("news_models.pkl")

# Create app
app = FastAPI()

# Input format
class NewsInput(BaseModel):
    text: str

# Prediction route
@app.post("/predict")
def predict_news(data: NewsInput):
    if not data.text.strip():
        return {"error": "⚠️ Please enter some news text"}

    prediction = model.predict([data.text])
    result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
    return {"prediction": result}
