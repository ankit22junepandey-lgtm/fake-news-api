from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("news_models.pkl")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    prediction = model.predict([item.text])
    return {"prediction": "Real" if prediction[0] == 1 else "Fake"}
