import gradio as gr
import joblib

# Load model
model = joblib.load("news_models.pkl")

# Prediction function
def predict_news(text):
    if not text.strip():
        return "⚠️ Please enter some news text"
    prediction = model.predict([text])
    return "✅ Real News" if prediction[0] == 1 else "❌ Fake News"

# Gradio UI
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Paste news content here..."),
    outputs="text",
    title="📰 Fake News Detector",
    description="Enter news text to check if it's Real or Fake."
)

iface.launch()
