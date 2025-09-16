import gradio as gr
import joblib
import os  # To get the port from environment

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

# Launch on 0.0.0.0 and the port provided by Render
port = int(os.environ.get("PORT", 7860))  # 7860 is default Gradio port locally
iface.launch(server_name="0.0.0.0", server_port=port)
