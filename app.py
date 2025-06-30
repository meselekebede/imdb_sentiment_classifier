# app.py
from flask import Flask, request, render_template
from utils import preprocess_text
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vec_text = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(vec_text)[0]
    label = "positive" if model.predict(vec_text)[0] == 1 else "negative"
    confidence = round(max(proba), 4)
    return {"label": label, "confidence": str(confidence)}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()
    if not review:
        return render_template("index.html", result=None)
    result = predict_sentiment(review)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)