# predict.py
from utils import preprocess_text
import sys
import joblib
import argparse

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def predict(text):
    cleaned_text = preprocess_text(text)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    vec_text = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(vec_text)[0]
    label = "positive" if model.predict(vec_text)[0] == 1 else "negative"
    confidence = round(max(proba), 4)
    return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of a review.")
    parser.add_argument("text", type=str, help="The review text to classify.")
    args = parser.parse_args()

    label, confidence = predict(args.text)
    print(f"Prediction: {label}, Confidence: {confidence}")