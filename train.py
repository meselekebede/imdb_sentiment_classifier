# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import preprocess_text
import joblib
import os

DATA_PATH = "data/IMDB_Dataset.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
SAMPLE_SIZE = 5000  # we can change this if needed

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(SAMPLE_SIZE)

    # Apply preprocessing
    df['review'] = df['review'].apply(preprocess_text)

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df['review'], df['sentiment']

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train()
