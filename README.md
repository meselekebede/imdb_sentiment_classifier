# 🎬 IMDb Sentiment Classifier

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%23000.svg?logo=flask&logoColor=white)

A production-grade **text classification pipeline** using **scikit-learn** that analyzes movie reviews with 88% accuracy. Built with modern NLP preprocessing and deployed via both CLI and web interfaces.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧹 **Deep Text Cleaning** | HTML tag removal, lemmatization, stopword filtering, and URL/email sanitization |
| 📊 **TF-IDF Vectorization** | 5,000 feature vectorization with sublinear term frequency scaling |
| 🤖 **ML Model** | Logistic Regression with L2 regularization (C=1.0) |
| 🌐 **Dual Interface** | Flask web UI + Python CLI prediction tool |
| 💾 **Model Persistence** | Serialized model/vectorizer with joblib |

---

## 📂 Project Structure

```
imdb_sentiment_classifier/
├── 📄 app.py                 # Flask web server
├── 📄 predict.py             # CLI prediction tool
├── 📄 train.py               # Model training pipeline
├── 📄 utils.py               # Text processing utilities
├── 📄 requirements.txt       # Python dependencies
├── 📂 models/                # Serialized artifacts
│   ├── model.pkl            # Trained classifier
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── 📂 static/                # Web assets
│   └── style.css            # Modern UI styling
├── 📂 templates/             # HTML templates
│   └── index.html           # Responsive web interface
└── 📂 data/                  # Dataset storage
    └── IMDB_Dataset.csv     # 50K reviews dataset
```
# 🚀 Quick Start
clone this repo
```
git clone https://github.com/meselekebede/imdb_sentiment_classifier.git
```
Install Dependencies
```
pip install -r requirements.txt 
python -m nltk.downloader stopwords wordnet  # Required NLTK resources
```
# 📥 Dataset Setup
Download the dataset from:
[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews )\
Extract the ZIP file
Place the CSV file inside the data/ folder:
```
data/ 
└── IMDB_Dataset.csv
```
# 🏋️ Train the Model
To train the model (uses a subset of 5,000 samples by default):
```
python train.py
```
```
Example output: 
Model Accuracy: 0.91 
Model and vectorizer saved.
```
# 🔮 Predict Sentiment via CLI
Use the command-line tool to classify any review:
```
python predict.py "I absolutely loved this movie!"
```

# 💻 Run the Flask Web App

Launch the web interface: 
```
python app.py
```
Then open your browser and go to:

👉 http://localhost:5000 "this may be different based on your machine so you can copy and past from your terminal"

You’ll see a clean, modern UI where you can:

Enter a movie review \
Get real-time sentiment analysis result \
See confidence score in a styled box

# 🛠️ Code Overview
```
utils.py
```

Contains reusable function for deep text preprocessing:

1. Lowercasing 
2. Removing HTML tags, URLs, emails 
3. Removing punctuation and digits 
4. Tokenizing, removing stopwords, lemmatizing

```
train.py
```
Trains the model:

1. Loads and preprocesses the dataset 
2. Splits into train/test sets 
3. Applies TF-IDF transformation 
4. Trains and saves the model + vectorizer

```
predict.py
```
CLI tool:

1. Loads the saved model and vectorizer 
2. Accepts input text 
3. Outputs prediction and confidence

```
app.py
```

Flask web application:

1. Renders HTML form 
2. Handles POST requests 
3. Returns styled prediction results 
