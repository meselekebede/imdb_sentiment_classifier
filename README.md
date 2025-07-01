# IMDb Sentiment Classifier

A complete **text classification pipeline** using **scikit-learn**, trained on the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews ), that classifies movie reviews as **positive** or **negative**.

🔧 Features:
- **Deep text preprocessing**: lowercasing, HTML tag removal, lemmatization, stopword removal
- TF-IDF vectorization
- Logistic Regression classifier
- Command-line prediction tool (`predict.py`)
- Visually appealing Flask web app with HTML/CSS frontend
- Model persistence (saved as `.pkl` files)

---

## 📁 Project Structure
imdb_sentiment_classifier/

│
├── templates/

│ └── index.html # Web UI template

│
├── static/

│ └── style.css # Styling for the UI

│

├── models/

│ ├── model.pkl # Trained ML model

│ └── vectorizer.pkl # TF-IDF Vectorizer

│

├── data/

│ └── IMDB_Dataset.csv # Kaggle dataset

│

├── utils.py # Text preprocessing utilities

├── train.py # Train the model

├── predict.py # CLI prediction script

├── app.py # Flask web app

├── requirements.txt # Python dependencies

└── README.md # You are here


---

## 🧰 Requirements

Install dependencies:

bash \
pip install -r requirements.txt

Required packages:

scikit-learn\
pandas\
joblib\
flask\
nltk


✅ Make sure to run this once to download NLTK resources:\
bash \
python -m nltk.downloader stopwords wordnet

📥 Dataset Setup
Download the dataset from:
[IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews )\
Extract the ZIP file
Place the CSV file inside the data/ folder:\
data/ \
└── IMDB_Dataset.csv

# 🏋️ Train the Model
To train the model (uses a subset of 5,000 samples by default):\
bash \
python train.py

This will:

Load and preprocess the dataset \
Apply deep text cleaning (lowercasing, lemmatization, etc.) \
Train a Logistic Regression classifier using TF-IDF features \
Save the trained model and vectorizer in the models/ directory \
 

Example output: \
Model Accuracy: 0.91 \
Model and vectorizer saved.

# 🔮 Predict Sentiment via CLI
Use the command-line tool to classify any review:

python predict.py "I absolutely loved this movie!"

# 💻 Run the Flask Web App

Launch the web interface: \
python app.py

Then open your browser and go to:

👉 http://localhost:5000

You’ll see a clean, modern UI where you can:

Enter a movie review \
Get real-time sentiment analysis result \
See confidence score in a styled box

# 🛠️ Code Overview

utils.py 

Contains reusable function for deep text preprocessing:

Lowercasing \
Removing HTML tags, URLs, emails \
Removing punctuation and digits \
Tokenizing, removing stopwords, lemmatizing


train.py \
Trains the model:

Loads and preprocesses the dataset \
Splits into train/test sets \
Applies TF-IDF transformation \
Trains and saves the model + vectorizer


predict.py \
CLI tool:

Loads the saved model and vectorizer \
Accepts input text \
Outputs prediction and confidence


app.py \
Flask web application:

Renders HTML form \
Handles POST requests \
Returns styled prediction results 
