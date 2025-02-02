import pickle
import pandas as pd

# Load the model, vectorizer, and label encoder
with open(r'D:\hate_detect\model\hate_text_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open(r'D:\hate_detect\model\vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
with open(r'D:\hate_detect\model\label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Function to preprocess text (same as in main.py)
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Example texts to test
texts = [
    "The administration is very supportive and helpful.",
    "The food in the canteen is terrible and unhygienic.",
    "I love the campus environment, it's very peaceful.",
    "The professors are rude and unapproachable."
]

# Preprocess and vectorize the texts
texts_preprocessed = [preprocess_text(text) for text in texts]
texts_vec = vectorizer.transform(texts_preprocessed)

# Predict using the loaded model
predictions = model.predict(texts_vec)
predictions_decoded = label_encoder.inverse_transform(predictions)

# Print the predictions
for text, prediction in zip(texts, predictions_decoded):
    print(f"Text: {text}\nPrediction: {prediction}\n")
