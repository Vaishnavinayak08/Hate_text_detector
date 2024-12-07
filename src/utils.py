from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

def save_model(model, filename):
    """Save the trained model"""
    joblib.dump(model, filename)

def load_model(filename):
    """Load a saved model"""
    return joblib.load(filename)

def save_vectorizer(vectorizer, filename):
    """Save the vectorizer"""
    joblib.dump(vectorizer, filename)

def save_label_encoder(label_encoder, filename):
    """Save the label encoder"""
    joblib.dump(label_encoder, filename)

def load_vectorizer(filename):
    """Load the vectorizer"""
    return joblib.load(filename)

def load_label_encoder(filename):
    """Load the label encoder"""
    return joblib.load(filename)

def vectorize_text(vectorizer, text_data):
    """Convert text data into vectors using the vectorizer"""
    return vectorizer.transform(text_data)

def encode_labels(label_encoder, labels):
    """Encode labels using the label encoder"""
    return label_encoder.fit_transform(labels)
