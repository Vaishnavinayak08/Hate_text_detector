import pandas as pd
import re

def clean_text(text):
    """Function to clean text data"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = text.strip()
    return text

def load_and_preprocess_data(file_path):
    """Load and preprocess dataset"""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Clean the text data
    df['text'] = df['text'].apply(clean_text)
    
    return df
