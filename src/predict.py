import pickle

# Load the model, vectorizer, and label encoder
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('models/label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Function to predict sentiment
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])  # Convert input text to feature vector
    prediction = model.predict(text_vec)  # Predict sentiment (0, 1, or 2)
    
    # Debugging: Print the raw prediction
    print("Raw prediction:", prediction)
    
    try:
        # Return the decoded label
        return label_encoder.inverse_transform(prediction)[0]
    except ValueError as e:
        # Handle cases where the model predicts an unseen label
        print("Prediction error:", e)
        return "Invalid prediction"

# Test the prediction function
new_text = "i love this product"  # Example input text
print(f"Sentiment: {predict_sentiment(new_text)}")
