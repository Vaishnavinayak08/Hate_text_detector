import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv(r'D:\hate_detect\data\institution_feedback_dataset.csv')  # Corrected path

# Preprocessing
X = df['text']  # Replace with your actual text column
y = df['label']  # Replace with your actual label column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train a machine learning model(Logical Regression model)
model = LogisticRegression()
model.fit(X_train_vec, y_train_encoded)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))

# Save the model and vectorizer
with open(r'D:\hate_detect\model\hate_text_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open(r'D:\hate_detect\model\vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open(r'D:\hate_detect\model\label_encoder.pkl', 'wb') as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)
