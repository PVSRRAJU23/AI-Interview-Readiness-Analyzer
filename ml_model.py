import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
data = pd.read_csv("data/interview_data.csv")

X = data["resume_skills"]
y = data["readiness"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save files
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(encoder, open("models/encoder.pkl", "wb"))

print("Model trained and saved")
