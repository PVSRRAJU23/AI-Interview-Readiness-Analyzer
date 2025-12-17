import pickle
from preprocess import clean_text

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))

resume = "python machine learning sql"
resume = clean_text(resume)

vec = vectorizer.transform([resume])
prediction = model.predict(vec)
result = encoder.inverse_transform(prediction)

print("Prediction:", result[0])