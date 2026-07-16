import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Combine subject and topic
df["text"] = (
    df["Subject"] + " " +
    df["Topic"] + " " +
    df["Level"]
)

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = NearestNeighbors(n_neighbors=5, metric="cosine")
model.fit(X)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")