import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Print working directory
print("Current working directory:", os.getcwd())

# Path to data
data_path = os.path.join(os.getcwd(), "data", "movies.csv")
print("Loading dataset from:", data_path)

# Load dataset
df = pd.read_csv(data_path, encoding="latin1")

# Show dataset info
print("Columns in dataset:", df.columns.tolist())
print("Number of rows:", len(df))

# ----- Feature Engineering -----
# Combine Genre + Director + Actor1 + Actor2 + Actor3
df["features"] = (
    df["Genre"].astype(str) + " "
    + df["Director"].astype(str) + " "
    + df["Actor 1"].astype(str) + " "
    + df["Actor 2"].astype(str) + " "
    + df["Actor 3"].astype(str)
)

# ----- Define Features and Target -----
X = df["features"]
y = df["Rating"]

# Clean target column (convert to numeric, drop missing)
y = pd.to_numeric(y, errors="coerce")
mask = y.notnull()
X = X[mask]
y = y[mask]

# ----- Build Pipeline -----
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("regressor", LinearRegression())
])

# Train model
print("Training model...")
pipeline.fit(X, y)

# ----- Save Model -----
model_path = os.path.join(os.getcwd(), "src", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Training complete. Model saved at:", model_path)
