import os
import pickle

# Path to model
model_path = os.path.join(os.getcwd(), "src", "model.pkl")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully.")

# ----- Example Movie Input -----
movie_features = "Action Christopher Nolan Christian Bale Heath Ledger Michael Caine"

# Predict rating
predicted_rating = model.predict([movie_features])[0]

print(f"🎬 Predicted Rating for the movie: {predicted_rating:.2f}")
