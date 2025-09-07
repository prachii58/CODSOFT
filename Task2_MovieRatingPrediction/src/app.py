import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Prediction App")
st.write("Fill in the movie details below and get the **predicted IMDb rating**!")

# Input fields
name = st.text_input("Movie Name")
year = st.number_input("Year of Release", min_value=1900, max_value=2100, step=1)
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=500, step=1)
genre = st.text_input("Genre (e.g. Action, Comedy, Drama)")
votes = st.number_input("Number of Votes", min_value=0, step=100)
director = st.text_input("Director Name")
actor1 = st.text_input("Lead Actor 1")
actor2 = st.text_input("Lead Actor 2")
actor3 = st.text_input("Lead Actor 3")

# Predict button
if st.button("🎥 Predict Rating"):
    # Create dataframe
    input_data = pd.DataFrame([{
        "Name": name,
        "Year": year,
        "Duration": duration,
        "Genre": genre,
        "Votes": votes,
        "Director": director,
        "Actor 1": actor1,
        "Actor 2": actor2,
        "Actor 3": actor3
    }])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"⭐ Predicted IMDb Rating: **{prediction:.2f}**")
import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Prediction App")
st.write("Fill in the movie details below and get the **predicted IMDb rating**!")

# Input fields
name = st.text_input("Movie Name")
year = st.number_input("Year of Release", min_value=1900, max_value=2100, step=1)
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=500, step=1)
genre = st.text_input("Genre (e.g. Action, Comedy, Drama)")
votes = st.number_input("Number of Votes", min_value=0, step=100)
director = st.text_input("Director Name")
actor1 = st.text_input("Lead Actor 1")
actor2 = st.text_input("Lead Actor 2")
actor3 = st.text_input("Lead Actor 3")

# Predict button
if st.button("🎥 Predict Rating"):
    # Create dataframe
    input_data = pd.DataFrame([{
        "Name": name,
        "Year": year,
        "Duration": duration,
        "Genre": genre,
        "Votes": votes,
        "Director": director,
        "Actor 1": actor1,
        "Actor 2": actor2,
        "Actor 3": actor3
    }])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"⭐ Predicted IMDb Rating: **{prediction:.2f}**")
