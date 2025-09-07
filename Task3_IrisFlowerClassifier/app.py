"""
A minimal Streamlit app for interactive Iris predictions.

Run:
    streamlit run app.py
"""

import pathlib
import numpy as np
import streamlit as st
from joblib import load

HERE = pathlib.Path(__file__).parent.resolve()
MODEL_FILE = HERE / "iris_model.joblib"

@st.cache_resource
def load_bundle():
    bundle = load(MODEL_FILE)
    return bundle["pipeline"], bundle["target_map"], bundle["feature_order"]

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Species Classifier")
st.caption("Enter measurements in centimeters.")

try:
    pipeline, target_map, feat_order = load_bundle()
except Exception as e:
    st.error("Model not found. Please run `python train.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sl = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    pl = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
with col2:
    sw = st.number_input("Sepal width (cm)",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    pw = st.number_input("Petal width (cm)",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict"):
    X = np.array([[sl, sw, pl, pw]], dtype=float)
    pred_idx = int(pipeline.predict(X)[0])
    label = target_map[pred_idx]
    st.success(f"Predicted species: **{label}**")

    proba = getattr(pipeline, "predict_proba", None)
    if callable(proba):
        probs = proba(X)[0]
        st.subheader("Class probabilities")
        for i, p in enumerate(probs):
            st.write(f"- {target_map[i]}: {p:.3f}")
