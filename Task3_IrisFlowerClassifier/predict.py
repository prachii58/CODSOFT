"""
Load the saved iris_model.joblib and predict from 4 numbers.
You can pass values via CLI or enter interactively if omitted.

Run:
    python predict.py --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
or:
    python predict.py
"""

import argparse
import sys
from joblib import load
import numpy as np
import pathlib

HERE = pathlib.Path(__file__).parent.resolve()
MODEL_FILE = HERE / "iris_model.joblib"

def load_model():
    if not MODEL_FILE.exists():
        sys.exit("Model not found. Run `python train.py` first.")
    bundle = load(MODEL_FILE)
    return bundle["pipeline"], bundle["target_map"], bundle["feature_order"]

def predict(sl, sw, pl, pw):
    pipeline, target_map, feat_order = load_model()
    # Ensure correct feature order
    X = np.array([[sl, sw, pl, pw]], dtype=float)
    # Some pipelines accept numpy array; if strictly requiring DataFrame you can adapt, but this works here.
    pred_idx = int(pipeline.predict(X)[0])
    # If the model outputs probabilities:
    proba = getattr(pipeline, "predict_proba", None)
    probs = proba(X)[0].tolist() if callable(proba) else None
    label = target_map[pred_idx]
    return label, probs, [target_map[i] for i in range(len(target_map))]

def main():
    p = argparse.ArgumentParser(description="Predict Iris species")
    p.add_argument("--sl", type=float, help="sepal length (cm)")
    p.add_argument("--sw", type=float, help="sepal width (cm)")
    p.add_argument("--pl", type=float, help="petal length (cm)")
    p.add_argument("--pw", type=float, help="petal width (cm)")
    args = p.parse_args()

    def ask(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Please enter a number, like 5.1")

    sl = args.sl if args.sl is not None else ask("Sepal length (cm): ")
    sw = args.sw if args.sw is not None else ask("Sepal width (cm):  ")
    pl = args.pl if args.pl is not None else ask("Petal length (cm): ")
    pw = args.pw if args.pw is not None else ask("Petal width (cm):  ")

    label, probs, class_order = predict(sl, sw, pl, pw)

    print(f"\n👉 Predicted species: {label}")
    if probs is not None:
        print("Class probabilities:")
        for name, pr in zip(class_order, probs):
            print(f"  - {name}: {pr:.3f}")

if __name__ == "__main__":
    main()
