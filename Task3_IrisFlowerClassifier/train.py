"""
Train a robust Iris classifier with simple model selection.
- Uses your local Iris.csv if present, else falls back to sklearn's dataset.
- Compares a few beginner-friendly models with cross-validation.
- Saves the best pipeline as iris_model.joblib and a small report.

Run:
    python train.py
"""

from __future__ import annotations
import json
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


RANDOM_STATE = 42
HERE = pathlib.Path(__file__).parent.resolve()
DATA_FILE = HERE / "Iris.csv"  # optional external CSV
MODEL_FILE = HERE / "iris_model.joblib"
REPORT_JSON = HERE / "report.json"
CONF_MAT_PNG = HERE / "confusion_matrix.png"
MODEL_CARD = HERE / "model_card.md"


def load_data() -> Tuple[pd.DataFrame, pd.Series, Dict[int, str]]:
    """
    Returns:
        X (DataFrame), y (Series), target_map (int->name)
    Accepts:
        - A local Iris.csv with columns similar to: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
        - Otherwise uses sklearn load_iris()
    """
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        # Normalize common column names
        rename_map = {
            "sepal_length": "sepal length (cm)", "sepalwidth": "sepal width (cm)",
            "SepalLengthCm": "sepal length (cm)", "SepalWidthCm": "sepal width (cm)",
            "PetalLengthCm": "petal length (cm)", "PetalWidthCm": "petal width (cm)",
            "species": "species", "Species": "species"
        }
        df = df.rename(columns=rename_map)

        # Minimal schema check
        needed = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        if not all(col in df.columns for col in needed):
            raise ValueError(f"CSV missing required columns. Need: {needed}")

        if "species" not in df.columns:
            raise ValueError("CSV must have a 'species' column.")

        X = df[needed].copy()
        y = df["species"].astype(str)

        # Build map int->name for consistency
        classes = sorted(y.unique().tolist())
        idx_map = {i: name for i, name in enumerate(classes)}
        # Convert y to numeric for compatibility with some tools if needed
        y = y.map({name: i for i, name in idx_map.items()})

        return X, y, idx_map

    # Fallback to sklearn dataset
    iris = datasets.load_iris(as_frame=True)
    X = iris.frame[iris.feature_names].copy()
    y = pd.Series(iris.target, name="target")
    idx_map = {i: name for i, name in enumerate(iris.target_names)}
    return X, y, idx_map


def build_candidates() -> Dict[str, Pipeline]:
    """
    A few solid starter models wrapped in pipelines with scaling where it helps.
    """
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "rf": Pipeline([
            # Trees don’t need scaling, but leaving it is fine & harmless
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE
            ))
        ]),
    }


def main():
    X, y, target_map = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    candidates = build_candidates()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[name] = {
            "mean_acc": float(np.mean(scores)),
            "std_acc": float(np.std(scores)),
            "folds": scores.tolist(),
        }

    # Pick the best by mean CV accuracy
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean_acc"])
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    to_save = {
        "pipeline": best_model,
        "target_map": target_map,     # int -> class name
        "feature_order": list(X.columns),
    }
    dump(to_save, MODEL_FILE)

    # Save report
    summary = {
        "chosen_model": best_name,
        "cv_scores": cv_scores,
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1m),
        "labels": target_map,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2))

    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    tick_labels = [target_map[i] for i in range(len(target_map))]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(CONF_MAT_PNG, dpi=180)
    plt.close(fig)

    # Tiny model card
    MODEL_CARD.write_text(f"""# Iris Classifier — Model Card

**Best model**: `{best_name}`  
**CV mean accuracy (train)**: {cv_scores[best_name]["mean_acc"]:.3f} ± {cv_scores[best_name]["std_acc"]:.3f}  
**Test accuracy**: {acc:.3f}  
**Test macro F1**: {f1m:.3f}

**Classes**: {", ".join([target_map[i] for i in range(len(target_map))])}

**Features order**:
{", ".join(to_save["feature_order"])}

This model was trained via a scikit-learn Pipeline and saved with joblib.
""")

    print("✅ Training complete.")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {acc:.3f} | Macro F1: {f1m:.3f}")
    print(f"Saved: {MODEL_FILE.name}, {REPORT_JSON.name}, {CONF_MAT_PNG.name}, {MODEL_CARD.name}")


if __name__ == "__main__":
    main()
