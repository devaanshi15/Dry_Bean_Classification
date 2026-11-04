"""
Train and evaluate five classification models on the UCI Dry Bean dataset.

Outputs:
    model/<name>.joblib        : one sklearn Pipeline (StandardScaler + classifier) per model
    model/metrics_summary.csv  : metrics on the held-out test split
    model/classes.json         : class label order used by the models
    test_data.csv              : held-out test split (features + Class) for the Streamlit app

Usage:
    python train_models.py --data path/to/Dry_Bean_Dataset.xlsx
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_DIR = "model"


def get_models():
    """Return the five classifiers, each wrapped with a StandardScaler."""
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "naive_bayes": GaussianNB(),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        for name, clf in classifiers.items()
    }


def evaluate(pipeline, X_test, y_test):
    """Compute the six required metrics for one fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        # Multi-class AUC: one-vs-rest, averaged over the 7 classes
        "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"),
        # Multi-class precision/recall/F1: macro average (treats each class equally)
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def load_dataset(data_path):
    """Load a CSV or Excel dataset based on its file extension."""
    extension = os.path.splitext(data_path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(data_path)
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(data_path)
    raise ValueError("Unsupported dataset format. Use a .csv, .xls, or .xlsx file.")


def main(data_path):
    df = load_dataset(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split keeps the class proportions the same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the test split so the Streamlit app can evaluate models on it
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv("test_data.csv", index=False)

    results = []
    for name, pipeline in get_models().items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.joblib"))

        metrics = evaluate(pipeline, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print({k: round(v, 4) for k, v in metrics.items() if k != "Model"})

    summary = pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

    # Class order used by predict_proba (sorted alphabetically by sklearn)
    classes = sorted(y.unique().tolist())
    with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    print("\nComparison table (test split):")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a Dry Bean dataset in CSV or Excel format",
    )
    args = parser.parse_args()
    main(args.data)
