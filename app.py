import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET = "Class"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


@st.cache_resource
def load_model(file_stem):
    """Load a saved sklearn Pipeline. Cached so each model is read from disk once."""
    return joblib.load(os.path.join(MODEL_DIR, f"{file_stem}.joblib"))


@st.cache_data
def load_classes():
    with open(os.path.join(MODEL_DIR, "classes.json")) as f:
        return json.load(f)


def compute_metrics(model, X, y):
    """Compute the six required metrics on the given data (no retraining)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y, y_pred, average="macro"),
        "Recall": recall_score(y, y_pred, average="macro"),
        "F1": f1_score(y, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


# ---------------- UI ----------------
st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification: Model Evaluation")
st.write(
    "Compare five classifiers trained on the UCI Dry Bean dataset (7 bean varieties, "
    "16 shape features). Upload a **test CSV** with the same feature columns and a "
    f"`{TARGET}` column, or use the bundled test split."
)

# Input source
st.sidebar.header("1. Test data")
use_bundled = st.sidebar.checkbox("Use bundled test split (test_data.csv)", value=False)
uploaded = st.sidebar.file_uploader("...or upload a test CSV", type=["csv"])

# Model selection
st.sidebar.header("2. Model")
model_label = st.sidebar.selectbox("Choose a model", list(MODEL_OPTIONS.keys()) + ["All models"])

# Load the data
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = f"uploaded file: {uploaded.name}"
elif use_bundled:
    data = pd.read_csv(TEST_CSV)
    source = "bundled test split"
else:
    st.info("Upload a test CSV in the sidebar, or tick 'Use bundled test split'.")
    st.stop()

st.caption(f"Data source: {source} | Rows: {len(data)}")

# Validate the input
classes = load_classes()
expected_features = [c for c in pd.read_csv(TEST_CSV, nrows=1).columns if c != TARGET]

if TARGET not in data.columns:
    st.error(f"The CSV must contain the target column `{TARGET}`.")
    st.stop()

missing = [c for c in expected_features if c not in data.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = data[expected_features]
y = data[TARGET]

st.subheader("Evaluation metrics")

if model_label == "All models":
    rows = []
    for label, stem in MODEL_OPTIONS.items():
        model = load_model(stem)
        m = compute_metrics(model, X, y)
        rows.append({"ML Model": label, **m})
    comparison = pd.DataFrame(rows).set_index("ML Model").round(4)
    st.dataframe(comparison, use_container_width=True)
    st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])

    st.subheader("Confusion matrix (select a single model to view its matrix)")
    st.info("Pick one model in the sidebar to see its confusion matrix and classification report.")
else:
    stem = MODEL_OPTIONS[model_label]
    model = load_model(stem)
    m = compute_metrics(model, X, y)

    cols = st.columns(6)
    for col, (name, value) in zip(cols, m.items()):
        col.metric(name, f"{value:.4f}")

    y_pred = model.predict(X)

    st.subheader("Confusion matrix")
    fig = plot_confusion_matrix(y, y_pred, classes, f"{model_label}: confusion matrix")
    st.pyplot(fig)

    st.subheader("Classification report")
    report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with st.expander("Reference: metrics on the held-out test split (from training)"):
    summary_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    if os.path.exists(summary_path):
        st.dataframe(pd.read_csv(summary_path).round(4), use_container_width=True)
