# Dry Bean Classification: Multi-Model Comparison with Streamlit

## a. Problem Statement

Manual inspection of dry bean varieties is slow and inconsistent. The goal of this project is to classify a dry bean into one of seven varieties using only geometric measurements taken from a single bean image (area, perimeter, axis lengths, shape factors, and so on). Five classification algorithms are trained on the same dataset and compared on six evaluation metrics. The best-performing model is served through an interactive Streamlit web application that lets a user upload a test CSV, select a model, and inspect its metrics, confusion matrix, and classification report.

## b. Dataset Description

- **Name:** Dry Bean Dataset (UCI Machine Learning Repository)
- **Type:** Multi-class classification (7 classes)
- **Instances:** 13,611
- **Features:** 16 numeric features (geometry and shape measurements of each bean)
- **Target column:** `Class`, with values `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA`, `DERMASON`
- **Missing values:** None
- **Class distribution:** Imbalanced. `DERMASON` (3,546) and `BOMBAY` (522) are the largest and smallest classes.

Features: `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`.

**Preprocessing and split:** Features are standardised with `StandardScaler` inside each model's pipeline. The data is split 80/20 into train (10,888 rows) and test (2,723 rows) using a stratified split with `random_state=42`, so class proportions are preserved. The test split is saved as `test_data.csv` and used by the Streamlit app.

## c. GitHub Repository Link

https://github.com/devaanshi15/Dry_Bean_Classification

## d. Models Used

All five models were trained on the same training split and evaluated on the same test split.

1. **Logistic Regression** (`max_iter=2000`)
2. **Decision Tree Classifier** (`max_depth=12`)
3. **K-Nearest Neighbors** (`n_neighbors=7`)
4. **Naive Bayes** (Gaussian)
5. **Support Vector Machine** (RBF kernel, `probability=True`)

**Metric notes:** This is a multi-class problem, so precision, recall, and F1 are macro-averaged (each class counts equally). AUC is the one-vs-rest macro-averaged ROC AUC, computed from predicted class probabilities.

### Comparison Table (test split)

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9214   | 0.9948 | 0.9354    | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree           | 0.9023   | 0.9595 | 0.9167    | 0.9158 | 0.9162 | 0.8819 |
| KNN                     | 0.9137   | 0.9858 | 0.9290    | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes             | 0.8979   | 0.9916 | 0.9112    | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine  | 0.9221   | 0.9949 | 0.9359    | 0.9330 | 0.9343 | 0.9059 |

### Observations on Model Performance

| ML Model Name           | Observation about model performance |
|-------------------------|-------------------------------------|
| Logistic Regression     | Strong linear baseline with 92.14% accuracy and 0.9335 macro F1; it separates the classes well after standardisation. |
| Decision Tree           | The least competitive tree-based model here, with 90.23% accuracy; its lower AUC suggests weaker probability ranking than the other models. |
| KNN                     | Performs well at 91.37% accuracy and 0.9265 macro F1, but is slightly behind the linear and kernel methods. |
| Naive Bayes             | Fast probabilistic baseline with the lowest accuracy (89.79%) and macro F1 (0.9091), likely because its feature-distribution assumptions are restrictive. |
| Support Vector Machine  | Best overall results, reaching 92.21% accuracy, 0.9949 AUC, and 0.9343 macro F1. |

**Overall winner for this dataset:** Support Vector Machine. It has the highest accuracy, AUC, macro precision, recall, F1, and MCC on the common held-out test split, so it is the selected model for this dataset.

## Repository Structure

```
Dry_Bean_Classification/
├── app.py                  # Streamlit web application
├── train_models.py         # Trains, evaluates and saves all five models
├── requirements.txt        # Pinned dependencies
├── README.md
├── test_data.csv           # Held-out test split (features + Class)
└── model/                  # Saved sklearn pipelines (.joblib) + metrics/classes files
```

## How to Run

```bash
pip install -r requirements.txt

# (Optional) retrain and regenerate model/ and test_data.csv
python train_models.py --data path/to/Dry_Bean_Dataset.xlsx

# Launch the app
streamlit run app.py
```

The app does not retrain models. It loads the saved pipelines from `model/` and computes metrics on the uploaded CSV.

## Using the App

1. Upload a CSV with the 16 feature columns and the `Class` column, or tick "Use bundled test split".
2. Choose a model from the dropdown, or choose "All models" for a side-by-side comparison.
3. Review the metrics, confusion matrix, and classification report.

## Live Streamlit App

Deploy this repository through Streamlit Community Cloud and replace this line with the generated URL:

`https://drybeanclassification-devaanshi-ml-project.streamlit.app/`
