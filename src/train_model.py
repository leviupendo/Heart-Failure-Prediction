"""
train_model.py
---------------
Trains a Logistic Regression model on the heart failure dataset with:
- GridSearchCV for hyperparameter tuning
- Threshold tuning optimized for F1-score
- Model + best threshold saved to disk for reuse in evaluate.py

Usage:
    python src/train_model.py
"""

import os
import warnings
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve

# Newer scikit-learn versions emit FutureWarning/UserWarning about the
# 'penalty' param in favor of 'l1_ratio'. Harmless for our use case (we pin
# explicit penalty/solver pairs), so we silence them to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from data_preprocessing import load_and_prepare_data

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_failure_clinical_records_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"

def build_param_grid():
    """
    Hyperparameter grid for Logistic Regression.
    - C: inverse regularization strength
    - penalty/solver: paired carefully since not all solvers support all penalties
    - class_weight: 'balanced' helps since DEATH_EVENT is imbalanced (~32% positive)
    """
    param_grid = [
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
        },
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1"],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced"],
            "max_iter": [1000],
        },
    ]
    return param_grid


def run_grid_search(X_train, y_train):
    """Run GridSearchCV with stratified 5-fold CV, scoring on F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=build_param_grid(),
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print("\nBest params:", grid.best_params_)
    print("Best CV F1-score:", round(grid.best_score_, 4))

    return grid.best_estimator_


def find_best_threshold(model, X_train, y_train):
    """
    Scan thresholds against predicted probabilities on the TRAINING set
    (test set stays untouched until evaluate.py) and pick the one that
    maximizes F1-score.
    """
    y_probs = model.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probs)

    # precision_recall_curve returns thresholds of length n-1 vs precisions/recalls of length n
    f1_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * p * r / (p + r))

    f1_scores = np.array(f1_scores)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\nBest threshold (train, F1-optimized): {best_threshold:.4f}")
    print(f"F1-score at that threshold (train): {best_f1:.4f}")

    return best_threshold


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(DATA_PATH)

    print("\nRunning GridSearchCV for Logistic Regression...")
    best_model = run_grid_search(X_train, y_train)

    print("\nFinding F1-optimized decision threshold...")
    best_threshold = find_best_threshold(best_model, X_train, y_train)

    print("\nSaving model, scaler, and threshold...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best_threshold, THRESHOLD_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved threshold to: {THRESHOLD_PATH}")
    print("\nDone. Run evaluate.py next to see test-set performance.")


if __name__ == "__main__":
    main()