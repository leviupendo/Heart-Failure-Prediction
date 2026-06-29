"""
evaluate.py
-----------
Loads the trained model, scaler, and F1-optimized threshold, then evaluates
on the held-out test set. Saves confusion matrix and ROC/PR curve plots
to outputs/.

Usage:
    python src/evaluate.py
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    average_precision_score,
)

from data_preprocessing import load_and_prepare_data

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_failure_clinical_records_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, scaler, threshold


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Survived", "Death"],
        yticklabels=["Survived", "Death"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(y_true, y_probs, save_path):
    precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(5, 4))
    plt.plot(recalls, precisions, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PR curve to {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data and trained artifacts...")
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(DATA_PATH)
    model, scaler, threshold = load_artifacts()

    print(f"\nUsing F1-optimized threshold from training: {threshold:.4f}")

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probs >= 0.5).astype(int)
    y_pred_tuned = (y_probs >= threshold).astype(int)

    print("\n=== Default threshold (0.5) ===")
    print(classification_report(y_test, y_pred_default, target_names=["Survived", "Death"]))
    print("F1-score:", round(f1_score(y_test, y_pred_default), 4))

    print("\n=== Tuned threshold (F1-optimized) ===")
    print(classification_report(y_test, y_pred_tuned, target_names=["Survived", "Death"]))
    print("F1-score:", round(f1_score(y_test, y_pred_tuned), 4))

    auc = roc_auc_score(y_test, y_probs)
    print(f"\nROC-AUC (threshold-independent): {auc:.4f}")

    # Plots use the tuned threshold predictions for the confusion matrix
    plot_confusion_matrix(y_test, y_pred_tuned, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_probs, os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_precision_recall_curve(y_test, y_probs, os.path.join(OUTPUT_DIR, "pr_curve.png"))

    print("\nEvaluation complete. Check outputs/ for saved plots.")


if __name__ == "__main__":
    main()