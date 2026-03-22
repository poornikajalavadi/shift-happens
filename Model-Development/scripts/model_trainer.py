import os
import sys
import pickle
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────
# Logging — outputs to both file and console
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_trainer.log"),
        logging.StreamHandler()
    ]
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MODEL_OUTPUT_DIR  = "models"
REPORTS_DIR       = "reports"
MLFLOW_EXPERIMENT = "ShiftHappens_Model_Development"
RANDOM_STATE      = 42

def get_candidate_models() -> dict:
    """
    Two candidate models for ShiftHappens:
    1. LogisticRegression — interpretable baseline
    2. LightGBM — gradient boosting, expected to outperform LR
    Both use class_weight='balanced' to handle TARGET class imbalance.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=500, class_weight="balanced",
            random_state=RANDOM_STATE, solver="lbfgs"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def _compute_metrics(y_test, y_pred, y_proba) -> dict:
    """Computes accuracy, ROC-AUC, F1, precision, recall."""
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
    }


def _save_confusion_matrix(y_test, y_pred, name: str) -> str:
    """Saves confusion matrix plot to reports/"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"confusion_matrix_{name}.png")
    plt.savefig(path)
    plt.close()
    return path


def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Trains both models, logs to MLflow, returns results sorted by AUC."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    models  = get_candidate_models()
    results = {}

    for name, model in models.items():
        logging.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            cm_path = _save_confusion_matrix(y_test, y_pred, name)
            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")
            results[name] = {"model": model, "metrics": metrics}
            logging.info(f"{name} | AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")

    return dict(sorted(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True))


def select_best_model(results: dict):
    """Selects model with highest ROC-AUC."""
    best_name = list(results.keys())[0]
    best      = results[best_name]
    logging.info(f"Best model: {best_name} | AUC: {best['metrics']['roc_auc']:.4f}")
    return best_name, best["model"], best["metrics"]


def save_model(model, name: str) -> str:
    """Saves model as pkl file."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT_DIR, f"best_model_{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model saved → {path}")
    return path


def plot_model_comparison(results: dict):
    """Bar chart comparing both models across all metrics."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    names   = list(results.keys())
    metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    x, w    = np.arange(len(names)), 0.15
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        vals = [results[n]["metrics"][metric] for n in names]
        ax.bar(x + i * w, vals, w, label=metric)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("ShiftHappens — Logistic Regression vs LightGBM")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(path)
    plt.close()
    logging.info(f"Comparison chart saved → {path}")


if __name__ == "__main__":
    # Add parent directory to path so 'scripts' module is found correctly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Starting Model Training Pipeline")
    logging.info("=" * 60)

    # Step 1 — Load processed data from Airflow pipeline output
    df = load_data()

    # Step 2 — Preprocess: encode, impute, apply CorrelationRemover
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Step 3 — Train Logistic Regression + LightGBM, evaluate both
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Step 4 — Save comparison bar chart to reports/
    plot_model_comparison(results)

    # Step 5 — Select best model by ROC-AUC
    best_name, best_model, best_metrics = select_best_model(results)

    # Step 6 — Save best model pkl to models/
    model_path = save_model(best_model, best_name)

    logging.info("=" * 60)
    logging.info(f"Training Complete! Best Model: {best_name}")
    logging.info(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    logging.info(f"  F1 Score:  {best_metrics['f1']:.4f}")
    logging.info(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    logging.info(f"  Precision: {best_metrics['precision']:.4f}")
    logging.info(f"  Recall:    {best_metrics['recall']:.4f}")
    logging.info(f"Model saved to: {model_path}")
    logging.info("=" * 60)
