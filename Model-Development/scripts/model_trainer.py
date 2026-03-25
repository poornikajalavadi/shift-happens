import os
import sys
import pickle
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_trainer.log"),
        logging.StreamHandler()
    ]
)

# Output directories and MLflow experiment identifier
MODEL_OUTPUT_DIR  = "models"
REPORTS_DIR       = "reports"
MLFLOW_EXPERIMENT = "ShiftHappens_Model_Development"
RANDOM_STATE      = 42


def get_candidate_models() -> dict:
    """
    Returns the two candidate models for training and comparison.

    LogisticRegression:
        Interpretable linear baseline. class_weight='balanced' compensates
        for the imbalanced TARGET distribution (~8% positive class).

    LightGBM:
        Gradient boosting model. n_estimators=300 with learning_rate=0.05
        provides stable convergence. class_weight='balanced' applied
        consistently with the baseline for fair comparison.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }


def _compute_metrics(y_test, y_pred, y_proba) -> dict:
    """
    Computes classification metrics on hold-out test predictions.
    ROC-AUC is the primary selection metric due to class imbalance.
    """
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
    }


def _save_confusion_matrix(y_test, y_pred, name: str) -> str:
    """
    Generates and saves confusion matrix to reports/.
    Returns the saved file path for MLflow artifact logging.
    """
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
    """
    Trains all candidate models, evaluates on hold-out test set,
    logs parameters, metrics, artifacts and model binaries to MLflow.

    Returns:
        dict of results sorted by ROC-AUC descending.
    """
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
            logging.info(
                f"{name} | AUC: {metrics['roc_auc']:.4f} | "
                f"F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}"
            )

    # Sort by ROC-AUC descending; index 0 is the best performing model
    return dict(sorted(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True))


def select_best_model(results: dict):
    """
    Selects the model with the highest ROC-AUC from the results dict.
    Results are pre-sorted in train_and_evaluate so index 0 is the winner.
    """
    best_name = list(results.keys())[0]
    best      = results[best_name]
    logging.info(f"Best model selected: {best_name} | AUC: {best['metrics']['roc_auc']:.4f}")
    return best_name, best["model"], best["metrics"]


def save_model(model, name: str) -> str:
    """
    Serialises the trained model to disk using pickle.
    Saved to models/best_model_<name>.pkl
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT_DIR, f"best_model_{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Model serialised to: {path}")
    return path


def plot_model_comparison(results: dict):
    """
    Generates a grouped bar chart comparing all candidate models
    across accuracy, ROC-AUC, F1, precision, and recall.
    Saved to reports/model_comparison.png
    """
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
    ax.set_title("ShiftHappens — Model Comparison: Logistic Regression vs LightGBM")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(path)
    plt.close()
    logging.info(f"Model comparison chart saved to: {path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Model Training Pipeline")
    logging.info("=" * 60)

    # Load processed dataset from Airflow pipeline output
    df = load_data()

    # Execute preprocessing: encoding, imputation, CorrelationRemover, split
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Train Logistic Regression and LightGBM; evaluate on hold-out test set
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Generate model comparison bar chart
    plot_model_comparison(results)

    # Select model with highest ROC-AUC as production candidate
    best_name, best_model, best_metrics = select_best_model(results)

    # Serialise best model to disk
    model_path = save_model(best_model, best_name)

    logging.info("=" * 60)
    logging.info(f"Training complete. Best model: {best_name}")
    for k, v in best_metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    logging.info(f"Model path: {model_path}")
    logging.info("=" * 60)
# trigger test
