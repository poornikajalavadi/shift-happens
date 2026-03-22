import os
import logging
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, roc_curve,
    precision_recall_curve, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_validator.log"),
        logging.StreamHandler()
    ]
)

REPORTS_DIR       = "reports"
MLFLOW_EXPERIMENT = "ShiftHappens_Model_Development"

# Thresholds adjusted for imbalanced dataset (only 8% positive class)
# F1 of 0.28+ is realistic for heavily imbalanced credit default data
PASS_THRESHOLDS = {
    "roc_auc":  0.70,   # Strong threshold — our model hits 0.7779
    "f1":       0.25,   # Realistic for 8% positive class imbalance
    "accuracy": 0.60,   # Our model hits 0.7335
}

def validate_model(model, X_test, y_test, model_name: str = "best_model") -> bool:
    """
    Validates model on hold-out test set.
    Generates ROC curve, PR curve, and classification report.
    Logs everything to MLflow.
    Returns True if model passes all PASS_THRESHOLDS, False otherwise.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logging.info(f"Validating model: {model_name}")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "val_roc_auc":  roc_auc_score(y_test, y_proba),
        "val_f1":       f1_score(y_test, y_pred, zero_division=0),
        "val_accuracy": accuracy_score(y_test, y_pred),
    }

    logging.info("--- Validation Metrics ---")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    # Save classification report to file
    report      = classification_report(y_test, y_pred, zero_division=0)
    report_path = os.path.join(REPORTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logging.info(f"Classification report saved → {report_path}")

    # ROC + PR curves saved to reports/
    fpr, tpr, _          = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"AUC = {metrics['val_roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {model_name}")
    axes[0].legend()
    axes[1].plot(recall, precision, color="darkorange", lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    curve_path = os.path.join(REPORTS_DIR, f"roc_pr_curves_{model_name}.png")
    plt.savefig(curve_path)
    plt.close()
    logging.info(f"ROC/PR curves saved → {curve_path}")

    # Log to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{model_name}_Validation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(curve_path)
        mlflow.log_artifact(report_path)

    # Threshold check
    passed = True
    for key, threshold in PASS_THRESHOLDS.items():
        metric_key = f"val_{key}"
        val = metrics.get(metric_key, 0)
        if val < threshold:
            logging.warning(f"VALIDATION FAILED: {metric_key} = {val:.4f} (required >= {threshold})")
            passed = False

    if passed:
        logging.info(f"Validation PASSED for {model_name}. All thresholds met.")
    else:
        logging.error(f"Validation FAILED for {model_name}. Check reports/.")

    return passed


if __name__ == "__main__":
    import sys
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Starting Model Validation")
    logging.info("=" * 60)

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Load best saved model
    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Validate on hold-out test set
    passed = validate_model(model, X_test, y_test, model_name="LightGBM")

    if passed:
        logging.info("Model passed all thresholds. Ready for bias detection.")
    else:
        logging.error("Model failed validation. Review reports/.")
