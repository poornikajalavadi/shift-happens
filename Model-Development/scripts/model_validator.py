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

# Minimum acceptable metric thresholds for deployment approval.
# F1 threshold set conservatively due to class imbalance (8% positive class).
PASS_THRESHOLDS = {
    "roc_auc":  0.70,
    "f1":       0.25,
    "accuracy": 0.60,
}


def validate_model(model, X_test, y_test, model_name: str = "best_model") -> bool:
    """
    Validates the trained model on the hold-out test set.

    Generates:
        - ROC curve and Precision-Recall curve saved to reports/
        - Classification report saved to reports/
        - Validation metrics logged to MLflow

    Returns:
        True if all PASS_THRESHOLDS are met, False otherwise.
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

    logging.info("Validation metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    # Save classification report to file
    report      = classification_report(y_test, y_pred, zero_division=0)
    report_path = os.path.join(REPORTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logging.info(f"Classification report saved: {report_path}")

    # Generate ROC and Precision-Recall curves
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
    logging.info(f"ROC and PR curves saved: {curve_path}")

    # Log metrics and artifacts to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{model_name}_Validation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(curve_path)
        mlflow.log_artifact(report_path)

    # Evaluate metrics against minimum thresholds
    passed = True
    for key, threshold in PASS_THRESHOLDS.items():
        val = metrics.get(f"val_{key}", 0)
        if val < threshold:
            logging.warning(f"Threshold not met: val_{key} = {val:.4f} (minimum: {threshold})")
            passed = False

    if passed:
        logging.info(f"Validation passed for {model_name}. All thresholds satisfied.")
    else:
        logging.error(f"Validation failed for {model_name}. Review reports/ for details.")

    return passed


if __name__ == "__main__":
    import sys
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Model Validation")
    logging.info("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    passed = validate_model(model, X_test, y_test, model_name="LightGBM")

    if passed:
        logging.info("Model approved. Proceed to bias detection.")
    else:
        logging.error("Model rejected. Review validation reports before proceeding.")
