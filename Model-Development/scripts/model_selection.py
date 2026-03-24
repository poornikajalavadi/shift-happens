import os
import sys
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_selection.log"),
        logging.StreamHandler()
    ]
)

REPORTS_DIR = "reports"
MODELS_DIR  = "models"


def select_final_model(
    validation_passed: bool,
    bias_passed: bool,
    model_name: str,
    metrics: dict
) -> bool:
    """
    Final model selection gate combining validation performance
    and bias detection results.

    Decision logic:
        - Validation passed AND bias within threshold  → APPROVED
        - Validation passed BUT bias detected          → APPROVED WITH WARNING
          (CorrelationRemover mitigation already applied; ThresholdOptimizer recommended)
        - Validation failed AND bias within threshold  → REJECTED
        - Validation failed AND bias detected          → REJECTED

    Only approved models proceed to the GCP registry push.

    Args:
        validation_passed: True if model met all metric thresholds.
        bias_passed:       True if group disparity is within threshold.
        model_name:        Identifier of the model under evaluation.
        metrics:           Dict of validation metrics.

    Returns:
        True if model is approved for deployment, False otherwise.
    """
    logging.info("=" * 60)
    logging.info("ShiftHappens — Final Model Selection Gate")
    logging.info("=" * 60)
    logging.info(f"Model: {model_name}")
    logging.info(f"Validation passed: {validation_passed}")
    logging.info(f"Bias check passed: {bias_passed}")
    logging.info("Metric summary:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    if validation_passed and bias_passed:
        logging.info("APPROVED — Model satisfies both validation and fairness criteria.")
        approved = True

    elif validation_passed and not bias_passed:
        logging.warning("APPROVED WITH WARNING — Validation passed. Bias detected.")
        logging.warning("Mitigation applied: CorrelationRemover at preprocessing stage.")
        logging.warning("Recommended: execute bias_mitigation.py to apply ThresholdOptimizer.")
        approved = True

    elif not validation_passed and bias_passed:
        logging.error("REJECTED — Model failed validation thresholds.")
        logging.error("Performance insufficient for deployment despite passing fairness checks.")
        approved = False

    else:
        logging.error("REJECTED — Model failed both validation and fairness checks.")
        logging.error("Retrain required with revised architecture or data.")
        approved = False

    logging.info("=" * 60)
    return approved


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess
    from scripts.model_validator import validate_model
    from scripts.bias_detector import detect_bias
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    logging.info("Loading data and model for final selection evaluation.")

    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    tuned_path = "models/best_model_LightGBM_tuned.pkl"
    base_path  = "models/best_model_LightGBM.pkl"
    model_path = tuned_path if os.path.exists(tuned_path) else base_path
    model_name = "LightGBM_tuned" if os.path.exists(tuned_path) else "LightGBM"

    logging.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    validation_passed = validate_model(model, X_test, y_test, model_name)
    bias_passed       = detect_bias(model, X_test, y_test, s_test, model_name)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc":  roc_auc_score(y_test, y_proba),
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    approved = select_final_model(validation_passed, bias_passed, model_name, metrics)

    if approved:
        logging.info("Model approved for registry push. Execute registry_push.py.")
    else:
        logging.error("Model rejected. Registry push blocked.")
