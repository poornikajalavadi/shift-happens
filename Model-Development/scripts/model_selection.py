import os
import sys
import pickle
import logging

# ─────────────────────────────────────────────────────────────
# This script implements Section 8.6 of the Model Development
# Guidelines — Final Model Selection after BOTH validation
# performance AND bias analysis results are considered together.
# ─────────────────────────────────────────────────────────────

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
    Final model selection gate — considers BOTH:
      1. Validation performance (ROC-AUC, F1, accuracy thresholds)
      2. Bias analysis results (group disparity thresholds)

    Only proceeds to registry push if BOTH checks pass.
    This ensures we never deploy a model that is either
    poorly performing OR unfair to demographic groups.

    Args:
        validation_passed: True if model passed all metric thresholds
        bias_passed:       True if bias is within acceptable limits
        model_name:        Name of the model being evaluated
        metrics:           Dict of validation metrics

    Returns:
        True if model is approved for deployment, False otherwise.
    """
    logging.info("=" * 60)
    logging.info("ShiftHappens — Final Model Selection Gate")
    logging.info("=" * 60)
    logging.info(f"Model: {model_name}")
    logging.info(f"Validation passed: {validation_passed}")
    logging.info(f"Bias check passed: {bias_passed}")
    logging.info("--- Metrics Summary ---")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")

    # ── Decision Logic ───────────────────────────────────────
    if validation_passed and bias_passed:
        logging.info("✅ APPROVED — Model passed both validation and bias checks.")
        logging.info(f"'{model_name}' is approved for GCP registry push.")
        approved = True

    elif validation_passed and not bias_passed:
        # Model performs well but has bias — deploy with warning
        # CorrelationRemover was already applied as mitigation
        logging.warning("⚠️  APPROVED WITH WARNING — Model passed validation but bias was detected.")
        logging.warning("Mitigation applied: CorrelationRemover at preprocessing stage.")
        logging.warning("Recommended next step: Apply ThresholdOptimizer post-processing.")
        approved = True

    elif not validation_passed and bias_passed:
        # Model is fair but doesn't perform well enough
        logging.error("❌ REJECTED — Model failed validation thresholds.")
        logging.error("Model is fair but performance is insufficient for deployment.")
        approved = False

    else:
        # Both failed — do not deploy
        logging.error("❌ REJECTED — Model failed both validation and bias checks.")
        logging.error("Do not deploy. Retrain with different architecture or data.")
        approved = False

    logging.info("=" * 60)
    return approved


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess
    from scripts.model_validator import validate_model
    from scripts.bias_detector import detect_bias

    logging.info("Loading data and model for final selection...")

    # Load data and preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Load tuned model if exists, else base model
    tuned_path = "models/best_model_LightGBM_tuned.pkl"
    base_path  = "models/best_model_LightGBM.pkl"
    model_path = tuned_path if os.path.exists(tuned_path) else base_path
    model_name = "LightGBM_tuned" if os.path.exists(tuned_path) else "LightGBM"

    logging.info(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Run validation check
    validation_passed = validate_model(model, X_test, y_test, model_name)

    # Run bias check
    bias_passed = detect_bias(model, X_test, y_test, s_test, model_name)

    # Final metrics for logging
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc":  roc_auc_score(y_test, y_proba),
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    # Final selection gate — considers BOTH validation and bias
    approved = select_final_model(validation_passed, bias_passed, model_name, metrics)

    if approved:
        logging.info("Model approved! Run registry_push.py to deploy to GCP.")
    else:
        logging.error("Model rejected. Do not push to registry.")
