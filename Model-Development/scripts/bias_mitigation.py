import os
import sys
import pickle
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────
# Post Bias Mitigation Script
# ─────────────────────────────────────────────────────────────
# After bias is detected in bias_detector.py, this script:
#   1. Loads the tuned LightGBM model
#   2. Filters out degenerate groups (XNA gender - Group 2)
#      that have only one class label — ThresholdOptimizer
#      cannot compute fairness constraints for these groups
#   3. Applies ThresholdOptimizer with equalized_odds constraint
#      which forces equal TPR and FPR across gender groups
#   4. Saves final debiased model → models/final_model_debiased.pkl
#   5. Shows before vs after bias comparison
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bias_mitigation.log"),
        logging.StreamHandler()
    ]
)

MODELS_DIR  = "models"
REPORTS_DIR = "reports"

def filter_degenerate_groups(X, y, sensitive):
    """
    Removes samples belonging to groups that have only one
    class label (degenerate groups). ThresholdOptimizer requires
    both classes (0 and 1) to be present in every group.

    In our dataset, Group 2 (XNA gender) has all the same
    TARGET value so we filter it out before mitigation.

    Returns filtered X, y, sensitive arrays.
    """
    import pandas as pd
    sensitive = np.array(sensitive)
    y         = np.array(y)

    valid_mask = np.ones(len(y), dtype=bool)

    for group in np.unique(sensitive):
        group_mask   = sensitive == group
        group_labels = y[group_mask]
        # If group has only one unique label — it's degenerate
        if len(np.unique(group_labels)) < 2:
            logging.warning(
                f"Group '{group}' has degenerate labels "
                f"(only class {np.unique(group_labels)}). Removing from mitigation."
            )
            valid_mask[group_mask] = False

    X_filtered         = X[valid_mask] if hasattr(X, 'iloc') else X[valid_mask]
    y_filtered         = y[valid_mask]
    sensitive_filtered = sensitive[valid_mask]

    logging.info(f"Samples after filtering degenerate groups: {valid_mask.sum()}")
    return X_filtered, y_filtered, sensitive_filtered


def apply_threshold_optimizer(model, X_train, y_train, sensitive_train):
    """
    Applies Fairlearn ThresholdOptimizer with equalized_odds constraint.
    equalized_odds forces equal TPR AND FPR across all gender groups.
    Degenerate groups are filtered out before fitting.
    """
    logging.info("Filtering degenerate groups before ThresholdOptimizer...")
    X_f, y_f, s_f = filter_degenerate_groups(X_train, y_train, sensitive_train)

    logging.info("Applying ThresholdOptimizer with equalized_odds constraint...")
    mitigated_model = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        objective="balanced_accuracy_score"
    )

    mitigated_model.fit(X_f, y_f, sensitive_features=s_f)
    logging.info("ThresholdOptimizer fitted successfully.")
    return mitigated_model

def compare_bias_before_after(original_model, mitigated_model,
                               X_test, y_test, sensitive_test):
    """
    Runs bias detection on both original and mitigated model.
    Generates before vs after comparison chart saved to reports/.
    """
    logging.info("Comparing bias before and after mitigation...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    metrics = {
        "accuracy":            accuracy_score,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate":  true_positive_rate,
    }

    # Before mitigation — original model predictions
    y_pred_before = original_model.predict(X_test)
    mf_before = MetricFrame(
        metrics=metrics, y_true=y_test,
        y_pred=y_pred_before, sensitive_features=sensitive_test
    )

    # After mitigation — ThresholdOptimizer predictions
    y_pred_after = mitigated_model.predict(
        X_test, sensitive_features=sensitive_test
    )
    mf_after = MetricFrame(
        metrics=metrics, y_true=y_test,
        y_pred=y_pred_after, sensitive_features=sensitive_test
    )

    # Log disparities before and after
    logging.info("--- Disparity BEFORE mitigation ---")
    for name, diff in mf_before.difference().items():
        logging.info(f"  {name}: {diff:.4f}")
    logging.info("--- Disparity AFTER mitigation ---")
    for name, diff in mf_after.difference().items():
        logging.info(f"  {name}: {diff:.4f}")

    # Plot before vs after comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mf_before.by_group.plot(kind="bar", ax=axes[0], colormap="Set2")
    axes[0].set_title("Bias by Group — BEFORE Mitigation")
    axes[0].set_ylabel("Score")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    mf_after.by_group.plot(kind="bar", ax=axes[1], colormap="Set2")
    axes[1].set_title("Bias by Group — AFTER Mitigation")
    axes[1].set_ylabel("Score")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    plt.suptitle("ShiftHappens — Bias Mitigation: Before vs After", fontsize=13)
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "bias_before_after_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    logging.info(f"Before vs after chart saved → {chart_path}")
    return mf_before.difference(), mf_after.difference()

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Post Bias Mitigation")
    logging.info("=" * 60)

    # Step 1 — Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Step 2 — Load the tuned model
    model_path = "models/best_model_LightGBM_tuned.pkl"
    logging.info(f"Loading tuned model from {model_path}...")
    with open(model_path, "rb") as f:
        original_model = pickle.load(f)

    # Step 3 — Apply ThresholdOptimizer (filters degenerate groups first)
    mitigated_model = apply_threshold_optimizer(
        original_model, X_train, y_train, s_train
    )

    # Step 4 — Compare bias before vs after
    diff_before, diff_after = compare_bias_before_after(
        original_model, mitigated_model, X_test, y_test, s_test
    )

    # Step 5 — Save final debiased model
    os.makedirs(MODELS_DIR, exist_ok=True)
    final_path = os.path.join(MODELS_DIR, "final_model_debiased.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(mitigated_model, f)

    logging.info("=" * 60)
    logging.info("Bias Mitigation Complete!")
    logging.info("Disparity BEFORE:")
    for k, v in diff_before.items():
        logging.info(f"  {k}: {v:.4f}")
    logging.info("Disparity AFTER:")
    for k, v in diff_after.items():
        logging.info(f"  {k}: {v:.4f}")
    logging.info(f"Final debiased model saved → {final_path}")
    logging.info("=" * 60)
