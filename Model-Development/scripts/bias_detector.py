import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bias_detector.log"),
        logging.StreamHandler()
    ]
)

REPORTS_DIR         = "reports"
DISPARITY_THRESHOLD = 0.05  # Maximum acceptable metric difference between groups


def detect_bias(model, X_test: pd.DataFrame,
                y_test, sensitive_test,
                model_name: str = "best_model") -> bool:
    """
    Evaluates model fairness by slicing test data across sensitive
    feature groups (CODE_GENDER) and computing per-group metrics.

    Metrics evaluated:
        - Accuracy per group
        - False Positive Rate per group
        - True Positive Rate per group

    Generates:
        - Bar chart of per-group metrics saved to reports/
        - Text bias report saved to reports/

    Returns:
        True if all group disparities are within DISPARITY_THRESHOLD.
        False if any metric exceeds the threshold (bias flagged).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logging.info(f"Initiating bias detection for model: {model_name}")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":            accuracy_score,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate":  true_positive_rate,
    }

    metric_frame = MetricFrame(
        metrics           =metrics,
        y_true            =y_test,
        y_pred            =y_pred,
        sensitive_features=sensitive_test
    )

    logging.info("Overall metrics:")
    for name, val in metric_frame.overall.items():
        logging.info(f"  {name}: {val:.4f}")

    logging.info("Per-group metrics:")
    for group, row in metric_frame.by_group.iterrows():
        logging.info(f"  Group '{group}':")
        for name, val in row.items():
            logging.info(f"    {name}: {val:.4f}")

    # Compute maximum disparity between any two groups per metric
    diffs      = metric_frame.difference()
    bias_found = False
    logging.info("Group disparity analysis:")
    for name, diff in diffs.items():
        logging.info(f"  Maximum difference in {name}: {diff:.4f}")
        if diff > DISPARITY_THRESHOLD:
            logging.warning(
                f"  Disparity threshold exceeded in '{name}': "
                f"{diff:.4f} (threshold: {DISPARITY_THRESHOLD})"
            )
            bias_found = True

    # Generate per-group metrics bar chart
    by_group = metric_frame.by_group
    ax       = by_group.plot(kind="bar", figsize=(10, 6), colormap="Set2")
    ax.set_title(f"Fairness Metrics by Group — {model_name}")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.axhline(y=DISPARITY_THRESHOLD, color="red", linestyle="--",
               label=f"Disparity threshold ({DISPARITY_THRESHOLD})")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, f"bias_by_group_{model_name}.png")
    plt.savefig(chart_path)
    plt.close()
    logging.info(f"Bias chart saved: {chart_path}")

    # Save full bias report to text file
    report_path = os.path.join(REPORTS_DIR, f"bias_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write("=== Overall Metrics ===\n")
        f.write(metric_frame.overall.to_string())
        f.write("\n\n=== Metrics by Group ===\n")
        f.write(metric_frame.by_group.to_string())
        f.write("\n\n=== Maximum Disparities ===\n")
        f.write(diffs.to_string())
    logging.info(f"Bias report saved: {report_path}")

    if bias_found:
        logging.warning("Bias detected. CorrelationRemover was applied at preprocessing stage.")
        logging.warning("Recommended mitigation: apply ThresholdOptimizer post-processing (bias_mitigation.py).")
    else:
        logging.info("No significant bias detected. Model meets fairness thresholds.")

    return not bias_found


if __name__ == "__main__":
    import sys
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Bias Detection")
    logging.info("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    passed = detect_bias(model, X_test, y_test, s_test, model_name="LightGBM")

    if passed:
        logging.info("Model meets fairness thresholds. Proceed to model selection.")
    else:
        logging.warning("Bias detected. CorrelationRemover applied at preprocessing.")
        logging.warning("Execute bias_mitigation.py to apply ThresholdOptimizer post-processing.")
