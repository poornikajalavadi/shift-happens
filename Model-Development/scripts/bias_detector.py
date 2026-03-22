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

REPORTS_DIR       = "reports"
DISPARITY_THRESHOLD = 0.05   # Flag if max group difference exceeds this


def detect_bias(model, X_test: pd.DataFrame,
                y_test, sensitive_test,
                model_name: str = "best_model") -> bool:
    """
    Slices test data by sensitive feature (gender) and computes:
      - Accuracy per group
      - False Positive Rate per group
      - True Positive Rate per group
    Flags high disparity and saves a group metrics bar chart.

    Returns:
        True if bias is within acceptable limits, False if flagged.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logging.info(f"Running bias detection for {model_name}...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":          accuracy_score,
        "false_positive_rate": false_positive_rate,
        "true_positive_rate":  true_positive_rate,
    }

    metric_frame = MetricFrame(
        metrics           = metrics,
        y_true            = y_test,
        y_pred            = y_pred,
        sensitive_features= sensitive_test
    )

    # Log overall metrics
    logging.info("--- Overall Metrics ---")
    for name, val in metric_frame.overall.items():
        logging.info(f"  {name}: {val:.4f}")

    # Log per-group metrics
    logging.info("--- Metrics by Group ---")
    for group, row in metric_frame.by_group.iterrows():
        logging.info(f"  Group '{group}':")
        for name, val in row.items():
            logging.info(f"    {name}: {val:.4f}")

    # Check disparities
    diffs      = metric_frame.difference()
    bias_found = False
    logging.info("--- Disparity Check ---")
    for name, diff in diffs.items():
        logging.info(f"  Max difference in {name}: {diff:.4f}")
        if diff > DISPARITY_THRESHOLD:
            logging.warning(
                f"  HIGH DISPARITY in '{name}': {diff:.4f} "
                f"(threshold={DISPARITY_THRESHOLD})"
            )
            bias_found = True

    # ── Bar chart of per-group metrics ───────────────────────
    by_group = metric_frame.by_group
    ax       = by_group.plot(kind="bar", figsize=(10, 6), colormap="Set2")
    ax.set_title(f"Fairness Metrics by Group — {model_name}")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.axhline(y=DISPARITY_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({DISPARITY_THRESHOLD})")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, f"bias_by_group_{model_name}.png")
    plt.savefig(chart_path)
    plt.close()
    logging.info(f"Bias chart saved → {chart_path}")

    # Save bias report as text
    report_path = os.path.join(REPORTS_DIR, f"bias_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write("=== Overall Metrics ===\n")
        f.write(metric_frame.overall.to_string())
        f.write("\n\n=== Metrics by Group ===\n")
        f.write(metric_frame.by_group.to_string())
        f.write("\n\n=== Max Disparities ===\n")
        f.write(diffs.to_string())
    logging.info(f"Bias report saved → {report_path}")

    if bias_found:
        logging.warning("Bias detected. CorrelationRemover was applied at preprocessing.")
        logging.warning("Further mitigation: consider ThresholdOptimizer post-processing.")
    else:
        logging.info("No significant bias detected. Model is within fairness thresholds.")

    return not bias_found


if __name__ == "__main__":
    import sys
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess

    logging.info("=" * 60)
    logging.info("ShiftHappens — Starting Bias Detection")
    logging.info("=" * 60)

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Load best saved model
    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Run bias detection — slices by gender, flags disparities
    passed = detect_bias(model, X_test, y_test, s_test, model_name="LightGBM")

    if passed:
        logging.info("No significant bias detected. Model is fair across groups.")
    else:
        logging.warning("Bias detected. CorrelationRemover was applied at preprocessing stage.")
        logging.warning("Consider ThresholdOptimizer as additional post-processing mitigation.")
