"""
predictor.py — Flexible prediction script that works on any data.

Usage:
    python3 scripts/predictor.py data/new_applications.csv

Input:
    Any CSV file — handles missing columns, extra columns,
    different formats automatically.

Output:
    predictions/<filename>_predictions.csv
    (Original data + PREDICTION + PREDICTION_PROBA columns)
"""

import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fairlearn.preprocessing import CorrelationRemover

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/predictor.log"),
        logging.StreamHandler()
    ]
)

MODELS_DIR        = "models"
PREDICTIONS_DIR   = "predictions"
SENSITIVE_FEATURE = "CODE_GENDER"
DROP_COLS         = ["SK_ID_CURR", "TARGET"]


def load_model():
    """Load the final debiased model."""
    model_path = os.path.join(MODELS_DIR, "final_model_debiased.pkl")

    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run model_trainer.py and hyperparameter_tuner.py first."
        )

    logging.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def get_model_features(model):
    """
    Extract the feature names the model was trained on.
    Works with LightGBM, scikit-learn, and ThresholdOptimizer models.
    """
    # For ThresholdOptimizer, get features from the underlying estimator
    if hasattr(model, 'estimator_'):
        inner = model.estimator_
        if hasattr(inner, 'feature_name_'):
            return inner.feature_name_
        elif hasattr(inner, 'booster_'):
            return inner.booster_.feature_name()
        elif hasattr(inner, 'feature_names_in_'):
            return list(inner.feature_names_in_)

    if hasattr(model, 'feature_name_'):
        return model.feature_name_
    elif hasattr(model, 'booster_'):
        return model.booster_.feature_name()
    elif hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    else:
        logging.warning("Could not extract feature names from model.")
        return None


def preprocess_new_data(df: pd.DataFrame, expected_features: list = None) -> pd.DataFrame:
    """
    Flexibly preprocess any incoming data to match what the
    model expects.

    Handles:
        - Extra columns in new data (drops them)
        - Missing columns in new data (fills with 0)
        - Text/categorical columns (encodes them)
        - Missing values (fills with median)
        - Gender bias removal (CorrelationRemover)
        - Different column ordering (reorders to match model)
    """
    logging.info(f"Preprocessing new data. Shape: {df.shape}")
    df_copy = df.copy()

    # Save gender column before processing
    sensitive_raw = df_copy[SENSITIVE_FEATURE].copy() if SENSITIVE_FEATURE in df_copy.columns else None

    # Drop ID and TARGET columns if they exist
    X = df_copy.drop(columns=[c for c in DROP_COLS if c in df_copy.columns], errors='ignore')

    # Encode all text/categorical columns to numbers
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing values with column median
    X = X.fillna(X.median(numeric_only=True))

    # Apply CorrelationRemover for gender bias
    if sensitive_raw is not None and SENSITIVE_FEATURE in X.columns:
        logging.info(f"Applying CorrelationRemover for: {SENSITIVE_FEATURE}")
        cr = CorrelationRemover(sensitive_feature_ids=[SENSITIVE_FEATURE])
        X_arr = cr.fit_transform(X)
        remaining_cols = [c for c in X.columns if c != SENSITIVE_FEATURE]
        X = pd.DataFrame(X_arr, columns=remaining_cols, index=X.index)

    # ── Make data match what the model expects ──────────────
    if expected_features is not None:
        # Find what's missing and what's extra
        current_cols  = set(X.columns)
        expected_cols = set(expected_features)

        missing_cols = expected_cols - current_cols
        extra_cols   = current_cols - expected_cols

        # Drop extra columns that model doesn't need
        if extra_cols:
            logging.info(f"Dropping {len(extra_cols)} extra columns: {list(extra_cols)[:5]}...")
            X = X.drop(columns=list(extra_cols), errors='ignore')

        # Add missing columns with value 0
        if missing_cols:
            logging.warning(f"Adding {len(missing_cols)} missing columns with default value 0: {list(missing_cols)[:5]}...")
            for col in missing_cols:
                X[col] = 0

        # Reorder columns to match model's expected order
        X = X[expected_features]

        logging.info(f"Aligned to {len(expected_features)} model features.")

    logging.info(f"Final feature matrix shape: {X.shape}")
    return X


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions on any incoming data.

    Automatically aligns the data to match what the model
    expects — handles missing columns, extra columns,
    different formats.

    Adds two columns:
        - PREDICTION: 0 (safe) or 1 (will default)
        - PREDICTION_PROBA: confidence score (0.0 to 1.0)
    """
    # Get what features the model expects
    expected_features = get_model_features(model)

    if expected_features:
        logging.info(f"Model expects {len(expected_features)} features.")
    else:
        logging.warning("Unknown model features. Predicting with available data.")

    # Preprocess and align data
    X = preprocess_new_data(df, expected_features)

    # Generate predictions
    logging.info("Generating predictions...")

    # Check if model is a ThresholdOptimizer (fairlearn debiased model)
    if hasattr(model, 'estimator_'):
        logging.info("Detected ThresholdOptimizer — passing sensitive_features.")
        sensitive = df[SENSITIVE_FEATURE].copy() if SENSITIVE_FEATURE in df.columns else None
        if sensitive is not None:
            sensitive_encoded = LabelEncoder().fit_transform(sensitive.astype(str))
            predictions = model.predict(X, sensitive_features=sensitive_encoded)
            try:
                probabilities = model.predict_proba(X, sensitive_features=sensitive_encoded)[:, 1]
            except (AttributeError, TypeError):
                logging.warning("predict_proba not available, using predictions as probabilities.")
                probabilities = predictions.astype(float)
        else:
            logging.error(f"Sensitive feature '{SENSITIVE_FEATURE}' not found for ThresholdOptimizer.")
            sys.exit(1)
    else:
        predictions   = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

    # Add prediction columns to original data
    result = df.copy()
    result["PREDICTION"]       = predictions
    result["PREDICTION_PROBA"] = np.round(probabilities, 4)

    # Log summary
    default_count = (predictions == 1).sum()
    total = len(predictions)
    logging.info(f"Done. {default_count}/{total} predicted as default ({default_count/total*100:.1f}%)")

    return result


def save_predictions(result: pd.DataFrame, input_path: str):
    """Save predictions CSV to predictions/ folder."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(PREDICTIONS_DIR, f"{base_name}_predictions.csv")

    result.to_csv(output_path, index=False)
    logging.info(f"Predictions saved → {output_path}")
    return output_path


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.info("=" * 60)
    logging.info("ShiftHappens — Prediction Pipeline")
    logging.info("=" * 60)

    # Check if user provided a CSV file path
    if len(sys.argv) < 2:
        logging.error("Usage: python3 scripts/predictor.py <path_to_csv>")
        logging.error("Example: python3 scripts/predictor.py data/new_applications.csv")
        sys.exit(1)

    input_path = sys.argv[1]

    # Support both CSV and PKL files
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Read input file
    logging.info(f"Input file: {input_path}")
    if input_path.endswith(".pkl"):
        df = pd.read_pickle(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path)
    else:
        logging.error("Unsupported file format. Use CSV, PKL, or Excel.")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")

    # Load model
    model = load_model()

    # Predict
    result = predict(model, df)

    # Save
    output_path = save_predictions(result, input_path)

    # Summary
    logging.info("=" * 60)
    logging.info("Prediction Summary:")
    logging.info(f"  Total records:      {len(result)}")
    logging.info(f"  Predicted default:  {(result['PREDICTION'] == 1).sum()}")
    logging.info(f"  Predicted safe:     {(result['PREDICTION'] == 0).sum()}")
    logging.info(f"  Output saved:       {output_path}")
    logging.info("=" * 60)