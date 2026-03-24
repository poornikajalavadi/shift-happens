import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fairlearn.preprocessing import CorrelationRemover

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/preprocessor.log"),
        logging.StreamHandler()
    ]
)

# Feature column identifiers
TARGET_COL        = "TARGET"
SENSITIVE_FEATURE = "CODE_GENDER"
DROP_COLS         = ["SK_ID_CURR", "TARGET"]
TEST_SIZE         = 0.2
RANDOM_STATE      = 42


def preprocess(df: pd.DataFrame):
    """
    Executes the full preprocessing pipeline prior to model training.

    Steps:
        1. Remove rows with missing TARGET values.
        2. Label-encode all categorical columns to numeric representations.
        3. Impute remaining NaN values with column-wise median.
        4. Apply Fairlearn CorrelationRemover to eliminate linear correlation
           between input features and the sensitive attribute CODE_GENDER.
        5. Perform stratified train/test split (80/20).

    Returns:
        X_train, X_test, y_train, y_test,
        sensitive_train, sensitive_test
    """
    logging.info("Initiating preprocessing pipeline.")
    df = df.copy()

    # Remove records where the target label is absent
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logging.info(f"Removed {before - len(df)} records with missing TARGET.")

    y = df[TARGET_COL].astype(int)

    # Retain sensitive feature array prior to feature matrix construction
    sensitive_raw = df[SENSITIVE_FEATURE].copy() if SENSITIVE_FEATURE in df.columns else None

    # Construct feature matrix excluding identifier and target columns
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Encode categorical columns to integer labels
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute missing values using column median
    X = X.fillna(X.median(numeric_only=True))
    logging.info(f"Feature matrix shape post encoding and imputation: {X.shape}")

    # Apply CorrelationRemover to reduce linear dependence between features
    # and the sensitive attribute as a pre-processing fairness intervention
    if sensitive_raw is not None and SENSITIVE_FEATURE in X.columns:
        logging.info(f"Applying CorrelationRemover for sensitive feature: {SENSITIVE_FEATURE}")
        sensitive_encoded = LabelEncoder().fit_transform(sensitive_raw.astype(str))
        cr    = CorrelationRemover(sensitive_feature_ids=[SENSITIVE_FEATURE])
        X_arr = cr.fit_transform(X)
        remaining_cols = [c for c in X.columns if c != SENSITIVE_FEATURE]
        X = pd.DataFrame(X_arr, columns=remaining_cols, index=X.index)
        logging.info("CorrelationRemover applied. Sensitive feature removed from feature matrix.")
    else:
        logging.warning(f"Sensitive feature '{SENSITIVE_FEATURE}' not found. CorrelationRemover skipped.")
        sensitive_encoded = np.zeros(len(y))

    # Stratified split preserving TARGET class distribution across train and test sets
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logging.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    logging.info(f"Class distribution — Train: {dict(y_train.value_counts())} | Test: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test, s_train, s_test
