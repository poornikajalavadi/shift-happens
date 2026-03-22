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

TARGET_COL        = "TARGET"
SENSITIVE_FEATURE = "CODE_GENDER"
DROP_COLS         = ["SK_ID_CURR", "TARGET"]
TEST_SIZE         = 0.2
RANDOM_STATE      = 42


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Drop rows with missing TARGET
      2. Label-encode all categorical columns
      3. Median-fill remaining NaNs
      4. Apply Fairlearn CorrelationRemover to debias features w.r.t. CODE_GENDER
      5. Stratified train/test split

    Returns:
        X_train, X_test, y_train, y_test,
        sensitive_train, sensitive_test  (encoded gender arrays)
    """
    logging.info("Starting preprocessing...")
    df = df.copy()

    # 1. Drop rows where TARGET is missing
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    logging.info(f"Dropped {before - len(df)} rows with missing TARGET.")

    y = df[TARGET_COL].astype(int)

    # 2. Preserve sensitive feature before we remove it from X
    sensitive_raw = df[SENSITIVE_FEATURE].copy() if SENSITIVE_FEATURE in df.columns else None

    # 3. Build feature matrix — drop ID and target
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 4. Label-encode all object/category columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 5. Fill remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))
    logging.info(f"Feature matrix shape after encoding & imputation: {X.shape}")


    # 6. Fairlearn CorrelationRemover — removes linear correlation between
    #    features and sensitive attribute (gender) as a pre-processing
    #    bias mitigation step.
    if sensitive_raw is not None and SENSITIVE_FEATURE in X.columns:
        logging.info(f"Applying CorrelationRemover for '{SENSITIVE_FEATURE}'...")
        sensitive_encoded = LabelEncoder().fit_transform(sensitive_raw.astype(str))
        cr = CorrelationRemover(sensitive_feature_ids=[SENSITIVE_FEATURE])
        X_arr = cr.fit_transform(X)
        remaining_cols = [c for c in X.columns if c != SENSITIVE_FEATURE]
        X = pd.DataFrame(X_arr, columns=remaining_cols, index=X.index)
        logging.info("CorrelationRemover applied. Sensitive feature removed from X.")
    else:
        logging.warning(f"'{SENSITIVE_FEATURE}' not found — skipping CorrelationRemover.")
        sensitive_encoded = np.zeros(len(y))

    # 7. Stratified train/test split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logging.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logging.info(f"Target split — Train: {dict(y_train.value_counts())} | Test: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test, s_train, s_test
