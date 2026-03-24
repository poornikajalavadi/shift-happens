import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocessor import preprocess
from scripts.model_trainer import train_and_evaluate, select_best_model, save_model


@pytest.fixture
def mock_merged_df() -> pd.DataFrame:
    """
    Simulates a minimal version of application_train_merged.pkl
    with the key columns required by the model development pipeline.
    Includes numeric, categorical, and sensitive feature columns.
    """
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        'SK_ID_CURR':       range(n),
        'TARGET':           np.random.randint(0, 2, n),
        'CODE_GENDER':      np.random.choice(['M', 'F'], n),
        'AMT_INCOME_TOTAL': np.random.uniform(50000, 500000, n),
        'AMT_CREDIT':       np.random.uniform(100000, 900000, n),
        'DAYS_EMPLOYED':    np.random.uniform(-3000, -100, n),
        'DAYS_BIRTH':       np.random.uniform(-25000, -7000, n),
        'EXT_SOURCE_1':     np.random.uniform(0, 1, n),
        'EXT_SOURCE_2':     np.random.uniform(0, 1, n),
        'EXT_SOURCE_3':     np.random.uniform(0, 1, n),
    })

def test_preprocess_output_shapes(mock_merged_df):
    """
    Verifies that preprocessor returns six objects with consistent
    and correctly sized shapes across train and test splits.
    """
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(mock_merged_df)

    assert len(X_train) == len(y_train) == len(s_train)
    assert len(X_test)  == len(y_test)  == len(s_test)
    assert X_train.shape[1] == X_test.shape[1]


def test_preprocess_no_nans(mock_merged_df):
    """
    Verifies that no NaN values remain in the feature matrices
    after preprocessing. Median imputation must eliminate all missing values.
    """
    X_train, X_test, *_ = preprocess(mock_merged_df)

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum()  == 0


def test_preprocess_target_binary(mock_merged_df):
    """
    Verifies that the target vector contains only binary values (0 and 1)
    after preprocessing and type conversion.
    """
    _, _, y_train, y_test, *_ = preprocess(mock_merged_df)

    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_preprocess_sensitive_feature_removed(mock_merged_df):
    """
    Verifies that CODE_GENDER is removed from the feature matrix
    after CorrelationRemover is applied in the preprocessing pipeline.
    """
    X_train, X_test, *_ = preprocess(mock_merged_df)

    assert 'CODE_GENDER' not in X_train.columns
    assert 'CODE_GENDER' not in X_test.columns


def test_train_returns_both_models(mock_merged_df):
    """
    Verifies that train_and_evaluate returns results for both
    LogisticRegression and LightGBM candidate models.
    """
    X_train, X_test, y_train, y_test, *_ = preprocess(mock_merged_df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    assert "LogisticRegression" in results
    assert "LightGBM" in results

def test_best_model_has_highest_auc(mock_merged_df):
    """
    Verifies that select_best_model returns the model with the
    highest ROC-AUC score across all candidate models.
    """
    X_train, X_test, y_train, y_test, *_ = preprocess(mock_merged_df)
    results    = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name, _, best_metrics = select_best_model(results)

    for name, info in results.items():
        assert best_metrics["roc_auc"] >= info["metrics"]["roc_auc"]


def test_metrics_within_valid_range(mock_merged_df):
    """
    Verifies that all computed metrics fall within the valid
    range of [0.0, 1.0] for both candidate models.
    """
    X_train, X_test, y_train, y_test, *_ = preprocess(mock_merged_df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    for name, info in results.items():
        for metric, val in info["metrics"].items():
            assert 0.0 <= val <= 1.0, (
                f"Metric out of valid range — model: {name}, "
                f"metric: {metric}, value: {val}"
            )


def test_save_model_creates_file(mock_merged_df, tmp_path, monkeypatch):
    """
    Verifies that save_model serialises the trained model to disk
    and that the output file exists at the expected path.
    """
    monkeypatch.setattr("scripts.model_trainer.MODEL_OUTPUT_DIR", str(tmp_path))

    X_train, X_test, y_train, y_test, *_ = preprocess(mock_merged_df)
    results    = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name, best_model, _ = select_best_model(results)
    model_path = save_model(best_model, best_name)

    assert os.path.exists(model_path)
    assert model_path.endswith(".pkl")


def test_train_test_split_ratio(mock_merged_df):
    """
    Verifies that the train/test split ratio is approximately 80/20
    with a tolerance of 2% to account for stratification adjustments.
    """
    X_train, X_test, *_ = preprocess(mock_merged_df)
    total      = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total

    assert abs(test_ratio - 0.20) < 0.02
