import os
import sys
import logging
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# ─────────────────────────────────────────────────────────────
# Logging — outputs to both file and console
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hyperparameter_tuner.log"),
        logging.StreamHandler()
    ]
)

MLFLOW_EXPERIMENT = "ShiftHappens_Model_Development"
RANDOM_STATE      = 42

# ─────────────────────────────────────────────────────────────
# Search spaces for each model
# RandomizedSearchCV samples from these distributions
# ─────────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "LogisticRegression": {
        "C":       uniform(0.01, 10),   # Regularization strength
        "penalty": ["l2"],              # L2 regularization only (lbfgs)
        "solver":  ["lbfgs", "saga"],   # Solvers compatible with L2
    },
    "LightGBM": {
        "n_estimators":  randint(100, 600),    # Number of boosting rounds
        "learning_rate": uniform(0.01, 0.2),   # Step size shrinkage
        "num_leaves":    randint(20, 100),      # Max leaves per tree
        "max_depth":     randint(3, 10),        # Max tree depth
        "subsample":     uniform(0.6, 0.4),    # Row subsampling ratio
    },
}

def tune_model(model_name: str, base_model, X_train, y_train):
    """
    Runs RandomizedSearchCV on the given model.
    Tries 20 random combinations from PARAM_GRIDS and picks
    the one with the best cross-validated ROC-AUC score.
    Logs best params and CV score to MLflow.

    Args:
        model_name:  String key matching PARAM_GRIDS
        base_model:  Unfitted sklearn-compatible estimator
        X_train, y_train: Training data

    Returns:
        best_estimator: Refitted model with best hyperparameters
    """
    if model_name not in PARAM_GRIDS:
        logging.warning(f"No param grid for '{model_name}'. Skipping tuning.")
        return base_model

    logging.info(f"Starting RandomizedSearchCV for {model_name}...")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=20,          # Try 20 random combinations
        scoring="roc_auc",  # Optimise for ROC-AUC
        cv=3,               # 3-fold cross validation
        n_jobs=-1,          # Use all CPU cores
        random_state=RANDOM_STATE,
        verbose=1,
    )

    with mlflow.start_run(run_name=f"{model_name}_HyperparamTuning"):
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score  = search.best_score_

        # Log best params and score to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_score)
        logging.info(f"Best CV AUC: {best_score:.4f}")
        logging.info(f"Best params: {best_params}")

    return search.best_estimator_


if __name__ == "__main__":
    import pickle
    # Add parent directory to path so scripts module is found
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess
    from scripts.model_trainer import save_model

    logging.info("=" * 60)
    logging.info("ShiftHappens — Starting Hyperparameter Tuning")
    logging.info("=" * 60)

    # Step 1 — Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    # Step 2 — Load best saved model
    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Step 3 — Run RandomizedSearchCV to find best hyperparameters
    tuned_model = tune_model("LightGBM", model, X_train, y_train)

    # Step 4 — Save tuned model
    tuned_path = save_model(tuned_model, "LightGBM_tuned")

    logging.info("=" * 60)
    logging.info("Hyperparameter tuning complete!")
    logging.info(f"Tuned model saved → {tuned_path}")
    logging.info("=" * 60)
