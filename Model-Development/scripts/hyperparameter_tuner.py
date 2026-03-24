import os
import sys
import logging
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

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

# Hyperparameter search distributions for each candidate model.
# RandomizedSearchCV samples from these distributions across n_iter iterations.
PARAM_GRIDS = {
    "LogisticRegression": {
        "C":       uniform(0.01, 10),   # Inverse regularisation strength
        "penalty": ["l2"],              # L2 regularisation (compatible with lbfgs/saga)
        "solver":  ["lbfgs", "saga"],
    },
    "LightGBM": {
        "n_estimators":  randint(100, 600),   # Number of boosting rounds
        "learning_rate": uniform(0.01, 0.2),  # Shrinkage rate applied per iteration
        "num_leaves":    randint(20, 100),     # Maximum number of leaves per tree
        "max_depth":     randint(3, 10),       # Maximum tree depth
        "subsample":     uniform(0.6, 0.4),   # Row subsampling ratio per iteration
    },
}


def tune_model(model_name: str, base_model, X_train, y_train):
    """
    Executes RandomizedSearchCV hyperparameter optimisation for the given model.

    Evaluates n_iter=20 random parameter combinations using 3-fold
    cross-validation scored by ROC-AUC. Best parameters and CV score
    are logged to MLflow.

    Args:
        model_name:  Key matching an entry in PARAM_GRIDS.
        base_model:  Unfitted sklearn-compatible estimator.
        X_train:     Training feature matrix.
        y_train:     Training target vector.

    Returns:
        Refitted estimator with best identified hyperparameters.
    """
    if model_name not in PARAM_GRIDS:
        logging.warning(f"No parameter grid defined for '{model_name}'. Tuning skipped.")
        return base_model

    logging.info(f"Initiating RandomizedSearchCV for: {model_name}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=20,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    with mlflow.start_run(run_name=f"{model_name}_HyperparamTuning"):
        search.fit(X_train, y_train)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_roc_auc", search.best_score_)
        logging.info(f"Best CV AUC: {search.best_score_:.4f}")
        logging.info(f"Best parameters: {search.best_params_}")

    return search.best_estimator_


if __name__ == "__main__":
    import pickle
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scripts.data_loader import load_data
    from scripts.preprocessor import preprocess
    from scripts.model_trainer import save_model

    logging.info("=" * 60)
    logging.info("ShiftHappens — Hyperparameter Tuning")
    logging.info("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)

    model_path = "models/best_model_LightGBM.pkl"
    logging.info(f"Loading base model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    tuned_model = tune_model("LightGBM", model, X_train, y_train)
    tuned_path  = save_model(tuned_model, "LightGBM_tuned")

    logging.info("=" * 60)
    logging.info("Hyperparameter tuning complete.")
    logging.info(f"Tuned model saved: {tuned_path}")
    logging.info("=" * 60)
