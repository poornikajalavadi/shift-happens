# ShiftHappens - Model Development

## Overview
This directory contains the end-to-end model development pipeline for **ShiftHappens**, a lightweight MLOps monitoring platform for SMEs. Using the Home Credit Default Risk dataset, we train, validate, and evaluate machine learning models to simulate a production credit scoring system. The pipeline covers model training, hyperparameter tuning, validation, bias detection, and registry push — all orchestrated via Apache Airflow.

## Repository Structure
* `scripts/`: Modular Python scripts for preprocessing, training, validation, bias detection, and registry push.
* `dags/`: Apache Airflow DAG definition (`model_pipeline_dag.py`) that orchestrates the full pipeline.
* `tests/`: Unit tests using `pytest` to validate preprocessing and model training logic.
* `reports/`: Auto-generated charts and reports from model runs (confusion matrices, ROC curves, bias charts).
* `models/`: Saved model `.pkl` files generated at runtime.
* `logs/`: Execution logs for every script.
* `.github/workflows/`: GitHub Actions CI/CD pipeline for automated model training and validation.

## Pipeline Architecture
The model development pipeline follows these steps in order:

```
preprocessor.py          → Encode + impute + Fairlearn CorrelationRemover
      ↓
model_trainer.py         → Train Logistic Regression + LightGBM → Select best by AUC
      ↓
hyperparameter_tuner.py  → RandomizedSearchCV on best model
      ↓
model_validator.py       → Validate on hold-out set → ROC/PR curves
      ↓
bias_detector.py         → Fairlearn MetricFrame slicing by gender
      ↓
model_selection.py       → Final gate — validation + bias results combined
      ↓
registry_push.py         → Push to GCP with rollback mechanism
```

## Environment Setup & Reproducibility
Ensure Python 3.10+ is installed and follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/semwalhritvik/shift-happens.git
cd shift-happens/Model-Development
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the Data Pipeline has been run first so this file exists:
```
data/processed/application_train_merged.pkl
```

## Running the Pipeline

### Option A — Run scripts individually (recommended for development)
```bash
python3 scripts/preprocessor.py
python3 scripts/model_trainer.py
python3 scripts/hyperparameter_tuner.py
python3 scripts/model_validator.py
python3 scripts/bias_detector.py
python3 scripts/model_selection.py
python3 scripts/registry_push.py
```

### Option B — Trigger Airflow DAG (recommended for production)
```bash
airflow dags trigger shifthappens_model_pipeline
```
Navigate to `http://localhost:8080` to monitor progress.

### Option C — CI/CD via GitHub Actions
Push any change to `scripts/` or `dags/` and the full pipeline runs automatically.

## Model Training & Selection
We trained two candidate models and compared them across all key metrics:

* **Logistic Regression** — interpretable baseline model with `class_weight='balanced'` to handle the 8% positive class imbalance.
* **LightGBM** — gradient boosting model with 300 estimators and learning rate 0.05, also with `class_weight='balanced'`.

The model with the highest **ROC-AUC** on the hold-out test set is automatically selected as the production model. All runs are tracked in **MLflow** with hyperparameters, metrics, and model artifacts logged per run.

### Model Comparison Results

![Model Comparison](reports/model_comparison.png)

| Model | ROC-AUC | F1 Score | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **LightGBM** ✅ Winner | **0.7779** | **0.2897** | **0.7335** | **0.1846** | **0.6733** |
| Logistic Regression | 0.6785 | 0.2169 | 0.6165 | — | — |

**LightGBM was selected** as the best model based on highest ROC-AUC score.

## Confusion Matrices

### LightGBM
![Confusion Matrix LightGBM](reports/confusion_matrix_LightGBM.png)

### Logistic Regression
![Confusion Matrix Logistic Regression](reports/confusion_matrix_LogisticRegression.png)

## Hyperparameter Tuning
We used `RandomizedSearchCV` with 20 iterations and 3-fold cross-validation to optimise LightGBM hyperparameters. The search space covered:

* `n_estimators`: 100–600
* `learning_rate`: 0.01–0.20
* `num_leaves`: 20–100
* `max_depth`: 3–10
* `subsample`: 0.6–1.0

**Best parameters found:**
* `learning_rate`: 0.0497
* `max_depth`: 9
* `n_estimators`: 238
* `num_leaves`: 27
* `subsample`: 0.8916
* **Best CV ROC-AUC: 0.7729**

All tuning runs are tracked in MLflow under the `ShiftHappens_Model_Development` experiment.

## Model Validation
The tuned LightGBM model was validated on a **hold-out test set (20% of data)** that was never used during training. Validation metrics and curves are saved to `reports/`.

### ROC & Precision-Recall Curves
![ROC and PR Curves](reports/roc_pr_curves_LightGBM.png)

### Validation Results
| Metric | Score | Threshold | Status |
|---|---|---|---|
| ROC-AUC | 0.7779 | ≥ 0.70 | ✅ PASSED |
| F1 Score | 0.2897 | ≥ 0.25 | ✅ PASSED |
| Accuracy | 0.7335 | ≥ 0.60 | ✅ PASSED |

The F1 score of 0.29 is realistic for this dataset given the severe class imbalance (only 8% of loans default). ROC-AUC is the primary metric as it handles imbalanced classes correctly.

## Bias Detection & Mitigation
We implement a two-stage fairness strategy following Fairlearn best practices.

### Pre-processing Mitigation — CorrelationRemover
Before training, `CorrelationRemover` in `preprocessor.py` mathematically removes the linear correlation between all features and `CODE_GENDER`. This ensures the model cannot learn gender-based patterns from the data.

### Post-training Detection — MetricFrame Slicing
After training, `bias_detector.py` uses Fairlearn's `MetricFrame` to slice the test data by gender groups and evaluate accuracy, false positive rate, and true positive rate per group.

### Bias Detection Results

![Bias by Group](reports/bias_by_group_LightGBM.png)

| Group | Accuracy | False Positive Rate | True Positive Rate |
|---|---|---|---|
| Group 0 (Male) | 0.7743 | 0.2133 | 0.6093 |
| Group 1 (Female) | 0.6545 | 0.3573 | 0.7586 |
| Group 2 (XNA) | 0.5000 | 0.5000 | 0.0000 |

**Bias was detected** across gender groups with maximum disparity of 0.2743 in accuracy. This is expected even after CorrelationRemover as it only removes linear correlations.

### Mitigation Strategy & Trade-offs
* **Applied:** Fairlearn `CorrelationRemover` at preprocessing — removes linear gender correlation before training
* **Recommended next step:** `ThresholdOptimizer` post-processing — adjusts decision thresholds per group to equalise outcomes
* **Trade-off:** Applying stricter fairness constraints may slightly reduce overall accuracy but improves equity across demographic groups

### Final Model Selection Gate
`model_selection.py` combines both validation and bias results before approving deployment:

* Validation PASSED ✅ + Bias detected ⚠️ → **APPROVED WITH WARNING**
* Mitigation documented and ThresholdOptimizer recommended as next step

## Experiment Tracking
All model runs are tracked using **MLflow**. To view the dashboard:
```bash
mlflow ui
# Open → http://localhost:5000
```

Each run logs:
* Hyperparameters
* Performance metrics (AUC, F1, accuracy, precision, recall)
* Model artifacts (confusion matrices, ROC curves)
* Model binary for version control

## Unit Testing
We follow Test-Driven Development (TDD) using `pytest`. The `tests/` directory contains unit tests validating preprocessing and model training logic.

```bash
pytest tests/ -v
```

Tests cover:
* Preprocessing output shapes are consistent
* No NaN values after preprocessing
* Target column contains only 0 and 1
* Both models are trained and returned
* Best model always has the highest ROC-AUC
* All metrics are within valid range (0 to 1)

## CI/CD Pipeline
GitHub Actions automatically triggers the full model pipeline on every push to `scripts/` or `dags/`. The pipeline runs in this order:

1. Install dependencies
2. Authenticate to GCP
3. Run unit tests
4. Train models
5. Tune hyperparameters
6. Validate model
7. Run bias detection
8. Run SHAP sensitivity analysis
9. Push to GCP registry
10. Upload reports as artifacts

If any step fails, a notification is posted automatically.

## GCP Model Registry
Once the model passes validation and bias checks, it is pushed to **Google Cloud Storage** as the model registry. A rollback mechanism compares the new model's AUC against the previous production model — if the new model performs worse, the push is blocked and the existing model stays in production.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export GCP_PROJECT=shifthappens-project
export GCS_BUCKET=shifthappens-model-registry
python3 scripts/registry_push.py
```

## Reports Generated
All saved to `reports/` at runtime:

| File | Description |
|---|---|
| `model_comparison.png` | Bar chart comparing LR vs LightGBM across all metrics |
| `confusion_matrix_LightGBM.png` | LightGBM predictions vs actuals |
| `confusion_matrix_LogisticRegression.png` | LR predictions vs actuals |
| `roc_pr_curves_LightGBM.png` | ROC and Precision-Recall curves |
| `classification_report_LightGBM.txt` | Full precision/recall/F1 per class |
| `bias_by_group_LightGBM.png` | Fairness metrics by gender group |
| `bias_report_LightGBM.txt` | Full disparity numbers across groups |


## Model Interpretability (SHAP Analysis)
To ensure our model's decisions are transparent, fair, and easily explainable to stakeholders, we implemented SHAP (SHapley Additive exPlanations) using the highly optimized TreeExplainer.

1. Global Feature Importance
   
![Global SHAP summary plot](assets/Figure_1.png)

The global summary plot reveals the top drivers of default risk across the entire dataset:

External Sources: Normalized scores from external credit data providers (EXT_SOURCE_1, 2, and 3) dominate the model's decision-making process. Low external scores strongly push the model toward predicting a default.

Historical Behaviors: Engineered features from our Airflow pipeline, such as INSTALLMENTS_AMT_PAYMENT_MIN, play a significant role, proving the value of our upstream relational data aggregation.

1. Categorical Risk Drivers: Occupation Type
   
![OCCUPATION_TYPR Dependence Plot](assets/Figure_2.png)

By utilizing SHAP dependence plots, we opened the "black box" to unpack the exact risk assigned to non-ordinal categories:

High-Risk Roles: The model identified "Low-skill Laborers" and "Drivers" as having a heavily increased historical risk of default.

Low-Risk Roles: Occupations like "Accountants" and "Core staff" consistently lower the predicted risk score.

The Power of 'Missing': Applicants who declined to state their occupation actually lean slightly toward lower risk. This validates our pipeline decision to explicitly label missing data rather than simply dropping or blindly imputing it.

1. Continuous Risk Discovery: Age Dynamics
   
![DAYS_BIRTH Dependence Plot](assets/Figure_3.png)

The dependence plot for DAYS_BIRTH showcases LightGBM's ability to capture complex, non-linear relationships that traditional linear models miss:

The Mid-Life Plateau: The highest risk concentration occurs among applicants in their late 20s to late 40s.

The Retirement Spike: The model independently discovered a sharp, sudden spike in default risk at approximately 65 years of age (around -24,000 days), effectively pinpointing the financial instability that can accompany the transition to fixed retirement incomes.