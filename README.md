# ShiftHappens: MLOps Monitoring for SMEs

## Project Objective
ShiftHappens is a lightweight, serverless MLOps monitoring platform designed for small and medium enterprises (SMEs) and AI consultancies. It acts as an "early warning system" for deployed machine learning models, detecting data drift and performance degradation before it impacts client relationships. 

It answers three core questions for deployed models:
1. **Is it broken?** (Health monitoring)
2. **Who broke it?** (Drift detection)
3. **Can we fix it?** (One-click remediation)

## Current Status: Sprint 2 (Data Pipeline)
We are currently in the Data Ingestion & Preprocessing phase. We are utilizing the **Home Credit Default Risk** dataset to simulate a production credit scoring model. 

### Phase 1 Deliverable: Automated Airflow Pipeline
We have built a fully automated, test-driven ETL pipeline. 
* **Location:** All pipeline code, DAGs, Pytest modules, and execution logs are located in the [`Data-Pipeline/`](./Data-Pipeline) directory.
* **Key Features:** Features include DVC integration for data versioning, parallelized Airflow tasks for optimized feature engineering across 8 relational tables, targeted anomaly treatment (e.g., handling erroneous `DAYS_EMPLOYED` records), and Fairlearn integration for demographic bias mitigation.

Please navigate to the `Data-Pipeline/README.md` for detailed instructions on reproducing the Airflow environment, viewing the Gantt chart optimizations, and running the unit tests.

## Phase 2 Deliverable: Model Development & CI/CD Pipeline

We have built an end-to-end model training, validation, and deployment pipeline with automated CI/CD.

- **Location:** All scripts, tests, and configuration are located in the `Model-Development/` directory.
- **Key Features:** Trains and compares Logistic Regression vs LightGBM, selects best model by ROC-AUC, tunes hyperparameters with RandomizedSearchCV, validates against performance thresholds, detects bias using Fairlearn MetricFrame, and pushes to GCS model registry with automatic rollback protection.
- **CI/CD:** Google Cloud Build trigger connected to GitHub automatically runs the full pipeline on every push to `main`. Training data is stored in GCS and downloaded at build time. Email notifications alert on pipeline failures.

Please navigate to the `Model-Development/README.md` for detailed instructions on reproducing the pipeline, viewing model comparison results, bias reports, and running the unit tests.
