import sys
import os
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_loader       import load_data
from scripts.preprocessor      import preprocess
from scripts.model_trainer     import train_and_evaluate, select_best_model, save_model, plot_model_comparison
from scripts.hyperparameter_tuner import tune_model
from scripts.model_validator   import validate_model
from scripts.sensitivity_analysis import run_shap_analysis
from scripts.bias_detector     import detect_bias
from scripts.registry_push     import push_model_to_registry

default_args = {
    'owner':           'shifthappens_mlops',
    'depends_on_past': False,
    'start_date':      datetime(2025, 1, 1),
    'email':           ['alerts@shifthappens.ai'],
    'email_on_failure': True,
    'email_on_retry':  False,
    'retries':         1,
    'retry_delay':     timedelta(minutes=5),
}

dag = DAG(
    'shifthappens_model_pipeline',
    default_args=default_args,
    description='ShiftHappens end-to-end model training, validation, bias detection & registry push.',
    schedule='@weekly',
    catchup=False,
)

# ─────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────

def task_load_and_preprocess(**kwargs):
    df = load_data()
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(df)
    # Pickle splits to XCom-friendly temp files
    splits = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        's_train': s_train, 's_test': s_test
    }
    splits_path = 'data/processed/model_splits.pkl'
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
    return splits_path


def task_train(**kwargs):
    ti          = kwargs['ti']
    splits_path = ti.xcom_pull(task_ids='load_and_preprocess')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    results = train_and_evaluate(
        splits['X_train'], splits['X_test'],
        splits['y_train'], splits['y_test']
    )
    plot_model_comparison(results)
    best_name, best_model, best_metrics = select_best_model(results)
    model_path = save_model(best_model, best_name)
    return {'model_path': model_path, 'model_name': best_name, 'metrics': best_metrics}


def task_tune(**kwargs):
    ti         = kwargs['ti']
    train_info = ti.xcom_pull(task_ids='train_models')
    splits_path = ti.xcom_pull(task_ids='load_and_preprocess')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    with open(train_info['model_path'], 'rb') as f:
        model = pickle.load(f)
    tuned_model = tune_model(train_info['model_name'], model, splits['X_train'], splits['y_train'])
    tuned_path  = save_model(tuned_model, f"{train_info['model_name']}_tuned")
    return {'model_path': tuned_path, 'model_name': train_info['model_name']}

def task_validate(**kwargs):
    ti          = kwargs['ti']
    tune_info   = ti.xcom_pull(task_ids='tune_hyperparameters')
    splits_path = ti.xcom_pull(task_ids='load_and_preprocess')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    with open(tune_info['model_path'], 'rb') as f:
        model = pickle.load(f)
    passed = validate_model(model, splits['X_test'], splits['y_test'], tune_info['model_name'])
    if not passed:
        raise ValueError("Model validation failed. Pipeline halted.")
    return tune_info['model_path']


def task_bias(**kwargs):
    ti          = kwargs['ti']
    model_path  = ti.xcom_pull(task_ids='validate_model')
    splits_path = ti.xcom_pull(task_ids='load_and_preprocess')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    passed = detect_bias(model, splits['X_test'], splits['y_test'],
                         splits['s_test'], "tuned_model")
    if not passed:
        logging.warning("Bias detected but pipeline continues — CorrelationRemover was applied.")
    return model_path


def task_shap(**kwargs):
    ti          = kwargs['ti']
    model_path  = ti.xcom_pull(task_ids='bias_detection')
    splits_path = ti.xcom_pull(task_ids='load_and_preprocess')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    run_shap_analysis(model, splits['X_test'], "tuned_model")
    return model_path


def task_push(**kwargs):
    ti          = kwargs['ti']
    model_path  = ti.xcom_pull(task_ids='shap_analysis')
    tune_info   = ti.xcom_pull(task_ids='tune_hyperparameters')
    train_info  = ti.xcom_pull(task_ids='train_models')
    pushed = push_model_to_registry(
        model_path  = model_path,
        model_name  = tune_info['model_name'],
        metrics     = train_info['metrics']
    )
    if not pushed:
        raise ValueError("Registry push failed or rollback triggered.")

# ─────────────────────────────────────────────
# Operator Definitions
# ─────────────────────────────────────────────

op_load = PythonOperator(
    task_id='load_and_preprocess', python_callable=task_load_and_preprocess, dag=dag)

op_train = PythonOperator(
    task_id='train_models', python_callable=task_train, dag=dag)

op_tune = PythonOperator(
    task_id='tune_hyperparameters', python_callable=task_tune, dag=dag)

op_validate = PythonOperator(
    task_id='validate_model', python_callable=task_validate, dag=dag)

op_bias = PythonOperator(
    task_id='bias_detection', python_callable=task_bias, dag=dag)

op_shap = PythonOperator(
    task_id='shap_analysis', python_callable=task_shap, dag=dag)

op_push = PythonOperator(
    task_id='push_to_registry', python_callable=task_push, dag=dag)

# ─────────────────────────────────────────────
# DAG Dependencies
# ─────────────────────────────────────────────
op_load >> op_train >> op_tune >> op_validate >> op_bias >> op_shap >> op_push
