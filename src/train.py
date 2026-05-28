"""
src/train.py
Model training pipeline:
  - Logistic Regression (baseline)
  - Random Forest (intermediate)
  - XGBoost default
  - XGBoost tuned with Optuna
"""
import os
import optuna
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from src.logger import get_logger
from src.utils import load_config, save_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)


def train_logistic_regression(X_train, y_train, config: dict):
    logger.info("Training Logistic Regression (baseline)...")
    lr_cfg = config["models"]["logistic_regression"]
    model = LogisticRegression(
        max_iter=lr_cfg["max_iter"],
        random_state=lr_cfg["random_state"]
    )
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
    logger.info(f"LR — CV F1 (5-fold): {cv_score*100:.2f}%")
    return model


def train_random_forest(X_train, y_train, config: dict):
    logger.info("Training Random Forest...")
    rf_cfg = config["models"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        random_state=rf_cfg["random_state"],
        n_jobs=rf_cfg["n_jobs"]
    )
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
    logger.info(f"RF  — CV F1 (5-fold): {cv_score*100:.2f}%")
    return model


def train_xgboost_default(X_train, y_train, config: dict):
    logger.info("Training XGBoost (default params)...")
    xgb_cfg = config["models"]["xgboost"]
    model = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        random_state=xgb_cfg["random_state"],
        eval_metric=xgb_cfg["eval_metric"],
        verbosity=xgb_cfg["verbosity"]
    )
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
    logger.info(f"XGB — CV F1 (5-fold): {cv_score*100:.2f}%")
    return model


def tune_xgboost_optuna(X_train, y_train, config: dict):
    """
    Tune XGBoost using Optuna Bayesian optimization.
    Searches over learning_rate, max_depth, n_estimators,
    subsample, colsample_bytree, min_child_weight.
    """
    logger.info("Tuning XGBoost with Optuna...")
    optuna_cfg = config["optuna"]

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"        : trial.suggest_int("max_depth", 3, 8),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "gamma"            : trial.suggest_float("gamma", 0, 5),
            "random_state"     : config["data"]["random_state"],
            "eval_metric"      : "logloss",
            "verbosity"        : 0,
        }
        model = XGBClassifier(**params)
        score = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring="f1", n_jobs=-1
        ).mean()
        return score

    study = optuna.create_study(direction=optuna_cfg["direction"])
    study.optimize(
        objective,
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg["timeout"],
        show_progress_bar=False
    )

    best_params = study.best_params
    best_score  = study.best_value
    logger.info(f"Optuna best F1: {best_score*100:.2f}%")
    logger.info(f"Best params: {best_params}")

    # Retrain on full training data with best params
    best_model = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        verbosity=0
    )
    best_model.fit(X_train, y_train)
    logger.info("XGBoost retrained on full train set with best params")

    return best_model, best_params, study


def run_training(X_train, X_test, y_train, y_test, config: dict) -> dict:
    """
    Run all model training stages and return a dict of trained models.
    """
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 50)

    models = {}

    models["logistic_regression"] = train_logistic_regression(X_train, y_train, config)
    models["random_forest"]       = train_random_forest(X_train, y_train, config)
    models["xgboost_default"]     = train_xgboost_default(X_train, y_train, config)

    models["xgboost_tuned"], best_params, study = tune_xgboost_optuna(
        X_train, y_train, config
    )

    # Save all models with versioning
    save_dir = config["models"]["save_dir"]
    for name, model in models.items():
        save_model(model, name, save_dir=save_dir, versioned=True)

    # Save best model without version tag (for app to load)
    joblib.dump(models["xgboost_tuned"], os.path.join(save_dir, "best_model.pkl"))
    logger.info("Best model saved -> models/best_model.pkl")

    # Save Optuna study
    joblib.dump(study, os.path.join(save_dir, "optuna_study.pkl"))
    logger.info("Optuna study saved -> models/optuna_study.pkl")

    logger.info("MODEL TRAINING COMPLETE")
    return models


if __name__ == "__main__":
    from src.preprocess import run_preprocessing
    config   = load_config()
    X_train, X_test, y_train, y_test, features, scaler = run_preprocessing(config)
    models   = run_training(X_train, X_test, y_train, y_test, config)
    print("Models trained:", list(models.keys()))
