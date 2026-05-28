"""
src/preprocess.py
Full preprocessing pipeline:
  - Cleaning
  - Encoding
  - Feature engineering
  - Train/test split
  - Scaling
  - SMOTE (on train only)
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.utils import load_config
from src.data_loader import load_raw_data, get_feature_types

logger = get_logger(__name__)


def clean(df: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    """Drop ID columns, impute TotalCharges, encode target."""
    df = df.copy()

    # Drop non-feature columns
    df.drop(columns=feature_types["drop"], inplace=True, errors="ignore")

    # Impute TotalCharges NaN with median
    median_tc = df["TotalCharges"].median()
    null_count = df["TotalCharges"].isnull().sum()
    df["TotalCharges"].fillna(median_tc, inplace=True)
    logger.info(f"Imputed {null_count} TotalCharges NaNs with median ${median_tc:.2f}")

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    logger.info("Target encoded: Yes->1, No->0")

    return df


def encode(df: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    """Encode binary and categorical features."""
    df = df.copy()

    # Binary Yes/No columns
    binary_map = {
        "Yes": 1, "No": 0,
        "No phone service": 0,
        "No internet service": 0
    }
    for col in feature_types["binary"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # Gender
    df["gender"] = (df["gender"] == "Male").astype(int)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=feature_types["categorical"], drop_first=False)

    # Convert all bool columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    logger.info(f"Encoding complete — shape after encoding: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features."""
    df = df.copy()

    # Avg charge per month (captures early vs long-term customer value)
    df["ChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Interaction: high charges + short tenure = high risk
    df["HighRiskFlag"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].median()) &
        (df["tenure"] < 12)
    ).astype(int)

    logger.info("Feature engineering: added ChargesPerMonth, HighRiskFlag")
    return df


def split_and_scale(df: pd.DataFrame, config: dict):
    """
    Train/test split → scale numericals → SMOTE on train.

    SMOTE is applied ONLY to training data.
    Test set remains the original distribution (no data leakage).

    Returns:
        X_train_res, X_test, y_train_res, y_test, feature_names, scaler
    """
    target = config["preprocessing"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = list(X.columns)

    # Train/test split
    test_size    = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Scale numerical features
    num_cols = config["preprocessing"]["numerical_columns"] + ["ChargesPerMonth"]
    num_cols = [c for c in num_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    logger.info("StandardScaler fitted on train, applied to test")

    # SMOTE — only on training data
    before = dict(pd.Series(y_train).value_counts())
    smote  = SMOTE(
        random_state=config["smote"]["random_state"],
        k_neighbors=config["smote"]["k_neighbors"]
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    after = dict(pd.Series(y_train_res).value_counts())
    logger.info(f"SMOTE — Before: {before} | After: {after}")

    return X_train_res, X_test, y_train_res, y_test, feature_names, scaler


def run_preprocessing(config: dict):
    """Full preprocessing pipeline — returns all splits + artifacts."""
    logger.info("=" * 50)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 50)

    feature_types = get_feature_types(config)

    df = load_raw_data(config["data"]["raw_path"])
    df = clean(df, feature_types)
    df = encode(df, feature_types)
    df = engineer_features(df)

    # Save processed data
    processed_path = config["data"]["processed_path"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved -> {processed_path}")

    X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale(df, config)

    # Save scaler and feature names
    save_dir = config["models"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.pkl"))
    logger.info("Scaler and feature names saved to models/")

    logger.info("PREPROCESSING COMPLETE")
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, features, scaler = run_preprocessing(config)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Features ({len(features)}): {features}")
