"""
src/data_loader.py
Data ingestion: load raw CSV, validate schema, basic fix-ups.
No business logic — just getting data in cleanly.
"""
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV, validate schema, fix known data issues.

    Args:
        path: Path to the raw CSV file

    Returns:
        pd.DataFrame with basic fixes applied
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    # ── Schema validation ────────────────────────────────────
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    logger.info(f"Schema validated — {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Fix TotalCharges stored as string with blanks ────────
    original_dtype = df["TotalCharges"].dtype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    fixed_count = df["TotalCharges"].isnull().sum()
    if fixed_count:
        logger.warning(f"TotalCharges: {fixed_count} blank strings -> NaN (will be imputed)")

    # ── Basic stats ──────────────────────────────────────────
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    logger.info(f"Churn rate: {churn_rate:.1f}%  |  Class imbalance detected")
    logger.info(f"Tenure range: {df['tenure'].min()}–{df['tenure'].max()} months")
    logger.info(f"Monthly charges: ${df['MonthlyCharges'].min():.0f}–${df['MonthlyCharges'].max():.0f}")

    return df


def get_feature_types(config: dict) -> dict:
    """Return feature type mappings from config."""
    return {
        "target"      : config["preprocessing"]["target_column"],
        "drop"        : config["preprocessing"]["drop_columns"],
        "binary"      : config["preprocessing"]["binary_columns"],
        "categorical" : config["preprocessing"]["categorical_columns"],
        "numerical"   : config["preprocessing"]["numerical_columns"],
    }


if __name__ == "__main__":
    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
