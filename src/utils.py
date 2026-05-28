"""
src/utils.py
Shared utilities: config loading, path resolution, model versioning.
"""
import os
import yaml
import pickle
import joblib
from datetime import datetime
from src.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def save_model(model, name: str, save_dir: str = "models", versioned: bool = True):
    """
    Save a model using joblib with optional versioning.

    Versioning appends a timestamp so old models are never overwritten.
    Example: models/xgboost_20240601_1423.pkl

    Args:
        model: Trained sklearn/XGBoost model
        name: Base name (e.g. 'xgboost')
        save_dir: Directory to save into
        versioned: If True, append timestamp to filename
    """
    os.makedirs(save_dir, exist_ok=True)

    if versioned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{name}_{timestamp}.pkl"
    else:
        filename = f"{name}.pkl"

    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved -> {path}")
    return path


def load_model(path: str):
    """Load a joblib-saved model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def get_latest_model(save_dir: str = "models", prefix: str = "xgboost") -> str:
    """
    Find the most recently saved model file matching a prefix.

    Returns the full path to the latest versioned model.
    """
    files = [
        f for f in os.listdir(save_dir)
        if f.startswith(prefix) and f.endswith(".pkl")
    ]
    if not files:
        raise FileNotFoundError(f"No model found with prefix '{prefix}' in {save_dir}")

    # Sort by timestamp embedded in filename
    files.sort(reverse=True)
    latest = os.path.join(save_dir, files[0])
    logger.info(f"Latest model: {latest}")
    return latest


def ensure_dirs(config: dict):
    """Create all required output directories if they don't exist."""
    dirs = [
        config["outputs"]["plots_dir"],
        config["outputs"]["reports_dir"],
        config["outputs"]["logs_dir"],
        config["models"]["save_dir"],
        "data/processed",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.debug("All output directories verified.")
