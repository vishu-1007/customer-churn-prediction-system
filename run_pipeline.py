"""
run_pipeline.py
One command to run the entire training pipeline end-to-end.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config configs/config.yaml
"""
import argparse
import sys
import os
import time

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import get_logger
from src.utils import load_config, ensure_dirs
from src.data_loader import load_raw_data
from src.preprocess import run_preprocessing
from src.train import run_training
from src.evaluate import run_evaluation

logger = get_logger("pipeline")


def run(config_path: str = "configs/config.yaml"):
    start = time.time()

    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Load config ──────────────────────────────────────────
    config = load_config(config_path)
    ensure_dirs(config)
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")

    # ── Stage 1: Preprocessing ───────────────────────────────
    logger.info("\n[STAGE 1/3] PREPROCESSING")
    X_train, X_test, y_train, y_test, features, scaler = run_preprocessing(config)

    # ── Stage 2: Training ────────────────────────────────────
    logger.info("\n[STAGE 2/3] TRAINING")
    models = run_training(X_train, X_test, y_train, y_test, config)

    # ── Stage 3: Evaluation ──────────────────────────────────
    logger.info("\n[STAGE 3/3] EVALUATION")
    metrics, explainer = run_evaluation(models, X_test, y_test, features, config)

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start
    best = max(metrics, key=lambda m: m["F1"])

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time     : {elapsed/60:.1f} minutes")
    logger.info(f"Best model     : {best['Model']}")
    logger.info(f"Best F1        : {best['F1']}%")
    logger.info(f"Best ROC-AUC   : {best['ROC-AUC']}%")
    logger.info("Artifacts saved:")
    logger.info("  models/best_model.pkl")
    logger.info("  models/scaler.pkl")
    logger.info("  models/feature_names.pkl")
    logger.info("  models/shap_explainer.pkl")
    logger.info("  outputs/plots/  (all charts)")
    logger.info("  outputs/reports/model_comparison.csv")
    logger.info("\nNext step -> launch the app:")
    logger.info("  streamlit run app/streamlit_app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn prediction pipeline")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    run(args.config)
