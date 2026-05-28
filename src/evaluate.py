"""
src/evaluate.py
Full model evaluation:
  - All metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Confusion matrices
  - ROC curves
  - SHAP explainability plots
  - Saves comparison report as CSV
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def compute_metrics(model, X_test, y_test, name: str, threshold: float = 0.5) -> dict:
    """Compute full evaluation metrics for a model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1"       : round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba) * 100, 2),
        "_y_pred"  : y_pred,
        "_y_proba" : y_proba,
    }
    logger.info(f"{name:30s} | F1: {metrics['F1']}% | AUC: {metrics['ROC-AUC']}%")
    return metrics


def plot_comparison(all_metrics: list, plots_dir: str):
    """Bar chart comparing all models across all metrics."""
    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_names = [m["Model"] for m in all_metrics]
    colors      = ["#185FA5", "#1D9E75", "#BA7517", "#E24B4A"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(20, 5))
    fig.suptitle("Model Comparison — All Metrics", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metric_keys):
        vals = [m[metric] for m in all_metrics]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)],
                      edgecolor="white", width=0.6)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=7)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison chart saved -> {path}")


def plot_confusion_matrices(models_dict: dict, X_test, y_test, plots_dir: str):
    """Confusion matrix grid for all models."""
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Stay", "Churn"],
                    yticklabels=["Stay", "Churn"],
                    linewidths=0.5)
        ax.set_title(name, fontweight="bold", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices saved -> {path}")


def plot_roc_curves(all_metrics: list, y_test, plots_dir: str):
    """Overlay ROC curves for all models."""
    colors = ["#185FA5", "#1D9E75", "#BA7517", "#E24B4A"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for m, col in zip(all_metrics, colors):
        fpr, tpr, _ = roc_curve(y_test, m["_y_proba"])
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{m['Model']} (AUC={m['ROC-AUC']}%)")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(plots_dir, "roc_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved -> {path}")


def run_shap_analysis(model, X_test, feature_names: list, config: dict):
    """Full SHAP analysis: summary + waterfall for best model."""
    plots_dir  = config["outputs"]["plots_dir"]
    max_display = config["shap"]["max_display"]

    logger.info("Running SHAP analysis on best model (XGBoost tuned)...")

    X_test_df   = pd.DataFrame(X_test, columns=feature_names)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test_df)

    # Save explainer for app use
    joblib.dump(explainer, os.path.join(config["models"]["save_dir"], "shap_explainer.pkl"))
    logger.info("SHAP explainer saved -> models/shap_explainer.pkl")

    # ── Plot 1: Summary bar (global importance) ──────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    plt.sca(ax1)
    shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                      max_display=max_display, show=False, color="#534AB7")
    ax1.set_title("SHAP Global Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_importance.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Beeswarm (direction of impact) ───────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    plt.sca(ax2)
    shap.summary_plot(shap_values, X_test_df, max_display=max_display, show=False)
    ax2.set_title("SHAP Summary Beeswarm", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_beeswarm.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Waterfall for one churned customer ───────────
    churned_idx = np.where(np.array(list(X_test_df.index)) != -1)[0][0]  # first sample
    exp = shap_values[0]
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    shap.waterfall_plot(exp, max_display=max_display, show=False)
    plt.title("SHAP Waterfall — Single Customer", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_waterfall.png"), dpi=130, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plots saved -> outputs/plots/")

    # Log top features
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = pd.Series(mean_abs, index=feature_names).nlargest(5)
    logger.info(f"Top 5 SHAP features:\n{top_features.to_string()}")

    return explainer, shap_values


def save_report(all_metrics: list, reports_dir: str):
    """Save model comparison table as CSV + JSON."""
    os.makedirs(reports_dir, exist_ok=True)

    report_data = [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in all_metrics
    ]

    # CSV
    df = pd.DataFrame(report_data).set_index("Model")
    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df.to_csv(csv_path)
    logger.info(f"Report saved -> {csv_path}")

    # JSON
    json_path = os.path.join(reports_dir, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Report saved -> {json_path}")

    return df


def run_evaluation(models_dict: dict, X_test, y_test, feature_names: list, config: dict):
    """Full evaluation pipeline for all models."""
    logger.info("=" * 50)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("=" * 50)

    plots_dir   = config["outputs"]["plots_dir"]
    reports_dir = config["outputs"]["reports_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    threshold   = config["evaluation"]["threshold"]
    all_metrics = []

    for name, model in models_dict.items():
        metrics = compute_metrics(model, X_test, y_test, name, threshold)
        all_metrics.append(metrics)
        # Detailed classification report
        logger.debug(f"\n{classification_report(y_test, metrics['_y_pred'])}")

    plot_comparison(all_metrics, plots_dir)
    plot_confusion_matrices(models_dict, X_test, y_test, plots_dir)
    plot_roc_curves(all_metrics, y_test, plots_dir)

    # SHAP on best model
    best_model = models_dict.get("xgboost_tuned", list(models_dict.values())[-1])
    explainer, shap_values = run_shap_analysis(best_model, X_test, feature_names, config)

    report_df = save_report(all_metrics, reports_dir)
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"\n{report_df.to_string()}")

    logger.info("EVALUATION COMPLETE")
    return all_metrics, explainer


if __name__ == "__main__":
    import joblib
    from src.preprocess import run_preprocessing
    from src.train import run_training

    config = load_config()
    X_train, X_test, y_train, y_test, features, scaler = run_preprocessing(config)
    models = run_training(X_train, X_test, y_train, y_test, config)
    run_evaluation(models, X_test, y_test, features, config)
