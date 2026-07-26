"""
Evaluation metrics — RMSE, Directional Accuracy, Spearman IC, and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("evaluation")

# Style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions with correct sign (direction)."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct) * 100


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (Information Coefficient)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return np.nan
    corr, _ = spearmanr(y_true[mask], y_pred[mask])
    return corr


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model",
) -> dict:
    """Compute all evaluation metrics.
    
    Returns:
        Dict with RMSE, directional_accuracy, spearman_ic.
    """
    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
        "Spearman_IC": spearman_ic(y_true, y_pred),
    }
    
    logger.info(
        f"{label} — RMSE: {metrics['RMSE']:.6f} | "
        f"Dir.Acc: {metrics['Directional_Accuracy']:.1f}% | "
        f"IC: {metrics['Spearman_IC']:.4f}"
    )
    return metrics


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_name: str = None,
):
    """Scatter plot of predictions vs actuals."""
    charts_dir = PROJECT_ROOT / "outputs" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#4a90d9")
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect Prediction")
    
    ax.set_xlabel("Actual Returns", fontsize=12)
    ax.set_ylabel("Predicted Returns", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    
    if save_name:
        fig.savefig(charts_dir / f"{save_name}.png", dpi=150)
        logger.info(f"Saved chart: {save_name}.png")
    plt.close(fig)


def plot_ic_over_time(
    ic_series: pd.Series,
    title: str = "IC Over Time",
    save_name: str = None,
):
    """Line plot of rolling IC over time."""
    charts_dir = PROJECT_ROOT / "outputs" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ic_series.index, ic_series.values, color="#2ecc71", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(ic_series.index, ic_series.values, 0, alpha=0.2, color="#2ecc71")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Spearman IC", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_name:
        fig.savefig(charts_dir / f"{save_name}.png", dpi=150)
        logger.info(f"Saved chart: {save_name}.png")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_name: str = None,
):
    """Plot LightGBM feature importance (top N)."""
    charts_dir = PROJECT_ROOT / "outputs" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feat_imp, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    
    if save_name:
        fig.savefig(charts_dir / f"{save_name}.png", dpi=150)
        logger.info(f"Saved chart: {save_name}.png")
    plt.close(fig)
