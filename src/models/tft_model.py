"""
Temporal Fusion Transformer — using Darts library for simplicity.
Darts provides a clean TFT implementation backed by PyTorch Lightning.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("tft")


def _prepare_darts_series(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Convert a stock DataFrame to Darts TimeSeries objects.
    
    Args:
        df: DataFrame with DatetimeIndex, features, and target column.
        feature_cols: List of covariate column names.
        target_col: Target column name.
    
    Returns:
        (target_series, covariate_series)
    """
    from darts import TimeSeries
    
    # Ensure sorted by date
    df = df.sort_index().copy()
    
    # Strip timezone info (Darts doesn't support tz-aware DatetimeIndex)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Drop rows where target is NaN
    valid = df.dropna(subset=[target_col])
    
    # Target series
    target_series = TimeSeries.from_dataframe(
        valid.reset_index(), time_col="Date" if "Date" in valid.reset_index().columns else valid.reset_index().columns[0],
        value_cols=target_col,
        fill_missing_dates=True,
        freq="B",
    )
    
    # Covariate series (past observed)
    available_covs = [c for c in feature_cols if c in valid.columns]
    if available_covs:
        covariate_series = TimeSeries.from_dataframe(
            valid.reset_index(), 
            time_col="Date" if "Date" in valid.reset_index().columns else valid.reset_index().columns[0],
            value_cols=available_covs,
            fill_missing_dates=True,
            freq="B",
        )
    else:
        covariate_series = None
    
    return target_series, covariate_series


def train_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "fwd_return_10",
    config: dict = None,
):
    """Train a Temporal Fusion Transformer model.
    
    Args:
        train_df: Training DataFrame (DatetimeIndex, features, target).
        val_df: Validation DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
        config: TFT config section.
    
    Returns:
        Trained TFT model.
    """
    from darts.models import TFTModel
    from pytorch_lightning.callbacks import EarlyStopping
    
    tft_cfg = config["tft"] if config else {}
    
    # Prepare series
    train_target, train_covs = _prepare_darts_series(train_df, feature_cols, target_col)
    val_target, val_covs = _prepare_darts_series(val_df, feature_cols, target_col)
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=tft_cfg.get("patience", 10),
        mode="min",
    )
    
    # Build model
    model = TFTModel(
        input_chunk_length=tft_cfg.get("input_chunk_length", 30),
        output_chunk_length=tft_cfg.get("output_chunk_length", 30),
        hidden_size=tft_cfg.get("hidden_size", 64),
        lstm_layers=tft_cfg.get("num_encoder_layers", 2),
        num_attention_heads=tft_cfg.get("attention_heads", 4),
        dropout=tft_cfg.get("dropout", 0.1),
        batch_size=tft_cfg.get("batch_size", 64),
        n_epochs=tft_cfg.get("max_epochs", 100),
        optimizer_kwargs={"lr": tft_cfg.get("learning_rate", 0.001)},
        random_state=tft_cfg.get("random_state", 42),
        add_relative_index=True,
        pl_trainer_kwargs={
            "callbacks": [early_stop],
            "accelerator": "auto",
            "enable_progress_bar": True,
        },
        force_reset=True,
        save_checkpoints=True,
        model_name=f"tft_{target_col}",
        work_dir=str(PROJECT_ROOT / "models" / "tft"),
    )
    
    logger.info(f"Training TFT for {target_col}...")
    
    # Train
    model.fit(
        series=train_target,
        past_covariates=train_covs,
        val_series=val_target,
        val_past_covariates=val_covs,
    )
    
    logger.info(f"TFT training complete for {target_col}")
    
    # Save
    save_dir = PROJECT_ROOT / "models" / "tft"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / f"tft_{target_col}.pt"))
    logger.info(f"TFT model saved")
    
    return model


def predict_tft(model, df: pd.DataFrame, feature_cols: list, target_col: str, n: int = 30) -> np.ndarray:
    """Generate predictions using trained TFT.
    
    Args:
        model: Trained TFT model.
        df: DataFrame for prediction context.
        feature_cols: Feature columns.
        target_col: Target column name.
        n: Number of steps to predict.
    
    Returns:
        Array of predictions.
    """
    target_series, covs = _prepare_darts_series(df, feature_cols, target_col)
    
    pred = model.predict(n=n, series=target_series, past_covariates=covs)
    return pred.values().flatten()


def load_tft(target_col: str = "fwd_return_10"):
    """Load a saved TFT model."""
    from darts.models import TFTModel
    
    path = PROJECT_ROOT / "models" / "tft" / f"tft_{target_col}.pt"
    model = TFTModel.load(str(path))
    logger.info(f"Loaded TFT model from {path}")
    return model
