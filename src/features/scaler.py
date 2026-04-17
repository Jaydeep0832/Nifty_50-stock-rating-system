"""
Feature scaler — RobustScaler or StandardScaler with save/load support.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("scaler")


def fit_scaler(df: pd.DataFrame, feature_cols: list, method: str = "robust"):
    """Fit a scaler on feature columns.
    
    Args:
        df: Training DataFrame.
        feature_cols: List of columns to scale.
        method: 'robust' or 'standard'.
    
    Returns:
        Fitted scaler object.
    """
    ScalerClass = RobustScaler if method == "robust" else StandardScaler
    scaler = ScalerClass()
    
    # Replace inf with NaN before fitting
    data = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    scaler.fit(data.dropna())
    
    logger.info(f"Fitted {method} scaler on {len(feature_cols)} features")
    return scaler


def transform_features(df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    """Transform features using fitted scaler.
    
    Args:
        df: DataFrame to transform.
        feature_cols: Columns to scale.
        scaler: Fitted scaler.
    
    Returns:
        DataFrame with scaled features (other columns unchanged).
    """
    df = df.copy()
    data = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = scaler.transform(data)
    return df


def save_scaler(scaler, name: str = "feature_scaler"):
    """Save scaler to models/ directory."""
    path = PROJECT_ROOT / "models" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to {path}")


def load_scaler(name: str = "feature_scaler"):
    """Load scaler from models/ directory."""
    path = PROJECT_ROOT / "models" / f"{name}.joblib"
    scaler = joblib.load(path)
    logger.info(f"Loaded scaler from {path}")
    return scaler
