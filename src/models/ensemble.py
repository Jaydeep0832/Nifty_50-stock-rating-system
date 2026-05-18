"""IC-weighted ensemble — combines XGBoost + TFT predictions using Spearman IC."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logger import setup_logger

logger = setup_logger("ensemble")


def compute_rolling_ic(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int = 60,
) -> pd.Series:
    """Compute rolling Spearman IC between predictions and actuals.
    
    Args:
        predictions: Model predictions array.
        actuals: Actual forward returns.
        dates: Corresponding dates.
        lookback: Rolling window for IC computation.
    
    Returns:
        Series of rolling IC values.
    """
    df = pd.DataFrame({
        "pred": predictions,
        "actual": actuals,
    }, index=dates)
    
    ic_list = []
    for i in range(len(df)):
        if i < lookback:
            ic_list.append(np.nan)
        else:
            window = df.iloc[i - lookback: i]
            valid = window.dropna()
            if len(valid) > 5:
                ic, _ = spearmanr(valid["pred"], valid["actual"])
                ic_list.append(ic)
            else:
                ic_list.append(np.nan)
    
    return pd.Series(ic_list, index=dates)


def ensemble_predictions(
    xgboost_preds: np.ndarray,
    tft_preds: np.ndarray,
    xgboost_actuals: np.ndarray = None,
    tft_actuals: np.ndarray = None,
    dates: pd.DatetimeIndex = None,
    method: str = "ic_weighted",
    lookback: int = 60,
) -> np.ndarray:
    """Combine predictions from XGBoost and TFT models.
    
    Args:
        xgboost_preds: XGBoost predictions.
        tft_preds: TFT predictions.
        xgboost_actuals: Actuals for XGBoost IC calculation.
        tft_actuals: Actuals for TFT IC calculation.
        dates: Date index.
        method: 'ic_weighted' or 'simple_average'.
        lookback: IC lookback window.
    
    Returns:
        Combined prediction array.
    """
    min_len = min(len(xgboost_preds), len(tft_preds))
    xgboost_preds = xgboost_preds[:min_len]
    tft_preds = tft_preds[:min_len]
    
    if method == "simple_average":
        combined = (xgboost_preds + tft_preds) / 2
        logger.info("Ensemble: simple average")
        return combined
    
    if xgboost_actuals is None or tft_actuals is None or dates is None:
        logger.warning("Missing actuals/dates for IC — falling back to simple average")
        return (xgboost_preds + tft_preds) / 2
    
    xgboost_actuals = xgboost_actuals[:min_len]
    tft_actuals = tft_actuals[:min_len]
    dates = dates[:min_len]
    
    xgboost_ic = compute_rolling_ic(xgboost_preds, xgboost_actuals, dates, lookback)
    tft_ic = compute_rolling_ic(tft_preds, tft_actuals, dates, lookback)
    
    xgboost_abs_ic = xgboost_ic.abs().fillna(0.5)
    tft_abs_ic = tft_ic.abs().fillna(0.5)
    
    total = xgboost_abs_ic + tft_abs_ic + 1e-8
    xgboost_weight = xgboost_abs_ic / total
    tft_weight = tft_abs_ic / total
    
    combined = xgboost_weight.values * xgboost_preds + tft_weight.values * tft_preds
    
    avg_xgb_w = xgboost_weight.mean()
    avg_tft_w = tft_weight.mean()
    logger.info(f"IC-weighted ensemble — Avg weights: XGBoost={avg_xgb_w:.3f}, TFT={avg_tft_w:.3f}")
    
    return combined
