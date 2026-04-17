"""IC-weighted ensemble — combines LightGBM + TFT predictions using Spearman IC."""

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
    lgbm_preds: np.ndarray,
    tft_preds: np.ndarray,
    lgbm_actuals: np.ndarray = None,
    tft_actuals: np.ndarray = None,
    dates: pd.DatetimeIndex = None,
    method: str = "ic_weighted",
    lookback: int = 60,
) -> np.ndarray:
    """Combine predictions from two models.
    
    Args:
        lgbm_preds: LightGBM predictions.
        tft_preds: TFT predictions.
        lgbm_actuals: Actuals for LightGBM IC calculation.
        tft_actuals: Actuals for TFT IC calculation.
        dates: Date index.
        method: 'ic_weighted' or 'simple_average'.
        lookback: IC lookback window.
    
    Returns:
        Combined prediction array.
    """
    min_len = min(len(lgbm_preds), len(tft_preds))
    lgbm_preds = lgbm_preds[:min_len]
    tft_preds = tft_preds[:min_len]
    
    if method == "simple_average":
        combined = (lgbm_preds + tft_preds) / 2
        logger.info("Ensemble: simple average")
        return combined
    
    if lgbm_actuals is None or tft_actuals is None or dates is None:
        logger.warning("Missing actuals/dates for IC — falling back to simple average")
        return (lgbm_preds + tft_preds) / 2
    
    lgbm_actuals = lgbm_actuals[:min_len]
    tft_actuals = tft_actuals[:min_len]
    dates = dates[:min_len]
    
    lgbm_ic = compute_rolling_ic(lgbm_preds, lgbm_actuals, dates, lookback)
    tft_ic = compute_rolling_ic(tft_preds, tft_actuals, dates, lookback)
    
    lgbm_abs_ic = lgbm_ic.abs().fillna(0.5)
    tft_abs_ic = tft_ic.abs().fillna(0.5)
    
    total = lgbm_abs_ic + tft_abs_ic + 1e-8
    lgbm_weight = lgbm_abs_ic / total
    tft_weight = tft_abs_ic / total
    
    combined = lgbm_weight.values * lgbm_preds + tft_weight.values * tft_preds
    
    avg_lgbm_w = lgbm_weight.mean()
    avg_tft_w = tft_weight.mean()
    logger.info(f"IC-weighted ensemble — Avg weights: LightGBM={avg_lgbm_w:.3f}, TFT={avg_tft_w:.3f}")
    
    return combined
