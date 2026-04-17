"""Rating system — converts ensemble predictions to 1-10 composite ratings."""

import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("rater")


def compute_ratings(
    predictions: pd.DataFrame,
    scale_min: int = 1,
    scale_max: int = 10,
) -> pd.DataFrame:
    """Convert predictions to percentile-based ratings.
    
    Process:
    1. Rank predictions for each horizon as percentiles (0-100)
    2. Map percentiles to scale_min–scale_max rating
    3. Average ratings across horizons for composite rating
    
    Args:
        predictions: DataFrame with 'pred_10', 'pred_20', 'pred_30' columns.
        scale_min: Minimum rating (default 1).
        scale_max: Maximum rating (default 10).
    
    Returns:
        DataFrame with percentile ranks and final composite rating.
    """
    result = predictions.copy()
    pred_cols = [c for c in result.columns if c.startswith("pred_")]
    
    if not pred_cols:
        logger.error("No prediction columns found (expected 'pred_' prefix)")
        return result
    
    rating_cols = []
    for col in pred_cols:
        pct_col = f"pct_{col}"
        result[pct_col] = result[col].rank(pct=True) * 100
        
        rating_col = f"rating_{col}"
        result[rating_col] = np.ceil(
            result[pct_col] / 100 * (scale_max - scale_min) + scale_min
        ).clip(scale_min, scale_max).astype(int)
        rating_cols.append(rating_col)
    
    result["composite_rating"] = result[rating_cols].mean(axis=1).round().astype(int)
    result["composite_rating"] = result["composite_rating"].clip(scale_min, scale_max)
    
    result = result.sort_values("composite_rating", ascending=False)
    
    logger.info(f"Ratings computed: {len(result)} stocks, scale {scale_min}-{scale_max}")
    return result
