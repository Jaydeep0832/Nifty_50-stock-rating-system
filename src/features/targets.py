"""Target engineering — creates forward return columns for prediction."""

import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("targets")


def add_forward_returns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create forward return targets for multi-horizon prediction.
    
    Forward returns: fwd_return_N = (Close[t+N] - Close[t]) / Close[t]
    Last N rows will have NaN for each target (no future data).

    Args:
        df: DataFrame with Close column, sorted by date.
        config: Config dict with targets.forward_days.
    
    Returns:
        DataFrame with added forward return columns.
    """
    df = df.copy()
    forward_days = config["targets"]["forward_days"]
    
    for n in forward_days:
        col_name = f"fwd_return_{n}"
        df[col_name] = df["Close"].pct_change(n).shift(-n)
        logger.debug(f"Added {col_name}: {df[col_name].notna().sum()} valid rows")
    
    logger.info(f"Added forward returns for horizons: {forward_days}")
    return df
