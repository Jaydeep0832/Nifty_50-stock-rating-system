"""Data loader — reads saved stock data from disk."""

import pandas as pd
from pathlib import Path

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.data.nifty50_tickers import NIFTY50_TICKERS

logger = setup_logger("loader")


def load_stock_data(ticker: str, config: dict = None) -> pd.DataFrame:
    """Load a single stock's raw data from disk.
    
    Args:
        ticker: e.g., 'RELIANCE.NS'
        config: Configuration dict.
    
    Returns:
        DataFrame with OHLCV data.
    """
    if config is None:
        config = load_config()
    
    raw_dir = get_path(config, "data.raw_dir")
    name = ticker.replace(".NS", "")
    file_format = config["data"].get("file_format", "parquet")
    
    if file_format == "parquet":
        path = raw_dir / f"{name}.parquet"
        df = pd.read_parquet(path)
    else:
        path = raw_dir / f"{name}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    logger.debug(f"Loaded {name}: {len(df)} rows")
    return df


def load_all_stocks(config: dict = None) -> dict:
    """Load all available stock data from disk.
    
    Returns:
        Dictionary of {ticker: DataFrame}.
    """
    if config is None:
        config = load_config()
    
    raw_dir = get_path(config, "data.raw_dir")
    file_format = config["data"].get("file_format", "parquet")
    ext = ".parquet" if file_format == "parquet" else ".csv"
    
    all_data = {}
    
    for ticker in NIFTY50_TICKERS:
        name = ticker.replace(".NS", "")
        path = raw_dir / f"{name}{ext}"
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        
        try:
            df = load_stock_data(ticker, config)
            all_data[ticker] = df
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
    
    logger.info(f"Loaded {len(all_data)} stocks from disk")
    return all_data
