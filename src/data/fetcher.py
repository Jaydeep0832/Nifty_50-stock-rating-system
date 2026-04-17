import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

from src.utils.config import load_config, get_path, PROJECT_ROOT
from src.utils.logger import setup_logger
from src.data.nifty50_tickers import NIFTY50_TICKERS

logger = setup_logger("fetcher")


def fetch_single_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker.
    
    Args:
        ticker: e.g., 'RELIANCE.NS'
        start: Start date string 'YYYY-MM-DD'
        end: End date string 'YYYY-MM-DD'
    
    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.
    """
    logger.info(f"Fetching {ticker} from {start} to {end}")
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, auto_adjust=True)
    
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()
    
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bdays)
    
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    
    df = df.dropna(subset=["Close"])
    
    df.index.name = "Date"
    df["Ticker"] = ticker
    
    return df


def fetch_all_stocks(config: dict = None) -> dict:
    """Fetch data for all NIFTY 50 stocks and save to disk.
    
    Args:
        config: Configuration dict. Loads default if None.
    
    Returns:
        Dictionary of {ticker: DataFrame}.
    """
    if config is None:
        config = load_config()
    
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    raw_dir = get_path(config, "data.raw_dir")
    file_format = config["data"].get("file_format", "parquet")
    
    all_data = {}
    failed = []
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Fetching stocks"):
        try:
            df = fetch_single_stock(ticker, start, end)
            if df.empty:
                failed.append(ticker)
                continue
            
            name = ticker.replace(".NS", "")
            if file_format == "parquet":
                df.to_parquet(raw_dir / f"{name}.parquet")
            else:
                df.to_csv(raw_dir / f"{name}.csv")
            
            all_data[ticker] = df
            logger.info(f"Saved {name}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)
    
    logger.info(f"Successfully fetched {len(all_data)}/{len(NIFTY50_TICKERS)} stocks")
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    
    return all_data
