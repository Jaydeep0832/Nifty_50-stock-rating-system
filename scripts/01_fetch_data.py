"""Fetch NIFTY 50 stock data from yfinance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.fetcher import fetch_all_stocks


def main():
    config = load_config()
    logger = setup_logger("fetch_data", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching NIFTY 50 Stock Data")
    logger.info("=" * 60)
    
    all_data = fetch_all_stocks(config)
    
    logger.info(f"Data fetch complete. {len(all_data)} stocks saved.")
    logger.info(f"Date range: {config['data']['start_date']} → {config['data']['end_date']}")
    
    for ticker, df in list(all_data.items())[:5]:
        name = ticker.replace(".NS", "")
        logger.info(f"  {name}: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
    
    if len(all_data) > 5:
        logger.info(f"  ... and {len(all_data) - 5} more stocks")


if __name__ == "__main__":
    main()
