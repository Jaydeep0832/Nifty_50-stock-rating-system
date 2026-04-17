"""Train Temporal Fusion Transformer models for multi-horizon prediction."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.models.tft_model import train_tft, predict_tft


def main():
    config = load_config()
    logger = setup_logger("train_tft", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 4: Training TFT Models")
    logger.info("=" * 60)
    
    processed_dir = get_path(config, "data.processed_dir")
    file_format = config["data"].get("file_format", "parquet")
    
    if file_format == "parquet":
        df = pd.read_parquet(processed_dir / "features_targets.parquet")
    else:
        df = pd.read_csv(processed_dir / "features_targets.csv", index_col=0, parse_dates=True)
    
    feature_cols = pd.read_csv(processed_dir / "feature_columns.csv").iloc[:, 0].tolist()
    
    n = len(df)
    train_end = int(n * config["split"]["train_ratio"])
    val_end = int(n * (config["split"]["train_ratio"] + config["split"]["val_ratio"]))
    
    target_cols = [f"fwd_return_{d}" for d in config["targets"]["forward_days"]]
    predictions_dir = get_path(config, "data.predictions_dir")
    
    tickers = df["Ticker"].unique() if "Ticker" in df.columns else ["ALL"]
    
    for target in target_cols:
        logger.info(f"\nTraining TFT for {target}...")
        
        all_preds = []
        
        for ticker in tickers[:5]:
            logger.info(f"  Processing {ticker}...")
            
            if ticker == "ALL":
                stock_df = df.dropna(subset=[target])
            else:
                stock_df = df[df["Ticker"] == ticker].dropna(subset=[target])
            
            if len(stock_df) < 200:
                logger.warning(f"  Skipping {ticker} — only {len(stock_df)} rows")
                continue
            
            sn = len(stock_df)
            s_train_end = int(sn * config["split"]["train_ratio"])
            s_val_end = int(sn * (config["split"]["train_ratio"] + config["split"]["val_ratio"]))
            
            train_df = stock_df.iloc[:s_train_end]
            val_df = stock_df.iloc[s_train_end:s_val_end]
            
            if len(train_df) < 100 or len(val_df) < 30:
                logger.warning(f"  Skipping {ticker} — insufficient data")
                continue
            
            try:
                model = train_tft(train_df, val_df, feature_cols, target, config)
                logger.info(f"  TFT trained for {ticker} / {target}")
            except Exception as e:
                logger.error(f"  TFT training failed for {ticker}: {e}")
                continue
        
        logger.info(f"TFT training complete for {target}")
    
    logger.info("TFT training pipeline complete!")


if __name__ == "__main__":
    main()
