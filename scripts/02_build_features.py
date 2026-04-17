"""Build technical features and forward return targets for all stocks."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.data.loader import load_all_stocks
from src.features.technical import add_technical_indicators, get_feature_columns
from src.features.targets import add_forward_returns
from src.features.scaler import fit_scaler, transform_features, save_scaler


def main():
    config = load_config()
    logger = setup_logger("build_features", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 2: Building Features and Targets")
    logger.info("=" * 60)
    
    all_data = load_all_stocks(config)
    
    if not all_data:
        logger.error("No data found. Run 01_fetch_data first.")
        return
    
    processed_dir = get_path(config, "data.processed_dir")
    warmup = config["features"]["warmup_days"]
    all_processed = []
    
    for ticker, df in all_data.items():
        name = ticker.replace(".NS", "")
        logger.info(f"Processing {name}...")
        
        df = add_technical_indicators(df, config)
        df = add_forward_returns(df, config)
        df = df.iloc[warmup:]
        
        target_cols = [f"fwd_return_{n}" for n in config["targets"]["forward_days"]]
        df = df.dropna(subset=target_cols, how="all")
        
        all_processed.append(df)
        logger.info(f"  {name}: {len(df)} rows after warmup + target alignment")
    
    combined = pd.concat(all_processed, axis=0)
    logger.info(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns")
    
    feature_cols = get_feature_columns(combined)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")
    
    combined[feature_cols] = combined[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    for col in feature_cols:
        median_val = combined[col].median()
        combined[col] = combined[col].fillna(median_val)
    
    n = len(combined)
    train_end = int(n * config["split"]["train_ratio"])
    
    train_data = combined.iloc[:train_end]
    scaler = fit_scaler(train_data, feature_cols, config["features"]["scaler"])
    save_scaler(scaler)
    
    # Transform all data
    combined = transform_features(combined, feature_cols, scaler)
    
    # Save processed data
    file_format = config["data"].get("file_format", "parquet")
    if file_format == "parquet":
        combined.to_parquet(processed_dir / "features_targets.parquet")
    else:
        combined.to_csv(processed_dir / "features_targets.csv")
    
    logger.info(f"Saved processed data: {processed_dir / 'features_targets.parquet'}")
    logger.info(f"Shape: {combined.shape}")
    
    # Save feature column list
    pd.Series(feature_cols).to_csv(processed_dir / "feature_columns.csv", index=False)
    logger.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
