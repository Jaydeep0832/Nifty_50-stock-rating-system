"""Generate predictions for all stocks + generate composite ratings."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_path, PROJECT_ROOT
from src.utils.logger import setup_logger
from src.models.lgbm_model import load_lgbm
from src.rating.rater import compute_ratings


def main():
    config = load_config()
    logger = setup_logger("ensemble_rate", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 5: Inference + Rating Generation")
    logger.info("=" * 60)
    
    processed_dir = get_path(config, "data.processed_dir")
    file_format = config["data"].get("file_format", "parquet")
    
    if file_format == "parquet":
        df = pd.read_parquet(processed_dir / "features_targets.parquet")
    else:
        df = pd.read_csv(processed_dir / "features_targets.csv", index_col=0, parse_dates=True)
    
    feature_cols = pd.read_csv(processed_dir / "feature_columns.csv").iloc[:, 0].tolist()
    
    horizons = config["targets"]["forward_days"]
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = df["Ticker"].unique() if "Ticker" in df.columns else []
    logger.info(f"Found {len(tickers)} unique stocks")
    
    stock_predictions = {}
    
    for ticker in tickers:
        stock_df = df[df["Ticker"] == ticker].sort_index()
        
        if stock_df.empty:
            continue
        
        latest = stock_df.iloc[-1:]
        X_latest = latest[feature_cols]
        X_latest = X_latest.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        name = ticker.replace(".NS", "")
        stock_predictions[ticker] = {"ticker": ticker}
        
        for horizon in horizons:
            target = f"fwd_return_{horizon}"
            
            try:
                model = load_lgbm(target)
                pred = model.predict(X_latest)[0]
                stock_predictions[ticker][f"pred_{horizon}"] = pred
            except Exception as e:
                logger.warning(f"Could not predict {target} for {name}: {e}")
                stock_predictions[ticker][f"pred_{horizon}"] = 0.0
        
        logger.debug(f"  {name}: predicted across {len(horizons)} horizons")
    
    if not stock_predictions:
        logger.error("No predictions generated. Train models first (03_train_lgbm).")
        return
    
    pred_df = pd.DataFrame(stock_predictions.values()).set_index("ticker")
    logger.info(f"Predictions for {len(pred_df)} stocks across {len(horizons)} horizons")
    
    rated = compute_ratings(
        pred_df,
        scale_min=config["rating"]["scale_min"],
        scale_max=config["rating"]["scale_max"],
    )
    
    rated.to_csv(tables_dir / "stock_ratings.csv")
    logger.info(f"Ratings saved to {tables_dir / 'stock_ratings.csv'}")
    
    predictions_dir = get_path(config, "data.predictions_dir")
    pred_df.to_csv(predictions_dir / "all_stocks_latest_predictions.csv")
    
    logger.info(f"\nTOP 10 Rated Stocks:")
    for _, row in rated.head(10).iterrows():
        pred_cols = [c for c in rated.columns if c.startswith("pred_")]
        preds_str = " | ".join([f"{c}: {row[c]:+.4f}" for c in pred_cols])
        logger.info(f"  {row.name} → Rating: {row['composite_rating']} ({preds_str})")
    
    logger.info(f"\nBOTTOM 5 Rated Stocks:")
    for _, row in rated.tail(5).iterrows():
        pred_cols = [c for c in rated.columns if c.startswith("pred_")]
        preds_str = " | ".join([f"{c}: {row[c]:+.4f}" for c in pred_cols])
        logger.info(f"  {row.name} → Rating: {row['composite_rating']} ({preds_str})")
    
    logger.info(f"\nRating Distribution:")
    for rating in range(10, 0, -1):
        count = (rated["composite_rating"] == rating).sum()
        bar = "█" * count
        logger.info(f"  Rating {rating:2d}: {bar} ({count})")


if __name__ == "__main__":
    main()
