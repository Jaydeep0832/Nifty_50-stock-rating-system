"""Train XGBoost models for multi-horizon forward return prediction."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.models.xgboost_model import train_xgboost, predict_xgboost, optimize_xgboost
from src.evaluation.metrics import evaluate_predictions, plot_pred_vs_actual, plot_feature_importance


def load_data(config):
    """Load processed features and targets."""
    processed_dir = get_path(config, "data.processed_dir")
    file_format = config["data"].get("file_format", "parquet")
    
    if file_format == "parquet":
        df = pd.read_parquet(processed_dir / "features_targets.parquet")
    else:
        df = pd.read_csv(processed_dir / "features_targets.csv", index_col=0, parse_dates=True)
    
    feature_cols = pd.read_csv(processed_dir / "feature_columns.csv").iloc[:, 0].tolist()
    return df, feature_cols


def time_series_split(df, config):
    """Time-based split: train / val / test."""
    n = len(df)
    train_end = int(n * config["split"]["train_ratio"])
    val_end = int(n * (config["split"]["train_ratio"] + config["split"]["val_ratio"]))
    
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()
    
    config = load_config()
    logger = setup_logger("train_xgboost", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 3: Training XGBoost Models")
    logger.info("=" * 60)
    
    df, feature_cols = load_data(config)
    train, val, test = time_series_split(df, config)
    
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    target_cols = [f"fwd_return_{n}" for n in config["targets"]["forward_days"]]
    predictions_dir = get_path(config, "data.predictions_dir")
    all_results = {}
    
    for target in target_cols:
        logger.info(f"\nTraining for {target}...")
        
        train_clean = train.dropna(subset=[target])
        val_clean = val.dropna(subset=[target])
        test_clean = test.dropna(subset=[target])
        
        X_train = train_clean[feature_cols]
        y_train = train_clean[target]
        X_val = val_clean[feature_cols]
        y_val = val_clean[target]
        X_test = test_clean[feature_cols]
        y_test = test_clean[target]
        
        if args.optimize:
            logger.info(f"Running Optuna optimization for {target}...")
            best_params = optimize_xgboost(
                X_train, y_train, X_val, y_val,
                n_trials=config["xgboost"]["optuna_trials"],
                target_name=target,
            )
            params = {**config["xgboost"], **best_params}
        else:
            params = config["xgboost"]
        
        model = train_xgboost(X_train, y_train, X_val, y_val, params, target)
        preds = predict_xgboost(model, X_test)
        
        metrics = evaluate_predictions(y_test.values, preds, label=f"XGBoost-{target}")
        all_results[target] = metrics
        
        pred_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": preds,
            "ticker": test_clean["Ticker"].values if "Ticker" in test_clean.columns else "unknown",
        }, index=test_clean.index)
        pred_df.to_csv(predictions_dir / f"xgb_{target}_predictions.csv")
        
        plot_pred_vs_actual(y_test.values, preds, f"XGBoost: {target}", f"xgb_{target}_scatter")
        plot_feature_importance(model, feature_cols, top_n=20, save_name=f"xgb_{target}_importance")
    
    logger.info("\n" + "=" * 60)
    logger.info("XGBoost Training Summary:")
    for target, m in all_results.items():
        logger.info(f"  {target} → RMSE={m['RMSE']:.6f} | Dir.Acc={m['Directional_Accuracy']:.1f}% | IC={m['Spearman_IC']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
