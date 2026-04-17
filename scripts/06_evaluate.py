"""Full model evaluation and visualization generation."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import (
    evaluate_predictions,
    plot_pred_vs_actual,
    plot_ic_over_time,
    plot_feature_importance,
)
from src.models.lgbm_model import load_lgbm
from src.models.ensemble import compute_rolling_ic


def main():
    config = load_config()
    logger = setup_logger("evaluate", config["logging"]["level"], config["logging"]["file"])
    
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluation and Visualization")
    logger.info("=" * 60)
    
    predictions_dir = get_path(config, "data.predictions_dir")
    processed_dir = get_path(config, "data.processed_dir")
    
    feature_cols = pd.read_csv(processed_dir / "feature_columns.csv").iloc[:, 0].tolist()
    
    horizons = config["targets"]["forward_days"]
    all_metrics = []
    
    for horizon in horizons:
        target = f"fwd_return_{horizon}"
        lgbm_file = predictions_dir / f"lgbm_{target}_predictions.csv"
        
        if not lgbm_file.exists():
            logger.warning(f"Predictions not found for {target}")
            continue
        
        preds = pd.read_csv(lgbm_file, index_col=0, parse_dates=True)
        
        metrics = evaluate_predictions(
            preds["actual"].values,
            preds["predicted"].values,
            label=f"LightGBM-{target}",
        )
        metrics["target"] = target
        all_metrics.append(metrics)
        
        plot_pred_vs_actual(
            preds["actual"].values,
            preds["predicted"].values,
            title=f"Predicted vs Actual: {target}",
            save_name=f"eval_{target}_scatter",
        )
        
        ic_series = compute_rolling_ic(
            preds["predicted"].values,
            preds["actual"].values,
            preds.index,
            lookback=config["ensemble"]["ic_lookback_days"],
        )
        plot_ic_over_time(
            ic_series.dropna(),
            title=f"Rolling IC: {target}",
            save_name=f"eval_{target}_ic",
        )
        
        # Feature importance
        try:
            model = load_lgbm(target)
            plot_feature_importance(
                model, feature_cols, top_n=20,
                save_name=f"eval_{target}_feat_importance",
            )
        except Exception as e:
            logger.warning(f"Could not load model for feature importance: {e}")
    
    # Summary table
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        tables_dir = get_path(config, "data.predictions_dir").parent.parent / "outputs" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(tables_dir / "evaluation_summary.csv", index=False)
        
        logger.info("\n📊 Evaluation Summary:")
        logger.info(summary.to_string(index=False))
    
    logger.info("Evaluation complete! Charts saved to outputs/charts/")


if __name__ == "__main__":
    main()
