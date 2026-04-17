"""XGBoost model — training, prediction, and Optuna hyperparameter tuning."""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from pathlib import Path
from scipy.stats import spearmanr

from src.utils.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("xgboost")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    target_name: str = "fwd_return_10",
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with early stopping.
    
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        params: XGBoost hyperparameters from config.
        target_name: Name for logging and saving.
    
    Returns:
        Trained XGBRegressor.
    """
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=params["random_state"],
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=params["early_stopping_rounds"],
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    
    logger.info(f"XGBoost trained for {target_name} — best iteration: {model.best_iteration}")
    
    save_dir = PROJECT_ROOT / "models" / "xgboost"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"xgb_{target_name}.joblib")
    logger.info(f"Model saved to {save_dir / f'xgb_{target_name}.joblib'}")
    
    return model


def predict_xgboost(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained XGBoost model."""
    return model.predict(X)


def load_xgboost(target_name: str = "fwd_return_10") -> xgb.XGBRegressor:
    """Load a saved XGBoost model."""
    path = PROJECT_ROOT / "models" / "xgboost" / f"xgb_{target_name}.joblib"
    return joblib.load(path)


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    target_name: str = "fwd_return_10",
) -> dict:
    """Optimize XGBoost hyperparameters with Optuna.
    
    Returns:
        Best hyperparameters dict.
    """
    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            early_stopping_rounds=100,
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((preds - y_val.values) ** 2))
        return rmse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Optuna best RMSE: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    
    save_path = PROJECT_ROOT / "models" / "xgboost" / f"best_params_{target_name}.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study.best_params, save_path)
    
    return study.best_params
