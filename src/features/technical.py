"""Technical indicator generator — produces 25+ ML features from OHLCV data."""

import pandas as pd
import numpy as np
import ta

from src.utils.logger import setup_logger

logger = setup_logger("features")


def add_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add 25+ technical indicators to a stock DataFrame.
    
    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.
        config: Feature config section.
    
    Returns:
        DataFrame with all original + indicator columns.
    """
    feat_cfg = config["features"]
    df = df.copy()
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    df["RSI"] = ta.momentum.RSIIndicator(close, window=feat_cfg["rsi_window"]).rsi()
    
    macd_obj = ta.trend.MACD(
        close,
        window_fast=feat_cfg["macd_fast"],
        window_slow=feat_cfg["macd_slow"],
        window_sign=feat_cfg["macd_signal"],
    )
    df["MACD"] = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"] = macd_obj.macd_diff()
    
    for w in feat_cfg["ema_windows"]:
        df[f"EMA_{w}"] = ta.trend.EMAIndicator(close, window=w).ema_indicator()
    
    for w in feat_cfg["sma_windows"]:
        df[f"SMA_{w}"] = ta.trend.SMAIndicator(close, window=w).sma_indicator()
    
    df["ATR"] = ta.volatility.AverageTrueRange(
        high, low, close, window=feat_cfg["atr_window"]
    ).average_true_range()
    
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_PctB"] = bb.bollinger_pband()
    
    vol_window = feat_cfg["volatility_window"]
    df["Volatility"] = close.pct_change().rolling(vol_window).std() * np.sqrt(252)
    
    vz_window = feat_cfg["volume_zscore_window"]
    vol_mean = volume.rolling(vz_window).mean()
    vol_std = volume.rolling(vz_window).std()
    df["Volume_ZScore"] = (volume - vol_mean) / (vol_std + 1e-8)
    
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    
    for w in feat_cfg["momentum_windows"]:
        df[f"Momentum_{w}"] = close.pct_change(w)
    
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    df["Williams_R"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    df["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
    
    adx_obj = ta.trend.ADXIndicator(high, low, close)
    df["ADX"] = adx_obj.adx()
    df["ADX_Pos"] = adx_obj.adx_pos()
    df["ADX_Neg"] = adx_obj.adx_neg()
    
    dd_window = feat_cfg["drawdown_window"]
    rolling_max = close.rolling(dd_window, min_periods=1).max()
    df["Drawdown"] = (close - rolling_max) / rolling_max
    
    df["Price_to_SMA50"] = close / df["SMA_50"] - 1
    df["Price_to_SMA200"] = close / df["SMA_200"] - 1
    
    df["Daily_Return"] = close.pct_change()
    df["Log_Volume"] = np.log1p(volume)
    
    logger.info(f"Generated {len(df.columns)} columns (incl. OHLCV)")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes OHLCV, Ticker, targets).
    
    Args:
        df: DataFrame with all columns.
    
    Returns:
        List of feature column names.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "Ticker",
               "fwd_return_10", "fwd_return_20", "fwd_return_30"}
    return [c for c in df.columns if c not in exclude]
