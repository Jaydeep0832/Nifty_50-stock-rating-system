"""FastAPI backend — serves stock ratings, predictions, and technical indicators."""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import load_config, get_path
from src.data.nifty50_tickers import NIFTY50_TICKERS, NIFTY50_NAMES

app = FastAPI(title="NIFTY 50 Stock Rating System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

config = load_config()


def _load_ratings() -> pd.DataFrame:
    """Load stock ratings from CSV."""
    path = PROJECT_ROOT / "outputs" / "tables" / "stock_ratings.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _load_raw_data(ticker: str) -> pd.DataFrame:
    """Load raw OHLCV data for a ticker."""
    raw_dir = get_path(config, "data.raw_dir")
    name = ticker.replace(".NS", "")
    path = raw_dir / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    csv_path = raw_dir / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return pd.DataFrame()


def _load_processed_stock(ticker: str) -> pd.DataFrame:
    """Load processed features for a single stock."""
    processed_dir = get_path(config, "data.processed_dir")
    path = processed_dir / "features_targets.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df[df["Ticker"] == ticker] if "Ticker" in df.columns else pd.DataFrame()


def _compute_indicator_signals(row: pd.Series) -> dict:
    """Compute buy/sell/neutral signals for technical indicators."""
    signals = {}

    rsi = row.get("RSI", 50)
    if pd.notna(rsi):
        if rsi < 30:
            signals["RSI"] = {"value": round(rsi, 2), "signal": "oversold", "action": "buy"}
        elif rsi > 70:
            signals["RSI"] = {"value": round(rsi, 2), "signal": "overbought", "action": "sell"}
        else:
            signals["RSI"] = {"value": round(rsi, 2), "signal": "neutral", "action": "hold"}

    macd = row.get("MACD", 0)
    macd_signal = row.get("MACD_Signal", 0)
    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal:
            signals["MACD"] = {"value": round(macd, 4), "signal": "bullish crossover", "action": "buy"}
        else:
            signals["MACD"] = {"value": round(macd, 4), "signal": "bearish crossover", "action": "sell"}

    adx = row.get("ADX", 0)
    if pd.notna(adx):
        if adx > 25:
            signals["ADX"] = {"value": round(adx, 2), "signal": "strong trend", "action": "buy"}
        else:
            signals["ADX"] = {"value": round(adx, 2), "signal": "weak trend", "action": "hold"}

    stoch_k = row.get("Stoch_K", 50)
    if pd.notna(stoch_k):
        if stoch_k < 20:
            signals["Stochastic"] = {"value": round(stoch_k, 2), "signal": "oversold", "action": "buy"}
        elif stoch_k > 80:
            signals["Stochastic"] = {"value": round(stoch_k, 2), "signal": "overbought", "action": "sell"}
        else:
            signals["Stochastic"] = {"value": round(stoch_k, 2), "signal": "neutral", "action": "hold"}

    williams = row.get("Williams_R", -50)
    if pd.notna(williams):
        if williams < -80:
            signals["Williams_%R"] = {"value": round(williams, 2), "signal": "oversold", "action": "buy"}
        elif williams > -20:
            signals["Williams_%R"] = {"value": round(williams, 2), "signal": "overbought", "action": "sell"}
        else:
            signals["Williams_%R"] = {"value": round(williams, 2), "signal": "neutral", "action": "hold"}

    cci = row.get("CCI", 0)
    if pd.notna(cci):
        if cci < -100:
            signals["CCI"] = {"value": round(cci, 2), "signal": "oversold", "action": "buy"}
        elif cci > 100:
            signals["CCI"] = {"value": round(cci, 2), "signal": "overbought", "action": "sell"}
        else:
            signals["CCI"] = {"value": round(cci, 2), "signal": "neutral", "action": "hold"}

    bb_pctb = row.get("BB_PctB", 0.5)
    if pd.notna(bb_pctb):
        if bb_pctb < 0:
            signals["Bollinger_%B"] = {"value": round(bb_pctb, 4), "signal": "below lower band", "action": "buy"}
        elif bb_pctb > 1:
            signals["Bollinger_%B"] = {"value": round(bb_pctb, 4), "signal": "above upper band", "action": "sell"}
        else:
            signals["Bollinger_%B"] = {"value": round(bb_pctb, 4), "signal": "within bands", "action": "hold"}

    for col in ["EMA_9", "EMA_21", "EMA_50", "EMA_200"]:
        val = row.get(col, None)
        close = row.get("Close", None)
        if pd.notna(val) and pd.notna(close):
            if close > val:
                signals[col] = {"value": round(val, 2), "signal": f"price above {col}", "action": "buy"}
            else:
                signals[col] = {"value": round(val, 2), "signal": f"price below {col}", "action": "sell"}

    for col in ["SMA_50", "SMA_200"]:
        val = row.get(col, None)
        close = row.get("Close", None)
        if pd.notna(val) and pd.notna(close):
            if close > val:
                signals[col] = {"value": round(val, 2), "signal": f"price above {col}", "action": "buy"}
            else:
                signals[col] = {"value": round(val, 2), "signal": f"price below {col}", "action": "sell"}

    vol = row.get("Volatility", 0)
    if pd.notna(vol):
        signals["Volatility"] = {
            "value": round(vol * 100, 2),
            "signal": "high" if vol > 0.3 else ("moderate" if vol > 0.15 else "low"),
            "action": "hold"
        }

    atr = row.get("ATR", 0)
    if pd.notna(atr):
        signals["ATR"] = {"value": round(atr, 2), "signal": "volatility measure", "action": "hold"}

    for w in [5, 10, 20]:
        col = f"Momentum_{w}"
        val = row.get(col, 0)
        if pd.notna(val):
            signals[col] = {
                "value": round(val * 100, 2),
                "signal": "positive momentum" if val > 0 else "negative momentum",
                "action": "buy" if val > 0 else "sell"
            }

    dd = row.get("Drawdown", 0)
    if pd.notna(dd):
        signals["Drawdown"] = {
            "value": round(dd * 100, 2),
            "signal": "deep drawdown" if dd < -0.2 else ("moderate" if dd < -0.1 else "healthy"),
            "action": "buy" if dd < -0.2 else "hold"
        }

    return signals


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>NIFTY 50 Stock Rating System</h1>"


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/tickers")
async def get_tickers():
    return {"tickers": [{"symbol": t, "name": NIFTY50_NAMES[t]} for t in NIFTY50_TICKERS]}


@app.get("/api/ratings")
async def get_ratings():
    df = _load_ratings()
    if df.empty:
        return {"stocks": [], "message": "No ratings available. Run the pipeline first."}

    stocks = []
    for ticker, row in df.iterrows():
        stock = {
            "ticker": str(ticker),
            "name": str(ticker).replace(".NS", ""),
            "composite_rating": int(row.get("composite_rating", 0)),
        }
        for col in df.columns:
            if col.startswith("pred_") or col.startswith("rating_") or col.startswith("pct_"):
                val = row[col]
                stock[col] = round(float(val), 6) if pd.notna(val) else None
        stocks.append(stock)

    return {"stocks": stocks, "count": len(stocks)}


@app.get("/api/stock/{ticker}")
async def get_stock_detail(ticker: str):
    clean = ticker.replace(".NS", "").upper()
    full_ticker = f"{clean}.NS"

    raw = _load_raw_data(full_ticker)
    if raw.empty:
        raise HTTPException(status_code=404, detail=f"Stock {clean} not found")

    # Price history (full available)
    price_data = []
    for date, row in raw.iterrows():
        price_data.append({
            "date": str(date)[:10],
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    # Rating
    ratings = _load_ratings()
    rating_info = {}
    for idx in ratings.index:
        if clean in str(idx):
            rr = ratings.loc[idx]
            rating_info = {"composite_rating": int(rr.get("composite_rating", 0))}
            for col in ratings.columns:
                if col.startswith("pred_") or col.startswith("rating_"):
                    val = rr[col]
                    rating_info[col] = round(float(val), 6) if pd.notna(val) else None
            break

    # Predictions
    predictions = {}
    pred_dir = get_path(config, "data.predictions_dir")
    for horizon in config["targets"]["forward_days"]:
        target = f"fwd_return_{horizon}"
        path = pred_dir / f"lgbm_{target}_predictions.csv"
        if path.exists():
            pred_df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "ticker" in pred_df.columns:
                sp = pred_df[pred_df["ticker"] == full_ticker].tail(60)
                if not sp.empty:
                    predictions[f"{horizon}d"] = {
                        "dates": [str(d)[:10] for d in sp.index],
                        "predicted": sp["predicted"].round(6).tolist(),
                        "actual": sp["actual"].round(6).tolist(),
                    }

    return {
        "ticker": clean,
        "full_ticker": full_ticker,
        "price_data": price_data,
        "rating": rating_info,
        "predictions": predictions,
        "stats": {
            "current_price": round(float(raw["Close"].iloc[-1]), 2),
            "prev_close": round(float(raw["Close"].iloc[-2]), 2) if len(raw) > 1 else None,
            "day_change": round(float(raw["Close"].iloc[-1] - raw["Close"].iloc[-2]), 2) if len(raw) > 1 else 0,
            "day_change_pct": round(float((raw["Close"].iloc[-1] / raw["Close"].iloc[-2] - 1) * 100), 2) if len(raw) > 1 else 0,
            "52w_high": round(float(raw["High"].tail(252).max()), 2),
            "52w_low": round(float(raw["Low"].tail(252).min()), 2),
            "avg_volume": int(raw["Volume"].tail(20).mean()),
            "total_rows": len(raw),
        },
    }


@app.get("/api/stock/{ticker}/candles")
async def get_candles(ticker: str, days: int = 365):
    """Get OHLCV candlestick data for a stock."""
    clean = ticker.replace(".NS", "").upper()
    full_ticker = f"{clean}.NS"

    raw = _load_raw_data(full_ticker)
    if raw.empty:
        raise HTTPException(status_code=404, detail=f"Stock {clean} not found")

    recent = raw.tail(days)
    candles = []
    for date, row in recent.iterrows():
        candles.append({
            "time": str(date)[:10],
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        })

    volumes = []
    for date, row in recent.iterrows():
        close = float(row["Close"])
        opn = float(row["Open"])
        volumes.append({
            "time": str(date)[:10],
            "value": int(row["Volume"]),
            "color": "rgba(16, 185, 129, 0.4)" if close >= opn else "rgba(239, 68, 68, 0.4)",
        })

    return {"ticker": clean, "candles": candles, "volumes": volumes}


@app.get("/api/stock/{ticker}/indicators")
async def get_indicators(ticker: str):
    """Get technical indicator summary for a stock."""
    clean = ticker.replace(".NS", "").upper()
    full_ticker = f"{clean}.NS"

    proc = _load_processed_stock(full_ticker)
    if proc.empty:
        raise HTTPException(status_code=404, detail=f"No processed data for {clean}")

    latest = proc.iloc[-1]
    signals = _compute_indicator_signals(latest)

    # Summary counts
    buy_count = sum(1 for s in signals.values() if s["action"] == "buy")
    sell_count = sum(1 for s in signals.values() if s["action"] == "sell")
    hold_count = sum(1 for s in signals.values() if s["action"] == "hold")
    total = buy_count + sell_count + hold_count

    # Overall recommendation
    if buy_count > sell_count + hold_count:
        overall = "Strong Buy"
    elif buy_count > sell_count:
        overall = "Buy"
    elif sell_count > buy_count + hold_count:
        overall = "Strong Sell"
    elif sell_count > buy_count:
        overall = "Sell"
    else:
        overall = "Hold"

    # Historical indicator data (last 90 days)
    history = {}
    hist_df = proc.tail(90)
    indicator_cols = ["RSI", "MACD", "MACD_Signal", "ADX", "Stoch_K", "Stoch_D",
                      "CCI", "Williams_R", "BB_PctB", "Volatility", "ATR"]
    for col in indicator_cols:
        if col in hist_df.columns:
            vals = hist_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            history[col] = {
                "dates": [str(d)[:10] for d in hist_df.index],
                "values": [round(float(v), 4) for v in vals.values],
            }

    return {
        "ticker": clean,
        "signals": signals,
        "summary": {
            "buy": buy_count,
            "sell": sell_count,
            "hold": hold_count,
            "total": total,
            "overall": overall,
        },
        "history": history,
        "current_price": round(float(latest.get("Close", 0)), 2),
    }


@app.get("/api/indicators/all")
async def get_all_indicators():
    """Get indicator summary for ALL stocks (used on indicators page)."""
    processed_dir = get_path(config, "data.processed_dir")
    path = processed_dir / "features_targets.parquet"
    if not path.exists():
        return {"stocks": []}

    df = pd.read_parquet(path)
    if "Ticker" not in df.columns:
        return {"stocks": []}

    results = []
    for ticker in df["Ticker"].unique():
        stock_df = df[df["Ticker"] == ticker]
        if stock_df.empty:
            continue
        latest = stock_df.iloc[-1]
        signals = _compute_indicator_signals(latest)

        buy_c = sum(1 for s in signals.values() if s["action"] == "buy")
        sell_c = sum(1 for s in signals.values() if s["action"] == "sell")
        hold_c = sum(1 for s in signals.values() if s["action"] == "hold")

        if buy_c > sell_c + hold_c:
            overall = "Strong Buy"
        elif buy_c > sell_c:
            overall = "Buy"
        elif sell_c > buy_c + hold_c:
            overall = "Strong Sell"
        elif sell_c > buy_c:
            overall = "Sell"
        else:
            overall = "Hold"

        results.append({
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "rsi": round(float(latest.get("RSI", 0)), 2) if pd.notna(latest.get("RSI")) else None,
            "macd": round(float(latest.get("MACD", 0)), 4) if pd.notna(latest.get("MACD")) else None,
            "adx": round(float(latest.get("ADX", 0)), 2) if pd.notna(latest.get("ADX")) else None,
            "stoch_k": round(float(latest.get("Stoch_K", 0)), 2) if pd.notna(latest.get("Stoch_K")) else None,
            "cci": round(float(latest.get("CCI", 0)), 2) if pd.notna(latest.get("CCI")) else None,
            "volatility": round(float(latest.get("Volatility", 0)) * 100, 2) if pd.notna(latest.get("Volatility")) else None,
            "atr": round(float(latest.get("ATR", 0)), 2) if pd.notna(latest.get("ATR")) else None,
            "momentum_20": round(float(latest.get("Momentum_20", 0)) * 100, 2) if pd.notna(latest.get("Momentum_20")) else None,
            "drawdown": round(float(latest.get("Drawdown", 0)) * 100, 2) if pd.notna(latest.get("Drawdown")) else None,
            "buy_signals": buy_c,
            "sell_signals": sell_c,
            "hold_signals": hold_c,
            "overall": overall,
            "close": round(float(latest.get("Close", 0)), 2) if pd.notna(latest.get("Close")) else None,
        })

    return {"stocks": results}


@app.get("/api/model/status")
async def get_model_status():
    """Check training status of all models."""
    lgbm_dir = PROJECT_ROOT / "models" / "lgbm"
    tft_dir = PROJECT_ROOT / "models" / "tft"

    lgbm_models = {}
    tft_models = {}

    for horizon in config["targets"]["forward_days"]:
        target = f"fwd_return_{horizon}"
        # LightGBM
        lgbm_path = lgbm_dir / f"lgbm_{target}.joblib"
        lgbm_models[target] = {
            "trained": lgbm_path.exists(),
            "path": str(lgbm_path) if lgbm_path.exists() else None,
            "size_mb": round(lgbm_path.stat().st_size / 1e6, 2) if lgbm_path.exists() else 0,
        }
        # TFT — Darts saves as directory with checkpoints
        tft_path = tft_dir / f"tft_{target}"
        tft_trained = tft_path.exists() and tft_path.is_dir()
        tft_models[target] = {
            "trained": tft_trained,
            "path": str(tft_path) if tft_trained else None,
        }

    return {
        "lgbm": lgbm_models,
        "tft": tft_models,
        "scaler_exists": (PROJECT_ROOT / "models" / "feature_scaler.joblib").exists(),
    }


@app.get("/api/model/accuracy")
async def get_model_accuracy():
    """Get model accuracy metrics and best parameters."""
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    eval_path = tables_dir / "evaluation_summary.csv"

    metrics = []
    if eval_path.exists():
        df = pd.read_csv(eval_path)
        for _, row in df.iterrows():
            metrics.append({
                "target": row.get("target", ""),
                "rmse": round(float(row.get("RMSE", 0)), 6),
                "directional_accuracy": round(float(row.get("Directional_Accuracy", 0)), 2),
                "spearman_ic": round(float(row.get("Spearman_IC", 0)), 4),
            })

    # Load best params if available
    lgbm_dir = PROJECT_ROOT / "models" / "lgbm"
    best_params = {}
    import joblib
    for horizon in config["targets"]["forward_days"]:
        target = f"fwd_return_{horizon}"
        param_path = lgbm_dir / f"best_params_{target}.joblib"
        if param_path.exists():
            best_params[target] = joblib.load(param_path)
        else:
            best_params[target] = config["lgbm"]

    # Current config params
    current_config = {
        "lgbm": config["lgbm"],
        "tft": config["tft"],
        "features": {
            "warmup_days": config["features"]["warmup_days"],
            "scaler": config["features"]["scaler"],
        },
        "split": config["split"],
    }

    # Charts available
    charts_dir = PROJECT_ROOT / "outputs" / "charts"
    charts = []
    if charts_dir.exists():
        for f in sorted(charts_dir.iterdir()):
            if f.suffix == ".png":
                charts.append(f.name)

    return {
        "metrics": metrics,
        "best_params": {k: {pk: round(pv, 6) if isinstance(pv, float) else pv for pk, pv in v.items()} for k, v in best_params.items()},
        "current_config": current_config,
        "charts": charts,
    }


@app.get("/api/charts/{filename}")
async def get_chart(filename: str):
    """Serve evaluation chart images."""
    from fastapi.responses import FileResponse
    charts_dir = PROJECT_ROOT / "outputs" / "charts"
    path = charts_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(str(path), media_type="image/png")
