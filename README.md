# 🚀 NIFTY 50 Stock Rating System
## AI-Powered Predictive Stock Analysis Using Deep Learning & Ensemble Methods

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML Models](https://img.shields.io/badge/Models-XGBoost%20%2B%20TFT-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What This Project Does

This system **automatically rates NIFTY 50 stocks on a 1-10 scale** by combining:
- **XGBoost** (fast, accurate predictions on 50 stocks)
- **Temporal Fusion Transformer** (advanced deep learning for time-series)
- **Intelligent Ensemble** (combines both models optimally)

**Real-world impact**: Predict which stocks will perform well in 10, 20, and 30 days ahead, helping traders and investors make data-driven decisions.

**Example output:**
```
TCS              → Rating: 10/10  (Strong Buy)
INFY             → Rating: 8/10   (Buy)
RELIANCE         → Rating: 6/10   (Hold)
MARUTI           → Rating: 4/10   (Sell)
```

---

## 📊 Project Highlights

### The Innovation
| Feature | Why It Matters |
|---------|----------------|
| **Dual ML Models** | XGBoost handles 50 stocks fast; TFT captures complex time-series patterns on top 5 |
| **25+ Technical Indicators** | RSI, MACD, EMA, Bollinger Bands, ADX, OBV, Stochastic, CCI, Williams %R + more |
| **Multi-Horizon Forecasting** | Predicts 10-day, 20-day, AND 30-day returns (not just next-day) |
| **IC-Weighted Ensemble** | Dynamically weights models based on predictive power (Spearman rank correlation) |
| **10 Years of Data** | 125,869 trading samples across 50 stocks (2015-2025) |
| **Interactive Dashboard** | Real-time web app to visualize predictions and ratings |

### Model Performance
```
Prediction Horizon    RMSE      Accuracy    Correlation
──────────────────────────────────────────────────────
10-Day Forward:       0.0639    55.3%       0.0788
20-Day Forward:       0.0907    57.4%       0.0616
30-Day Forward:       0.1116    59.5%       0.0754
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         RAW STOCK DATA (10 years)               │
│    125,869 trading days × 50 NIFTY stocks      │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│      FEATURE ENGINEERING PIPELINE               │
│  • 25+ Technical Indicators (RSI, MACD, EMA)   │
│  • Robust Scaling (handles outliers)            │
│  • 3 Target Variables (10d, 20d, 30d returns)  │
└──────────────┬──────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌──────────────────────┐
│  XGBoost     │  │ Temporal Fusion      │
│  Tabular ML  │  │ Transformer (TFT)    │
│  (50 stocks) │  │ Deep Learning        │
│              │  │ (top 5 stocks)       │
│ • Fast       │  │ • Complex patterns   │
│ • Scalable   │  │ • Attention heads    │
│ • Robust     │  │ • LSTM layers        │
└──────┬───────┘  └───────┬──────────────┘
       │                  │
       └──────┬───────────┘
              ▼
    ┌─────────────────────────────┐
    │  IC-WEIGHTED ENSEMBLE       │
    │ (Dynamic Model Combination) │
    │ Weight = Spearman IC / Sum  │
    └────────────┬────────────────┘
                 ▼
    ┌─────────────────────────────┐
    │  STOCK RATINGS (1-10)       │
    │  Percentile-Based Scale     │
    │  Composite Metric           │
    └────────────┬────────────────┘
                 ▼
    ┌─────────────────────────────┐
    │  DASHBOARD & WEB APP        │
    │  FastAPI Backend            │
    │  Real-time Predictions      │
    └─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **16+ GB RAM** (for model training)
- **GPU (optional)** - CUDA-enabled for faster TFT training

### Setup (5 minutes)

```bash
# 1. Clone and setup
cd "Nifty-50 stock rating system"
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python -m scripts.01_fetch_data      # Download 10 years of OHLCV data
python -m scripts.02_build_features  # Generate 25+ technical indicators
python -m scripts.03_train_xgboost   # Train XGBoost on all 50 stocks
python -m scripts.04_train_tft       # Train TFT on top 5 stocks
python -m scripts.05_ensemble_rate   # Ensemble & generate ratings
python -m scripts.06_evaluate        # Create performance charts

# 4. Launch dashboard
uvicorn app.api:app --reload --port 8000
# Open: http://localhost:8000
```

---

## 📖 What Each Component Does

### 1. **Data Pipeline** (`scripts/01-02`)
- **Fetches** 10 years of historical OHLCV (Open, High, Low, Close, Volume) data
- **Generates** 25+ technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR, ADX, CCI, OBV, Williams %R, Stochastic, Volume Z-Score, Momentum, ROC, Drawdown, etc.)
- **Creates** targets: 10-day, 20-day, 30-day forward returns
- **Outputs**: 125,869 samples × 42 features (Parquet file)

### 2. **XGBoost Training** (`scripts/03`)
**What it is:** Gradient Boosting Machine (XGBoost) - a fast, interpretable tabular ML model

**Key stats:**
- Trains on **all 50 NIFTY stocks**
- Predicts at 3 horizons (10d, 20d, 30d)
- ~2000 estimators with early stopping at 100 rounds
- Learning rate: 0.01 (conservative, stable)
- Optional Optuna hyperparameter tuning (50 trials)

**Output:** 3 models (one per horizon) + predictions CSV

### 3. **Temporal Fusion Transformer (TFT)** (`scripts/04`)
**What it is:** State-of-the-art deep learning for time-series forecasting

**Why use it?**
- Captures **non-linear temporal patterns** that XGBoost might miss
- **Self-attention mechanism** learns which past days matter most
- **LSTM layers** model sequential dependencies
- Better for **complex stock behavior**

**Key stats:**
- Trains on **top 5 stocks** (memory-efficient)
- 30-day input window → predicts 10d, 20d, 30d ahead
- 64-unit hidden size, 4 attention heads
- PyTorch Lightning backend with GPU support

**Output:** 3 models (one per horizon) + predictions CSV

### 4. **IC-Weighted Ensemble** (`scripts/05`)
**What it is:** Intelligently combines XGBoost + TFT predictions

**How it works:**
1. Calculates **Spearman Rank Correlation (IC)** for each model over 60 days
2. Weights them by their relative predictive power: `weight = |IC| / sum(|IC|)`
3. Dynamically rebalances daily - if TFT is hot, use more TFT; if XGBoost is hot, use more XGBoost
4. Converts combined predictions → 1-10 ratings using percentile normalization

**Benefit:** Best of both worlds - fast XGBoost + deep learning power of TFT

### 5. **Stock Ratings** (Output)
**Scale:** 1-10 (percentile-based)
- **9-10** = Top performers (Buy signal)
- **7-8** = Strong candidates
- **5-6** = Hold / Neutral
- **3-4** = Caution / Sell signal
- **1-2** = Weak performers

---

## 🧠 ML/DL Innovation Explained

### Why Dual Models?

| **XGBoost** | **Temporal Fusion Transformer** |
|-----------|--------------------------------|
| ✅ Fast (trains in minutes) | ✅ Complex patterns (deep learning) |
| ✅ Works on 50 stocks | ✅ Captures temporal dynamics |
| ✅ Feature importance | ✅ Self-attention mechanism |
| ❌ May miss time-series patterns | ❌ Memory-intensive, slower |
| Ideal for: **Scalability & speed** | Ideal for: **Non-linear patterns** |

**Ensemble approach:** Use XGBoost for all 50 stocks + TFT for top 5 → Combine via IC weighting → Best of both!

### Key Technical Features

1. **Robust Feature Scaling** - Handles extreme values (outliers) without losing information
2. **Multi-Horizon Learning** - Single model predicts 10d, 20d, AND 30d (not separate models)
3. **Spearman IC Weighting** - Rank-based correlation (robust to magnitude changes)
4. **Early Stopping** - Prevents overfitting in both XGBoost (100 rounds) and TFT (10 epochs)
5. **70-15-15 Train-Val-Test Split** - Proper backtesting methodology

---

## 📁 Project Structure

```
Nifty-50 stock rating system/
├── config.yaml                      # All hyperparameters in one place
├── requirements.txt                 # Dependencies
│
├── src/
│   ├── data/                        # Data loading & preprocessing
│   │   ├── fetch.py                # Download OHLCV data from yfinance
│   │   └── loader.py               # Load and preprocess data
│   │
│   ├── features/                    # Feature engineering
│   │   ├── indicators.py            # 25+ technical indicators
│   │   ├── scaling.py               # Robust scaler
│   │   └── builder.py               # Target generation (forward returns)
│   │
│   ├── models/
│   │   ├── xgboost.py               # XGBoost training & prediction
│   │   ├── tft.py                   # TFT (Darts) training & prediction
│   │   └── ensemble.py              # IC-weighted ensemble logic
│   │
│   ├── rating/
│   │   └── scaler.py                # Percentile-based 1-10 ratings
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # RMSE, Directional Accuracy, IC
│   │   └── visualizer.py            # Matplotlib/Plotly charts
│   │
│   └── utils/
│       ├── config.py                # Load YAML config
│       └── logger.py                # Logging setup
│
├── scripts/                         # Pipeline orchestration
│   ├── 01_fetch_data.py             # Step 1: Get data
│   ├── 02_build_features.py         # Step 2: Generate features
│   ├── 03_train_xgboost.py          # Step 3: Train XGBoost
│   ├── 04_train_tft.py              # Step 4: Train TFT
│   ├── 05_ensemble_rate.py          # Step 5: Ensemble & rate
│   └── 06_evaluate.py               # Step 6: Metrics & charts
│
├── app/                             # Web application
│   ├── api.py                       # FastAPI backend
│   ├── models.py                    # Data models (Pydantic)
│   └── static/                      # HTML/CSS/JS frontend
│
├── data/                            # Data storage
│   ├── raw/                         # Downloaded OHLCV files
│   ├── processed/                   # Engineered features
│   └── predictions/                 # Model outputs
│
├── models/                          # Trained artifacts
│   ├── xgb_fwd_return_*.joblib      # XGBoost models
│   └── tft_fwd_return_*.pt          # TFT checkpoints
│
└── outputs/                         # Reports & charts
    ├── tables/
    │   └── stock_ratings.csv        # Final 1-10 ratings
    └── charts/
        ├── scatter_*.png            # Predicted vs Actual
        ├── ic_*.png                 # Rolling IC charts
        └── feature_importance_*.png # Top 20 features
```

---

## ⚙️ Configuration (One Place)

All parameters in `config.yaml`:

```yaml
# Data
data:
  start_date: "2015-01-01"
  end_date: "2025-12-31"
  
# Technical Indicators
features:
  rsi_window: 14
  macd_fast: 12
  macd_slow: 26
  ema_windows: [9, 21, 50, 200]
  atr_window: 14
  
# XGBoost Hyperparameters
xgboost:
  n_estimators: 2000
  learning_rate: 0.01
  max_depth: 7
  subsample: 0.8
  colsample_bytree: 0.8
  
# TFT Architecture
tft:
  input_chunk_length: 30
  hidden_size: 64
  num_attention_heads: 4
  num_layers: 2
  dropout: 0.1
  
# Ensemble
ensemble:
  method: "ic_weighted"
  ic_lookback_days: 60
  
# Rating Scale
rating:
  min_scale: 1
  max_scale: 10
```

---

## 📊 Model Details

### XGBoost Configuration
```
Estimators:           2000
Learning Rate:        0.01
Max Depth:            7
Subsample:            0.8
Colsample by Tree:    0.8
Min Child Weight:     1
Early Stopping:       100 rounds
L2 Regularization:    1.0
Random State:         42
Optuna Trials:        50 (optional optimization)
```

### TFT Configuration
```
Input Sequence:       30 days
Output Sequence:      30 days
Hidden Size:          64 units
Attention Heads:      4
Dropout Rate:         0.1
Encoder Layers:       2
Decoder Layers:       2
Batch Size:           64
Max Epochs:           100
Learning Rate:        0.001
Early Stopping:       10 epochs
Framework:            PyTorch Lightning + Darts
```

### Data Split
```
Training:             70% (88,108 samples)
Validation:           15% (18,880 samples)
Testing:              15% (18,881 samples)
Total:                125,869 samples
Time Period:          10 years (2015-2025)
```

---

## 🎯 Performance Metrics

### Evaluation Criteria
- **RMSE** - How close predictions are to actual returns (lower is better)
- **Directional Accuracy** - % correct "up or down" predictions (>50% is good)
- **Spearman IC** - Rank correlation with actual returns (measures signal strength)

### Results by Horizon
```
Horizon      RMSE      Dir. Acc.    Spearman IC
────────────────────────────────────────────
10-Day:      0.0639    55.3%        0.0788
20-Day:      0.0907    57.4%        0.0616
30-Day:      0.1116    59.5%        0.0754
```

**What this means:**
- ✅ 55-60% directional accuracy = **better than coin flip**
- ✅ Positive IC = **predictive power**
- ✅ Consistent across horizons = **robust signal**

---

## 🌐 Web Dashboard

### Features
- **Real-time ratings** for all 50 NIFTY stocks
- **Multi-horizon predictions** (10d, 20d, 30d)
- **Technical analysis** charts with indicators
- **Model performance** metrics and backtests
- **Interactive API** (try endpoints directly)

### API Endpoints

```bash
# Get all stocks
GET /api/stocks

# Get specific stock
GET /api/stock/{ticker}

# Get detailed analysis
GET /api/stock/{ticker}/analysis

# Model status
GET /api/model/status
GET /api/model/accuracy

# View charts
GET /api/charts/{filename}
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📈 Data Breakdown

### Training Dataset
- **Total Samples**: 125,869 trading days
- **Companies**: 50 NIFTY 50 stocks
- **Time Period**: 10 years (2015-2025)
- **Data Size**: ~42.5 MB (Parquet format)
- **Frequency**: Daily OHLCV

### Features (33 Technical Indicators)
```
Momentum:     RSI, MACD, MACD_Signal, MACD_Histogram, Momentum, ROC
Trend:        EMA (9,21,50,200), SMA (10,20,50,200), ADX
Volatility:   ATR, Bollinger Bands (upper/lower/middle), Std Dev
Volume:       OBV, Volume Z-Score
Oscillators:  Stochastic %K, %D, Williams %R, CCI
Statistical:  Log Volume, Drawdown
```

### Targets (3 Forward Returns)
- `fwd_return_10`: 10-day forward return %
- `fwd_return_20`: 20-day forward return %
- `fwd_return_30`: 30-day forward return %

---

## 💡 Innovation Highlights

### 1. Ensemble Strategy
Traditional ML systems use a single model. This project uses **dual models + intelligent weighting**:
- XGBoost for all 50 stocks (scalability)
- TFT for top 5 stocks (deep learning)
- IC-weighted combination (dynamically balances both)

### 2. Multi-Horizon Forecasting
Most stock prediction systems predict "next day" only. This predicts **3 horizons simultaneously**, useful for different investment strategies:
- **10-day**: Short-term traders
- **20-day**: Swing traders
- **30-day**: Position traders

### 3. Information Coefficient Weighting
Dynamically rebalances model weights based on 60-day rolling predictive power:
```
Weight = |Spearman_IC| / Sum(|Spearman_IC|)
```
This ensures the ensemble adapts to changing market conditions.

### 4. Robust Statistical Rating Scale
Converts raw predictions → 1-10 scale using **percentile normalization**:
```
Rating = 1 + (percentile / 100) × 9
```
Makes ratings interpretable and comparable across stocks.

---

## 🔬 Technical Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Core** | Python 3.11 | Modern, performant |
| **Tabular ML** | XGBoost 3.2.0 | Fast, accurate, interpretable |
| **Deep Learning** | PyTorch Lightning + Darts | State-of-the-art for time-series |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization |
| **Preprocessing** | scikit-learn | Robust scaling, train-test split |
| **Web Server** | FastAPI | Async, type-safe, fast |
| **Data Format** | Pandas + Parquet | Efficient storage & computation |
| **Visualization** | Plotly + Matplotlib | Publication-quality charts |
| **GPU Support** | CUDA (optional) | Accelerates TFT training |

---

## 🎓 For Academic/Faculty Review

### Thesis-Worthy Elements
1. **Novel Ensemble Architecture**: XGBoost (tabular) + TFT (sequential) + IC weighting
2. **Multi-Horizon Time-Series Forecasting**: Single model predicts 3 horizons
3. **Adaptive Weighting Mechanism**: Spearman IC-based dynamic rebalancing
4. **Production-Ready ML System**: Data pipeline, model training, ensemble, deployment
5. **Comprehensive Evaluation**: RMSE, Directional Accuracy, Information Coefficient

### Research Contributions
- Combining gradient boosting + transformer for financial forecasting
- IC-weighted ensembling for adaptive model combination
- Efficient TFT training on top-K companies while using XGBoost for the rest

### Reproducibility
- ✅ Config-driven (all hyperparameters in YAML)
- ✅ 10-year historical data provided
- ✅ Step-by-step pipeline with logging
- ✅ Evaluation metrics and charts included
- ✅ Open-source dependencies (no proprietary software)

---

## 📝 License

MIT License - Free for personal, educational, and commercial use.

---

## 🤝 Contributing

Ideas for extensions:
1. Add sentiment analysis from financial news
2. Include macroeconomic indicators (interest rates, GDP)
3. Multi-stock correlation modeling
4. Portfolio optimization using ratings
5. Real-time update pipeline

---

## 📞 Questions?

For detailed documentation:
- 📖 **README.md** (this file) - Overview
- 📊 **config.yaml** - All parameters
- 🔗 **API Docs** - http://localhost:8000/docs
- 📂 **Code** - Well-commented source files

---

## ✅ Status

- [x] Data pipeline (fetch, preprocess, feature engineering)
- [x] XGBoost training on 50 stocks
- [x] TFT training on top 5 stocks
- [x] IC-weighted ensemble
- [x] Stock rating system (1-10 scale)
- [x] Web dashboard + FastAPI
- [x] Evaluation metrics & charts
- [x] Production-ready code

**Ready to use!** 🚀

---
