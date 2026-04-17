# 🚀 NIFTY 50 Stock Rating System

AI-Powered stock analysis using **XGBoost + Temporal Fusion Transformer** ensemble to predict multi-horizon forward returns and rate NIFTY 50 stocks on a 1-10 scale.

## 📊 Core ML Features

- **25+ Technical Indicators** — RSI, MACD, EMA, ATR, Bollinger Bands, ADX, Stochastic, CCI, OBV, Williams %R, and derivatives
- **Dual-Model Ensemble** — XGBoost (tabular) + Temporal Fusion Transformer (time-series)
- **Multi-Horizon Predictions** — 10-day, 20-day, 30-day forward returns
- **IC-Weighted Ensemble** — Dynamic model weighting via Spearman rank correlation
- **Statistical Rating Scale** — Percentile-based composite 1-10 ratings
- **Interactive Dashboard** — FastAPI backend with real-time predictions

## Quick Start

### 1. Create Virtual Environment
```bash
cd "Nifty-50 stock rating system"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline (step-by-step)

```bash
# Step 1: Fetch 10 years of NIFTY 50 data
python -m scripts.01_fetch_data

# Step 2: Generate features + targets
python -m scripts.02_build_features

# Step 3: Train XGBoost models
python -m scripts.03_train_xgboost

# Step 3 (optional): With Optuna hyperparameter tuning
python -m scripts.03_train_xgboost --optimize

# Step 4: Train TFT models
python -m scripts.04_train_tft

# Step 5: Ensemble + Generate ratings
python -m scripts.05_ensemble_rate

# Step 6: Evaluation charts
python -m scripts.06_evaluate
```

### 4. Launch Web App
```bash
uvicorn app.api:app --reload --port 8000
```
Open http://localhost:8000 in your browser.

## Project Structure

```
├── config.yaml              # Model & data parameters
├── requirements.txt
├── src/
│   ├── data/                # Data fetching & loading
│   ├── features/            # 25+ technical indicators & scaling
│   ├── models/              # XGBoost, TFT, ensemble architecture
│   ├── rating/              # Percentile-based rating system
│   ├── evaluation/          # Performance metrics & visualization
│   └── utils/               # Configuration & logging
├── scripts/                 # Pipeline scripts (01-06)
├── app/                     # FastAPI backend & UI
├── data/                    # Raw & processed data
├── models/                  # Trained model artifacts
└── outputs/                 # Generated charts & reports
```

## Configuration

All parameters are in `config.yaml`:
- Data dates and file formats
- Feature windows (RSI, MACD, EMA, ATR, Bollinger Bands, etc.)
- XGBoost hyperparameters & Optuna trials
- TFT architecture (attention heads, layers, dropout)
- Ensemble weighting strategy
- Rating scale parameters

## Models & Performance

### XGBoost (Tabular Model)
**Architecture & Training:**
- **Estimators**: 2000
- **Learning Rate**: 0.01
- **Max Depth**: 7
- **Subsample**: 0.8
- **Colsample by Tree**: 0.8
- **Min Child Weight**: 1
- **Gamma (Min Loss Reduction)**: 0
- **L1 Regularization (Alpha)**: 0
- **L2 Regularization (Lambda)**: 1
- **Early Stopping Rounds**: 100
- **Optuna Trials**: 50
- **Random State**: 42

**Performance Metrics:**
```
Horizon    RMSE      Directional Accuracy    Spearman IC
10-Day:    0.063867  55.3%                   0.0788
20-Day:    0.090662  57.4%                   0.0616
30-Day:    0.111565  59.5%                   0.0754
```

### Temporal Fusion Transformer (TFT)
**Architecture & Training:**
- **Input Sequence Length**: 30 days
- **Output Sequence Length**: 30 days
- **Hidden Size**: 64 units
- **Attention Heads**: 4
- **Dropout Rate**: 0.1
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Batch Size**: 64
- **Max Epochs**: 100
- **Learning Rate**: 0.001
- **Early Stopping Patience**: 10 epochs
- **Random State**: 42

**Training Coverage:**
- Trained on top 5 NIFTY stocks (market cap weighted)
- 3 prediction horizons (10d, 20d, 30d)
- PyTorch Lightning backend with Tesla GPU support

### Ensemble Strategy
**Method**: IC-Weighted (Information Coefficient)
- **Lookback Window**: 60 days
- **Weight Formula**: Absolute Spearman IC / Total IC
- **Dynamic Rebalancing**: Per prediction date
- **Components**: XGBoost + TFT predictions

## Data Pipeline

### Training Dataset
- **Total Samples**: 125,869 trading days
- **Companies**: 50 NIFTY stocks (2015-2025)
- **Time Period**: 10 years of historical data
- **Train/Val/Test Split**: 70% / 15% / 15%

### Features (33 Technical Indicators)
```
Momentum Indicators:    RSI, MACD, MACD Signal, MACD Histogram
                       Momentum, Rate of Change
Trend Indicators:       EMA (9, 21, 50, 200), SMA (10, 20, 50, 200)
                       ADX
Volatility Indicators:  ATR, Bollinger Bands (upper, lower, middle)
                       Standard Deviation
Volume Indicators:      OBV, Volume Z-Score
Oscillators:            Stochastic %K, Stochastic %D
                       Williams %R, CCI
Statistical:            Log Volume, Drawdown
```

### Targets (3 Forward Returns)
- `fwd_return_10`: 10-day forward return
- `fwd_return_20`: 20-day forward return
- `fwd_return_30`: 30-day forward return

## Model Outputs

### Stock Ratings
- **File**: `outputs/tables/stock_ratings.csv`
- **Scale**: 1-10 (percentile-based)
- **Companies**: All 50 NIFTY stocks
- **Update Frequency**: Daily (after market close)

### Predictions
- **XGBoost**: `data/predictions/xgb_fwd_return_*.csv`
- **Ensemble**: Dynamic IC-weighted combination
- **Formats**: CSV with actual vs predicted values

### Evaluation Charts (21 Total)
- **Scatter Plots**: Predicted vs Actual (3 horizons)
- **IC Charts**: Rolling Information Coefficient (3 horizons)
- **Feature Importance**: Top 20 features per model (3 horizons)

## Web Application

### Dashboard (http://localhost:8000)
- Real-time stock ratings (1-10 scale)
- Multi-horizon predictions
- Technical indicators analysis
- Model performance metrics
- Interactive charts and visualizations

### API Endpoints

**Stock Information:**
```bash
GET /api/stocks                      # List all NIFTY stocks
GET /api/stock/{ticker}             # Stock details & ratings
GET /api/stock/{ticker}/analysis    # Detailed analysis
```

**Model Status:**
```bash
GET /api/model/status               # Training status
GET /api/model/accuracy             # Performance metrics
GET /api/charts/{filename}          # Evaluation visualizations
```

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Try endpoints** directly without code

## Temporal Fusion Transformer (TFT)
- 30-day input window → multi-horizon output
- Multi-head attention mechanism
- Trained via PyTorch Lightning with early stopping
- Uses Darts library for clean API



## Implementation Details

### Technology Stack
- **Python 3.11** — Core language
- **XGBoost 3.2.0** — Gradient Boosting
- **PyTorch Lightning** — Deep Learning framework
- **Darts** — Time-series modeling
- **Optuna** — Hyperparameter optimization
- **FastAPI** — Web server (async)
- **scikit-learn** — Preprocessing & metrics
- **pandas / numpy** — Data manipulation
- **plotly / matplotlib** — Visualization

### Hardware Requirements
- **CPU**: 4+ cores (development)
- **RAM**: 16+ GB (model training)
- **Storage**: 500+ MB (models + data)
- **GPU**: Optional (CUDA for TFT training)

## License

MIT
