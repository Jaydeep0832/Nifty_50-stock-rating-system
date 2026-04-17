# 🚀 NIFTY 50 Stock Rating System

AI-Powered stock analysis using **LightGBM + Temporal Fusion Transformer** ensemble to predict multi-horizon forward returns and rate NIFTY 50 stocks on a 1-10 scale.

## 📊 Core ML Features

- **25+ Technical Indicators** — RSI, MACD, EMA, ATR, Bollinger Bands, ADX, Stochastic, CCI, OBV, Williams %R, and derivatives
- **Dual-Model Ensemble** — LightGBM (tabular) + Temporal Fusion Transformer (time-series)
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

# Step 3: Train LightGBM models
python -m scripts.03_train_lgbm

# Step 3 (optional): With Optuna hyperparameter tuning
python -m scripts.03_train_lgbm --optimize

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
│   ├── models/              # LightGBM, TFT, ensemble architecture
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
- LightGBM hyperparameters & Optuna trials
- TFT architecture (attention heads, layers, dropout)
- Ensemble weighting strategy
- Rating scale parameters

## Models

### LightGBM
- 2000 estimators with early stopping
- Optuna hyperparameter optimization (50 trials)
- Evaluated on RMSE, Directional Accuracy, Spearman IC

### Temporal Fusion Transformer (TFT)
- 30-day input window → multi-horizon output
- Multi-head attention mechanism
- Trained via PyTorch Lightning with early stopping
- Uses Darts library for clean API

### Ensemble
- IC-weighted averaging (rolling 60-day Spearman correlation)
- Dynamic weight allocation based on model performance

## Cloud Integration

Set `cloud.enabled: true` in `config.yaml` and configure your AWS/GCP credentials:

```yaml
cloud:
  enabled: true
  provider: "aws"
  aws:
    bucket: "your-bucket-name"
    region: "ap-south-1"
```

## License

MIT
