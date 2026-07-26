# NIFTY 50 Stock Rating System - Quick Start Guide

## For the Impatient (5 Minutes)

### What You Get
- 🎯 Stock ratings on 1-10 scale for all 50 NIFTY stocks
- 📊 Predictions 10, 20, 30 days into the future
- 🚀 Interactive web dashboard
- 📈 Model performance metrics

### The Fastest Path

```bash
# 1. Setup (2 minutes)
cd "Nifty-50 stock rating system"
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install (2 minutes)
pip install -r requirements.txt

# 3. Run pipeline (if pre-trained models exist)
python -m scripts.01_fetch_data      # 1 min
python -m scripts.02_build_features  # 1 min
python -m scripts.03_train_xgboost   # 5 min
python -m scripts.04_train_tft       # 30 min (or skip if exists)
python -m scripts.05_ensemble_rate   # 1 min

# 4. Launch (1 minute)
uvicorn app.api:app --reload --port 8000
# Open: http://localhost:8000
```

**Total Time**: ~45 minutes with TFT training, ~15 minutes if models exist

---

## Understanding the Output

### Stock Ratings (Main Output)
```
Stock       Rating  Interpretation
TCS         10/10   🟢 Strong Buy
INFY        8/10    🟢 Buy
RELIANCE    6/10    🟡 Hold
BAJAJ       4/10    🔴 Sell
MARUTI      2/10    🔴 Strong Sell
```

### What Each Number Means
- **9-10**: Expected to perform well (top 10%)
- **7-8**: Good prospects (top 25%)
- **5-6**: Neutral (middle 50%)
- **3-4**: Weak prospects (bottom 25%)
- **1-2**: Expected to underperform (bottom 10%)

### Multi-Horizon Predictions
Same stock shows different ratings for different timeframes:
```
Stock       10-Day  20-Day  30-Day  Composite
TCS         10      9       8       9
INFY        9       8       7       8
RELIANCE    7       6       5       6
```

Interpretation:
- Declining ratings = momentum weakening
- Stable ratings = consistent signal
- Rising ratings = building strength

---

## System Requirements

### Minimum
- Python 3.11
- 8 GB RAM
- 1 GB disk space

### Recommended
- Python 3.11
- 16+ GB RAM (for model training)
- 2 GB disk space
- GPU with CUDA (optional, for faster TFT training)

### Supported Platforms
- ✅ Windows 10/11
- ✅ macOS (Intel & Apple Silicon)
- ✅ Linux (Ubuntu, CentOS, etc.)

---

## Step-by-Step Setup

### Step 1: Install Python 3.11

**Windows**:
- Download from python.org
- Check "Add Python to PATH"

**macOS**:
```bash
brew install python@3.11
```

**Linux** (Ubuntu):
```bash
sudo apt-get install python3.11 python3.11-venv
```

### Step 2: Create Virtual Environment

```bash
cd "Nifty-50 stock rating system"
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data)
- xgboost (gradient boosting)
- pytorch, pytorch-lightning (deep learning)
- darts (time-series)
- optuna (hyperparameter tuning)
- fastapi, uvicorn (web server)
- plotly, matplotlib (visualization)

### Step 4: Verify Installation

```bash
python -c "import pandas, xgboost, torch; print('✅ All good!')"
```

---

## Running the Pipeline

### Option A: Full Pipeline (First Time)

```bash
# Step 1: Fetch Data (1-2 minutes)
# Downloads 10 years of OHLCV data for all 50 NIFTY stocks
python -m scripts.01_fetch_data

# Step 2: Build Features (2-3 minutes)
# Creates 33 technical indicators and forward return targets
python -m scripts.02_build_features

# Step 3: Train XGBoost (5-10 minutes)
# Trains XGBoost on all 50 stocks
python -m scripts.03_train_xgboost

# Optional: Optimize hyperparameters (30+ minutes)
python -m scripts.03_train_xgboost --optimize

# Step 4: Train TFT (30-60 minutes)
# Trains Transformer on top 5 stocks
python -m scripts.04_train_tft

# Step 5: Ensemble & Rate (1-2 minutes)
# Combines XGBoost + TFT, generates ratings
python -m scripts.05_ensemble_rate

# Step 6: Evaluate (2-3 minutes)
# Creates performance charts and metrics
python -m scripts.06_evaluate
```

### Option B: Update Predictions (Subsequent Runs)

If models are already trained:

```bash
# Just update data and predictions
python -m scripts.01_fetch_data      # New daily data
python -m scripts.02_build_features  # New features
python -m scripts.05_ensemble_rate   # New ratings
```

### Option C: Skip to Dashboard

If everything is already done:

```bash
# Just launch the web app
uvicorn app.api:app --reload --port 8000
```

---

## Using the Web Dashboard

### Access
Open browser to: **http://localhost:8000**

### Dashboard Features

1. **Stock List** - All 50 NIFTY stocks with ratings
   - Click any stock for detailed view
   - Filter by rating range

2. **Stock Detail Page** - For individual stock
   - Current 1-10 rating
   - 10d, 20d, 30d predictions
   - Technical indicator chart
   - Model performance

3. **Performance Metrics** - Model evaluation
   - RMSE (prediction error)
   - Directional accuracy
   - Information coefficient
   - Per-horizon breakdown

4. **API Documentation** - At `/docs`
   - See all available endpoints
   - Try endpoints interactively
   - See request/response formats

### Example API Calls

```bash
# Get all stocks (from terminal)
curl http://localhost:8000/api/stocks

# Get specific stock
curl http://localhost:8000/api/stock/TCS

# Get model status
curl http://localhost:8000/api/model/status
```

---

## Configuration Quick Reference

Edit `config.yaml` to change settings:

```yaml
# Data
data:
  start_date: "2015-01-01"  # Historical start
  end_date: "2025-12-31"    # Historical end

# Technical indicators (feature engineering)
features:
  rsi_window: 14
  macd_fast: 12
  macd_slow: 26

# XGBoost (gradient boosting)
xgboost:
  n_estimators: 2000    # Number of trees
  learning_rate: 0.01   # How fast to learn
  max_depth: 7          # Tree depth
  subsample: 0.8        # Row sampling

# TFT (transformer)
tft:
  input_chunk_length: 30
  hidden_size: 64
  num_attention_heads: 4

# Ensemble
ensemble:
  method: "ic_weighted"
  ic_lookback_days: 60
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution**: Reinstall requirements
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "CUDA out of memory" (GPU error)

**Solution**: Use CPU instead (slower but works)
```bash
# In config.yaml, set:
# device: "cpu"
```

### Issue: "KeyError: 'Close'" (Data error)

**Solution**: Check data was downloaded
```bash
ls data/raw/  # Should have files like RELIANCE.NS.csv
python -m scripts.01_fetch_data  # Re-download
```

### Issue: TFT training very slow

**Solutions**:
1. Use GPU (CUDA enabled): Much faster
2. Reduce batch size in config.yaml: `batch_size: 32`
3. Train only 1 stock: Comment out others in script

### Issue: "Connection refused" on dashboard

**Solution**: Make sure server is running
```bash
uvicorn app.api:app --reload --port 8000
```

### Issue: Port 8000 already in use

**Solution**: Use different port
```bash
uvicorn app.api:app --reload --port 8001
# Then open: http://localhost:8001
```

---

## Performance Tips

### Make Training Faster

1. **Use GPU for TFT**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Reduce training data**
   ```yaml
   # In config.yaml
   data:
     start_date: "2020-01-01"  # Start from 2020 instead of 2015
   ```

3. **Train fewer stocks**
   ```python
   # In scripts/04_train_tft.py, reduce:
   top_k_stocks = 3  # Instead of 5
   ```

### Make Predictions Faster

1. **Reduce feature window sizes** - But may hurt accuracy
2. **Use simpler XGBoost** - Lower `max_depth`, fewer `n_estimators`
3. **Skip TFT** - Use only XGBoost for speed

---

## Understanding Each Script

### 01_fetch_data.py
**What**: Downloads historical stock prices
**Input**: Date range from config.yaml
**Output**: CSV files in `data/raw/`
**Time**: ~1-2 minutes
**Can Skip?**: No (need raw data)

### 02_build_features.py
**What**: Creates 33 technical indicators
**Input**: Raw OHLCV data
**Output**: `data/processed/features_targets.parquet`
**Time**: ~2-3 minutes
**Can Skip?**: No (need features for models)

### 03_train_xgboost.py
**What**: Trains XGBoost on all 50 stocks
**Input**: Feature data
**Output**: 3 models in `models/` directory
**Time**: ~5-10 minutes (30+ with --optimize)
**Can Skip?**: Only if models exist
**Flags**: `--optimize` for hyperparameter tuning

### 04_train_tft.py
**What**: Trains Temporal Fusion Transformer
**Input**: Feature data
**Output**: 3 TFT models, ~1.4 MB each
**Time**: ~30-60 minutes
**Can Skip?**: Yes (XGBoost alone is sufficient)
**GPU**: Much faster with CUDA

### 05_ensemble_rate.py
**What**: Combines XGBoost + TFT, generates ratings
**Input**: Model predictions
**Output**: `outputs/tables/stock_ratings.csv`
**Time**: ~1-2 minutes
**Can Skip?**: No (this is final output)

### 06_evaluate.py
**What**: Creates performance charts
**Input**: Predictions and actuals
**Output**: Charts in `outputs/charts/`
**Time**: ~2-3 minutes
**Can Skip?**: Yes (for faster iteration)

---

## Model Outputs Explained

### Stock Ratings CSV
**File**: `outputs/tables/stock_ratings.csv`
```
ticker,composite_rating,10d_rating,20d_rating,30d_rating
TCS,9.8,10,9,8
INFY,7.9,8,8,7
RELIANCE,5.8,6,6,5
...
```

### Predictions CSV
**File**: `data/predictions/xgb_fwd_return_10.csv`
```
date,ticker,actual_return,predicted_return
2025-12-01,TCS,0.032,0.028
2025-12-01,INFY,0.018,0.015
...
```

### Evaluation Charts
**Location**: `outputs/charts/`
```
scatter_10day.png          - Predicted vs Actual (10d)
scatter_20day.png          - Predicted vs Actual (20d)
ic_rolling_10day.png       - Information Coefficient over time
feature_importance_10day.png - Top 20 features
```

---

## Next Steps After Setup

1. **Understand the Data**
   ```bash
   # See what's in the processed data
   python -c "import pandas as pd; df = pd.read_parquet('data/processed/features_targets.parquet'); print(df.head())"
   ```

2. **Check Model Performance**
   - Open `http://localhost:8000/api/model/accuracy`
   - Verify RMSE, accuracy, IC metrics

3. **Analyze Stock Ratings**
   ```bash
   # View top-rated stocks
   python -c "import pandas as pd; df = pd.read_csv('outputs/tables/stock_ratings.csv'); print(df.nlargest(5, 'composite_rating'))"
   ```

4. **Experiment with Config**
   - Try different technical indicator parameters
   - Adjust XGBoost hyperparameters
   - Run `python -m scripts.03_train_xgboost --optimize`

5. **Deploy to Production**
   - Keep dashboard running: `uvicorn app.api:app --host 0.0.0.0 --port 8000`
   - Schedule daily updates via cron/Task Scheduler
   - Monitor model performance weekly

---

## Getting Help

### Check Logs
```bash
# Each script logs to console and file
# Logs are in project root: *.log files
```

### Read Inline Documentation
```bash
# Code has docstrings
python -c "from src.models.xgboost import XGBoostTrainer; help(XGBoostTrainer)"
```

### API Documentation
```
http://localhost:8000/docs
```

### Project Documentation
- README.md - Overview
- NIFTY_PROJECT_REPORT.md - Detailed explanation
- Code comments - Implementation details

---

## Common Use Cases

### Use Case 1: "I want to see which stocks to buy today"
```bash
# 1. Get latest data
python -m scripts.01_fetch_data

# 2. Update predictions
python -m scripts.02_build_features
python -m scripts.05_ensemble_rate

# 3. Check ratings
cat outputs/tables/stock_ratings.csv | head -20

# 4. Launch dashboard to see visually
uvicorn app.api:app --reload --port 8000
```

### Use Case 2: "I want to improve model accuracy"
```bash
# 1. Optimize XGBoost hyperparameters
python -m scripts.03_train_xgboost --optimize

# 2. Train TFT with more data
# Edit config.yaml: start_date: "2010-01-01"

# 3. Evaluate new models
python -m scripts.06_evaluate
```

### Use Case 3: "I want to understand how the model works"
```bash
# 1. Check feature importance
# See: outputs/charts/feature_importance_*.png

# 2. Analyze predictions
python -c "
import pandas as pd
preds = pd.read_csv('data/predictions/xgb_fwd_return_10.csv')
print(preds.describe())
"

# 3. Study the code
# Read: src/models/xgboost.py, src/models/tft.py
```

### Use Case 4: "I want to use this for my portfolio"
```bash
# 1. Get ratings for all 50 stocks
curl http://localhost:8000/api/stocks

# 2. Filter for high-rated stocks
# Rating > 7: Buy candidates
# Rating > 8: Strong buy
# Rating < 4: Avoid

# 3. Check consensus
# If multiple horizons agree → confidence is high
```

---

## Cheat Sheet

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python -m scripts.0{1,2,3,4,5,6}  # Linux/Mac
python -m scripts.01_fetch_data && python -m scripts.02_build_features && ...  # Windows

# Quick predictions
python -m scripts.01_fetch_data && python -m scripts.02_build_features && python -m scripts.05_ensemble_rate

# Launch dashboard
uvicorn app.api:app --reload

# View ratings
cat outputs/tables/stock_ratings.csv

# Check performance
curl http://localhost:8000/api/model/accuracy

# Get specific stock
curl http://localhost:8000/api/stock/TCS
```

---

## Estimated Time Breakdown

| Task | Time | Frequency |
|------|------|-----------|
| Setup (first time) | 5 min | Once |
| Fetch data | 1 min | Daily |
| Build features | 2 min | Daily |
| Train XGBoost | 5-10 min | Weekly |
| Train TFT | 30-60 min | Weekly (optional) |
| Ensemble & rate | 1 min | Daily |
| Evaluate | 2 min | Weekly |
| **Total (first)** | **~60 min** | Once |
| **Total (daily)** | **~5 min** | Every day |
| **Total (weekly)** | **~50 min** | Once/week |

---

## You're Ready! 🚀

You now have everything to:
- ✅ Run the full ML pipeline
- ✅ Train models on 10 years of data
- ✅ Get stock ratings automatically
- ✅ Deploy a web dashboard
- ✅ Use the REST API

**Start with**: `python -m scripts.01_fetch_data`

**Then check**: `http://localhost:8000` (after all scripts)

Good luck! 📊

