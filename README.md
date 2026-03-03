# 📈 AI-Powered Quantitative Stock Rating System (NIFTY-50)

An end-to-end Machine Learning pipeline that predicts multi-horizon stock returns and generates AI-driven ratings for NIFTY-50 companies using:

- 🌳 LightGBM (Gradient Boosting)
- 🔁 LSTM (Sequence Learning)
- 🧠 1D-CNN (Temporal Pattern Recognition)
- 📊 Ensemble Blending
- ⭐ Percentile-based Stock Ratings (1–10)

This project combines traditional financial indicators with deep learning to create a robust stock ranking system.

## 🚀 Project Overview

This system:
1. Downloads 10 years of historical data for NIFTY-50 stocks
2. Engineers 30+ technical and statistical features
3. Trains 3 different ML models:
   - LightGBM (Tree-based)
   - LSTM (Recurrent Neural Network)
   - 1D-CNN (Convolutional Neural Network)
4. Blends predictions across:
   - 10-day returns
   - 20-day returns
   - 30-day returns
5. Generates a composite score
6. Converts it into a rating from ⭐ 1 to ⭐ 10

The result is a fully automated AI stock scoring system.

## 🧠 Why we need ?

Financial markets are:
- Noisy
- Non-linear
- Regime-dependent
- Sequential in nature
A single model cannot capture all behaviors.

This system combines:
| Model     | Strength 
|-----------|----------------------------
| LightGBM  | Tabular feature interactions 
| LSTM      | Long-term temporal dependencies 
| CNN       | Local time-based patterns 

Then blends them for improved robustness.

## 📊 Data Used

- Source: Yahoo Finance (via `yfinance`)
- Universe: NIFTY-50 stocks
- Timeframe: Last 10 years
- Frequency: Daily

## 🛠 Feature Engineering

Over 30 features including:
# 📈 Trend Indicators
- EMA (20, 50)
- SMA (100)
- 252-day rolling mean
- Trend residual

# ⚡ Momentum
- RSI (14)
- MACD Difference
- ROC (10)

# 📉 Volatility
- 20 & 60-day volatility
- ATR (14)
- High-Low range
- Volatility regime detection

# 📦 Volume Features
- Volume change
- Volume z-score
- Price-volume trend
- Divergence metrics

# 🔄 Relative Strength
- RS 50
- RS 200

# 📉 Risk
- Drawdown

## 🎯 Targets

The system predicts:
- `future_return_10d`
- `future_return_20d`
- `future_return_30d`
Multi-horizon prediction allows better stability and ensemble blending.

## 🧪 Model Architecture

# 🌳 LightGBM
- Gradient Boosted Decision Trees
- Early stopping
- Feature scaling
- RMSE optimization

# 🔁 LSTM
- 2 layers
- Hidden size: 64
- Dropout regularization
- Huber loss
- Gradient clipping
- Learning rate scheduler

Sequence length: 30 days

# 🧠 1D-CNN
- Conv1D layers
- Batch normalization
- GELU activation
- Max pooling
- Adaptive average pooling

Designed to capture short-term temporal patterns.


## ⚖️ Ensemble Strategy

Final score is computed as:

Model Weights:
- LGB → 18.52%
- LSTM → 45.38%
- CNN → 36.10%

Horizon Weights:
- 10d → 20%
- 20d → 30%
- 30d → 50%

Final composite score:
Blended prediction × Horizon weight → Summed → Percentile rank → Rating (1–10)

## ⭐ Rating System

Each stock is ranked based on composite score.
Percentile → Rating:
- Top 10% → ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ (10)
- Bottom 10% → ⭐ (1)

Output(added):
- `stock_ratings.csv`
- Visualization chart

## 📊 Evaluation Metrics

For each model and ensemble:
- RMSE (Regression accuracy)
- Directional Accuracy
- Information Coefficient (Spearman Rank Correlation)

## 📦 Installation

```bash
pip install yfinance lightgbm ta torch scikit-learn pandas numpy matplotlib
```

## ▶️ How To Run

```bash
python main_pipeline.py
```

Pipeline will:
1. Download data
2. Train models
3. Generate ratings
4. Save CSV + chart

## 📈 Example Output

```
Rating  Company                        Score
10      Reliance Industries             0.02451
9       HDFC Bank                       0.02173
...
```

## 🔬 Technical Highlights

- Multi-model ensemble
- Sequence modeling
- Feature normalization
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Robust validation split (time-based)
- Production-style modular structure
- Fully reproducible pipeline

## 🎓 Learning Outcomes

This project demonstrates:
- Financial feature engineering
- Deep learning for time series
- Ensemble modeling
- Model evaluation in finance
- Ranking system design
- End-to-end ML pipeline deployment


## 🌟 Future Improvements
- Transformer-based time-series model
- Cross-sectional ranking loss
- Walk-forward backtesting
- Transaction cost modeling
- Portfolio simulation
- Live deployment via API

# 🚀 Pipeline Complete

End-to-end AI stock rating engine built using:
Machine Learning + Deep Learning + Ensemble Intelligence
