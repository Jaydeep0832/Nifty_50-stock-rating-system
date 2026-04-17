---
description: Run the NIFTY 50 ML pipeline end-to-end
---

# NIFTY 50 Rating Pipeline

## Prerequisites
Ensure the virtual environment is activated.

// turbo-all

## Steps

1. Activate virtual environment:
```
.\venv\Scripts\activate
```

2. Fetch stock data (10 years of NIFTY 50 OHLCV):
```
.\venv\Scripts\python.exe -m scripts.01_fetch_data
```

3. Build features and targets (25+ indicators + forward returns):
```
.\venv\Scripts\python.exe -m scripts.02_build_features
```

4. Train LightGBM models (all horizons):
```
.\venv\Scripts\python.exe -m scripts.03_train_lgbm
```

5. (Optional) Train LightGBM with Optuna optimization:
```
.\venv\Scripts\python.exe -m scripts.03_train_lgbm --optimize
```

6. Train TFT models:
```
.\venv\Scripts\python.exe -m scripts.04_train_tft
```

7. Generate ensemble predictions and ratings:
```
.\venv\Scripts\python.exe -m scripts.05_ensemble_rate
```

8. Generate evaluation charts:
```
.\venv\Scripts\python.exe -m scripts.06_evaluate
```

9. Launch web app:
```
.\venv\Scripts\uvicorn.exe app.api:app --reload --port 8000
```

Open http://localhost:8000 in your browser.
