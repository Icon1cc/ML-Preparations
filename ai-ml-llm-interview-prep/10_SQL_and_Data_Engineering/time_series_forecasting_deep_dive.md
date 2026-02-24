# Time Series Forecasting Deep Dive

## Foundations
Time series require order-aware methods due to trend, seasonality, and autocorrelation.

## Stationarity
Stationary series have stable moments over time.
Tests:
- ADF (unit-root null)
- KPSS (stationarity null)

## Classical models
- ARIMA/SARIMA for linear temporal structures.
- ETS/Holt-Winters for trend-seasonality smoothing.
- Prophet for decomposable trend/seasonality with holidays.

## ML framing
Convert to supervised learning with:
- lag features
- rolling statistics
- calendar features
Then train LightGBM/XGBoost.

## Deep models
- LSTM/GRU for sequence dependencies.
- TCN for efficient long receptive fields.
- TFT/N-BEATS/PatchTST for modern multi-horizon forecasting.

## Validation
Always use walk-forward validation.
Random splits leak future context.

```mermaid
flowchart LR
    A[Historical window 1] --> B[Train]
    B --> C[Validate future slice 1]
    C --> D[Expand window]
    D --> E[Train]
    E --> F[Validate slice 2]
```

## Metrics
- MAE, RMSE
- MAPE/SMAPE
- MASE
- interval coverage for probabilistic forecasts

## Hierarchical forecasting
Need coherence across levels (global/country/region/depot).
Use reconciliation (top-down, bottom-up, MinT).

## Interview questions
1. Why random CV is invalid for time series?
2. ARIMA vs Prophet vs LightGBM tradeoffs?
3. How handle cold start series?
