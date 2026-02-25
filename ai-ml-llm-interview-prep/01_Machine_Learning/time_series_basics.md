# Time Series Basics

Time series data is a sequence of data points indexed in time order (e.g., daily sales, hourly server CPU load). Predicting the future based on this past data is one of the most common and difficult tasks in enterprise ML.

---

## 1. Components of a Time Series
A time series is mathematically decomposed into three distinct components:
1.  **Trend:** The long-term progression of the series (e.g., DHL package volume is generally increasing year over year).
2.  **Seasonality:** Repeating, predictable patterns that occur at fixed intervals. (e.g., Volume spikes every December for Christmas, and drops every Sunday).
3.  **Residual (Noise/Irregular):** The random, unpredictable variance left over after removing the trend and seasonality.

## 2. Stationarity (The Golden Rule of Classic Forecasting)
Traditional statistical models (like ARIMA) mathematically require the time series to be **Stationary** before they can predict anything.
*   **What is it?** A stationary series has a constant Mean and constant Variance over time. It has no trend and no seasonality.
*   **Why?** If the mean is constantly moving upward, the math cannot lock onto a reliable pattern.
*   **How to achieve it (Differencing):** Instead of predicting the actual value, you predict the *change* from the previous value. `Value(t) = Actual(t) - Actual(t-1)`. If it's still not stationary, you difference it again. You must also take the `Log` to stabilize expanding variance.
*   **Testing it:** Use the **Augmented Dickey-Fuller (ADF) Test**.

## 3. Classic Statistical Models
*   **ARIMA (AutoRegressive Integrated Moving Average):**
    *   *AR (AutoRegressive):* Predicts the next point based on a linear combination of previous points (Lags).
    *   *I (Integrated):* The differencing steps applied to make it stationary.
    *   *MA (Moving Average):* Predicts the next point based on past forecast *errors*.
*   **SARIMA:** Adds a Seasonal component to ARIMA.
*   *Pros:* Great for univariate (single variable) forecasting. Highly interpretable.
*   *Cons:* Cannot easily ingest external features (like "Is it raining tomorrow?") without moving to ARIMAX. Struggles with highly non-linear patterns.

## 4. Modern Machine Learning for Time Series

### Tree-Based Models (XGBoost / LightGBM)
The current champions of Kaggle time-series competitions.
*   **The Trick:** XGBoost has no inherent concept of time. You must manually create "Lag Features" (e.g., `sales_yesterday`, `sales_7_days_ago`) and "Rolling Features" (e.g., `7_day_moving_average`) as standard tabular columns.
*   **Pros:** Can easily ingest hundreds of external features (weather, holidays, categorical IDs like Store_ID). Fast to train.

### Prophet (by Meta)
An additive regression model where non-linear trends are fit with yearly, weekly, and daily seasonality.
*   **Pros:** Works directly out of the box with messy data (missing values, large outliers). Handles holidays incredibly well natively. Very user-friendly.

### Deep Learning
*   **LSTMs:** Good for continuous, high-frequency signals.
*   **Temporal Fusion Transformers (TFT):** The state-of-the-art for enterprise forecasting. It natively combines:
    1. Static Metadata (e.g., Store Location).
    2. Known Future Inputs (e.g., Upcoming Holidays, Weather Forecast).
    3. Unknown Past Inputs (e.g., Historical Foot Traffic).

## Interview Trap: Cross-Validation
(See `cross_validation.md`). Never, ever use random K-Fold cross-validation on time series data. You will leak future data into the past. Always use **Time-Series Split (Expanding Window or Sliding Window)**.