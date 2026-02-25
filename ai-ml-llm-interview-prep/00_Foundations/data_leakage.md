# Data Leakage

## What is Data Leakage?

**Plain English:**
Data leakage happens when information from outside the training dataset "leaks" into your model, making it perform unrealistically well during training/testing but fail catastrophically in production.

**Think of it like cheating on an exam:**
- You accidentally see the test questions before the exam
- You ace the test
- But you didn't actually learn anything
- When faced with new questions (real world), you fail

---

## Why This Matters for Interviews

**Data leakage is a favorite interview topic because:**
1. It's a common real-world mistake
2. Tests your understanding of the ML pipeline
3. Shows whether you think about production scenarios
4. Reveals attention to detail

**Interviewers will ask:**
- "How do you prevent data leakage?"
- "Have you ever encountered leakage? How did you detect it?"
- Given a scenario, identify if leakage exists

---

## Types of Data Leakage

### 1. Target Leakage

**Definition:** Features that wouldn't be available at prediction time

**Example 1 - Credit Card Fraud:**
```python
# ❌ WRONG: Using information only available AFTER fraud occurs
features = [
    'transaction_amount',
    'merchant_id',
    'fraud_investigation_started',  # ← LEAKAGE!
    'fraud_report_filed'             # ← LEAKAGE!
]

# Problem: You only know these AFTER determining fraud
# At prediction time, these don't exist yet!
```

**Example 2 - Hospital Readmission:**
```python
# ❌ WRONG
features = [
    'initial_diagnosis',
    'treatment_plan',
    'patient_satisfaction_score',    # ← LEAKAGE!
    'discharge_summary'               # ← LEAKAGE!
]

# Problem: Satisfaction and discharge summary only exist
# AFTER the hospital visit you're trying to predict
```

**Example 3 - Delivery Time Prediction:**
```python
# ❌ WRONG
features = [
    'package_weight',
    'distance',
    'actual_route_taken',           # ← LEAKAGE!
    'delivery_confirmation_time'    # ← LEAKAGE!
]

# You don't know the actual route until AFTER delivery!
```

**How to detect:**
- Ask: "Would this feature be available at prediction time?"
- If a feature is too predictive (e.g., 0.99 feature importance), investigate

### 2. Train-Test Contamination

**Definition:** Information from test set influences training

**Example 1 - Scaling Before Split:**
```python
# ❌ WRONG: Scaling on entire dataset
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)  # Uses ALL data
X_train, X_test = train_test_split(X_scaled)

# Problem: Test set statistics (mean, std) influenced training data
```

**✅ CORRECT:**
```python
X_train, X_test = train_test_split(X)  # Split FIRST

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)         # Transform test using train stats
```

**Example 2 - Feature Selection on Entire Dataset:**
```python
# ❌ WRONG
from sklearn.feature_selection import SelectKBest

# Select features using ALL data
selector = SelectKBest(k=10).fit(X, y)  # Sees test data!
X_selected = selector.transform(X)
X_train, X_test = train_test_split(X_selected)
```

**✅ CORRECT:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Select features using ONLY training data
selector = SelectKBest(k=10).fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

**Example 3 - Filling Missing Values:**
```python
# ❌ WRONG: Using mean from entire dataset
X['age'].fillna(X['age'].mean(), inplace=True)  # Includes test data
X_train, X_test = train_test_split(X)

# ✅ CORRECT: Use mean from training data only
X_train, X_test = train_test_split(X)
train_age_mean = X_train['age'].mean()
X_train['age'].fillna(train_age_mean, inplace=True)
X_test['age'].fillna(train_age_mean, inplace=True)
```

### 3. Temporal Leakage

**Definition:** Using future information to predict the past

**Example 1 - Time Series Forecasting:**
```python
# ❌ WRONG: Random split for time series
X_train, X_test = train_test_split(df, test_size=0.2)  # Random!

# Problem: Training on future data to predict past
# Jan data in test, Dec data in train → future predicts past!
```

**✅ CORRECT:**
```python
# Split by time
split_date = '2023-06-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Or use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
```

**Example 2 - Feature Engineering with Future Data:**
```python
# ❌ WRONG: Rolling average includes future
df['sales_7day_avg'] = df['sales'].rolling(window=7, center=True).mean()

# center=True uses 3 days before + 3 days after
# You don't know future sales at prediction time!
```

**✅ CORRECT:**
```python
# Only use past data
df['sales_7day_avg'] = df['sales'].rolling(window=7).mean()  # Only looks back
```

### 4. Leakage via Duplicates

**Definition:** Same samples in both train and test

**Example:**
```python
# Dataset has duplicate rows (same person, different ID)
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'features': [...],
    'target': [0, 1, 0, 1, 0]
})

# Row 2 and 4 are actually the same person (duplicates)
# Random split might put one in train, one in test
# Model "memorizes" this person and looks artificially good
```

**Solution:**
```python
# Check for duplicates before splitting
print(df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

# Or group by person ID before splitting
```

### 5. Leakage via Group Structure

**Definition:** Related samples split across train/test

**Example - Medical Data:**
```python
# Patient visits hospital multiple times
# Visit 1, 2 → training set
# Visit 3 → test set

# Problem: Model learns patient-specific patterns
# "Memorizes" patients from visit 1,2 and applies to visit 3
# Doesn't generalize to NEW patients
```

**Solution:**
```python
from sklearn.model_selection import GroupShuffleSplit

# Ensure same patient is entirely in train OR test, not both
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)
for train_idx, test_idx in splitter.split(X, y, groups=patient_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## Real-World Examples

### Example 1: E-commerce Purchase Prediction

**Scenario:** Predict if user will purchase

**❌ Data Leakage:**
```python
features = [
    'user_id',
    'product_viewed',
    'time_spent_on_page',
    'items_in_cart',
    'payment_method_entered',    # ← LEAKAGE!
    'shipping_address_provided'  # ← LEAKAGE!
]

# Payment and shipping only provided DURING purchase
# Not available when predicting IF they'll purchase
```

**✅ Correct Features:**
```python
features = [
    'user_previous_purchases',
    'product_category',
    'time_spent_browsing',
    'cart_additions',
    'historical_conversion_rate'
]
```

### Example 2: Loan Default Prediction

**❌ Data Leakage:**
```python
features = [
    'loan_amount',
    'income',
    'credit_score',
    'number_of_missed_payments',  # ← LEAKAGE!
    'account_closed_flag'         # ← LEAKAGE!
]

# Missed payments happen AFTER loan is issued
# Account closure happens AFTER default
```

**✅ Correct:**
```python
features = [
    'loan_amount',
    'income',
    'credit_score',
    'previous_loan_history',      # Historical only
    'debt_to_income_ratio'
]
```

### Example 3: Supply Chain Delay Prediction

**❌ Leakage:**
```python
# Predicting shipment delays
features = [
    'origin',
    'destination',
    'package_weight',
    'actual_route',              # ← LEAKAGE!
    'weather_at_destination',    # ← LEAKAGE (if using future weather)
    'customer_complaint_filed'   # ← LEAKAGE!
]
```

**✅ Correct:**
```python
features = [
    'origin',
    'destination',
    'package_weight',
    'planned_route',             # Available at shipping time
    'weather_forecast',          # Forecast, not actual
    'historical_delay_rate_for_route'
]
```

---

## How to Detect Data Leakage

### 1. Suspiciously High Performance

```python
# Training accuracy: 99.9%
# Test accuracy: 99.8%

# 🚨 RED FLAG for most real-world problems
# Investigate feature importance
```

### 2. Feature Importance Analysis

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Check feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head())

# If one feature has importance > 0.5, investigate!
# Ask: "Would this be available at prediction time?"
```

### 3. Sudden Performance Drop in Production

```python
# Test accuracy: 95%
# Production accuracy: 60%

# Classic sign of leakage:
# Model learned from leaked features that don't exist in production
```

### 4. Check for Temporal Consistency

```python
# For time series: ensure proper time-based split
train_dates = train_df['date'].min(), train_df['date'].max()
test_dates = test_df['date'].min(), test_df['date'].max()

print(f"Train: {train_dates}")
print(f"Test: {test_dates}")

# Test dates should be AFTER train dates
assert test_dates[0] > train_dates[1], "Temporal leakage detected!"
```

### 5. Cross-Validation Sanity Check

```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# For time series
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

# If CV score much worse than train-test split → might have leakage in split
```

---

## Prevention Checklist

### ✅ Before Training

- [ ] **Understand the business problem**
  - When will predictions be made?
  - What information is available at that time?

- [ ] **Split data FIRST** (before any preprocessing)
  ```python
  # Correct order:
  # 1. Split
  # 2. Preprocess (using only train stats)
  # 3. Train
  ```

- [ ] **Check for temporal ordering** (if time series)
  - Use time-based splits, not random

- [ ] **Identify groups/clusters** (patients, users, etc.)
  - Use GroupKFold if samples are related

- [ ] **Remove duplicates**

### ✅ During Feature Engineering

- [ ] **Ask for each feature:** "Will this be available at prediction time?"

- [ ] **Use only historical data**
  - Rolling averages: look back only
  - Aggregations: from past only

- [ ] **Be careful with target encoding**
  - Can leak if not done properly

### ✅ After Training

- [ ] **Analyze feature importance**
  - Investigate suspiciously high importance

- [ ] **Check performance is realistic**
  - 99% accuracy on fraud detection? Probably leakage.

- [ ] **Simulate production environment**
  - Can you make predictions with only available data?

---

## Interview Questions

### Q1: "Have you dealt with data leakage? Give an example."

**Sample Answer:**
"Yes, I encountered leakage in a customer churn prediction project. We initially achieved 94% accuracy, which seemed great. However, I noticed one feature—'customer_service_calls_last_month'—had 0.7 feature importance.

Upon investigation, I realized we were including support calls that happened AFTER customers had already decided to churn (they were calling to cancel). This inflated our performance.

The fix: I changed the feature to 'customer_service_calls_previous_months' excluding the current month, and ensured a proper time-based split. Accuracy dropped to 78%, which was realistic. The model performed much better in production."

### Q2: "How do you prevent leakage in time series?"

**Answer:**
"Three key practices:

1. **Time-based splits**: Always split by time, never random. Test set must be chronologically after training set.

2. **Rolling features**: Use only past data. For example, `rolling(7).mean()` not `rolling(7, center=True).mean()`.

3. **Cross-validation**: Use `TimeSeriesSplit`, which respects temporal order, not regular k-fold.

I also document the 'knowledge cutoff date' for each feature to ensure we're not using future information."

### Q3: "What's wrong with this code?"

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
```

**Answer:**
"This has train-test contamination. The scaler is fit on the entire dataset (including test data), so the test set statistics (mean, std) influence the training data transformation. This causes leakage.

The correct approach:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)  # Split first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)         # Transform with train stats
```

### Q4: "You achieve 98% accuracy in testing but 65% in production. What could be wrong?"

**Answer:**
"This is a classic symptom of data leakage. Possible causes:

1. **Target leakage**: Features available in training but not at prediction time
2. **Train-test contamination**: Information from test set leaked into training
3. **Temporal leakage**: Used future information to predict past
4. **Different distributions**: Test set doesn't represent production data

To diagnose:
- Review feature engineering pipeline for time-awareness
- Check feature importance for suspiciously predictive features
- Ensure proper train-test split methodology
- Verify test set represents production distribution

I'd start by auditing each feature: 'Is this available at prediction time in production?'"

---

## Key Takeaways

1. **Data leakage makes models look artificially good but fail in production**
2. **Always ask: "Is this feature available at prediction time?"**
3. **Split data BEFORE any preprocessing** (scaling, feature selection, etc.)
4. **Time series requires special care** (time-based splits, backward-looking features)
5. **High performance can be a red flag** (investigate feature importance)
6. **Prevention is easier than detection** (build good habits in your pipeline)

---

**Next:** [Feature Engineering Fundamentals](./feature_engineering_fundamentals.md) | **Back:** [README](./README.md)
