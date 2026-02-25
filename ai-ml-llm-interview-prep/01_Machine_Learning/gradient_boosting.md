# Gradient Boosting: XGBoost, LightGBM, CatBoost

## What is Gradient Boosting? (Plain English)

**Simple Analogy:**

Imagine you're trying to predict house prices:
- **First attempt:** You guess the average price ($300K). You're off by ±$100K.
- **Second attempt:** You build a model to predict your errors from attempt 1. Now you're off by ±$50K.
- **Third attempt:** You build another model to predict the remaining errors. Now you're off by ±$25K.
- ...and so on.

Each model focuses on fixing the mistakes of the previous models. This is **gradient boosting**.

**Technical Definition:**
Gradient boosting builds an ensemble of weak learners (typically decision trees) sequentially, where each new tree corrects the errors of the previous ensemble.

---

## Why Gradient Boosting Dominates

### The Kaggle Winner

**Gradient boosting (XGBoost/LightGBM/CatBoost) wins ~80% of Kaggle competitions on tabular data.**

**Why?**
- ✅ Handles non-linear relationships
- ✅ Automatic feature interactions
- ✅ Built-in feature importance
- ✅ Handles missing values natively
- ✅ Built-in regularization
- ✅ Works with mixed data types
- ✅ Robust to outliers
- ✅ Excellent performance out-of-the-box

---

## How Gradient Boosting Works

### Core Algorithm

```
1. Start with initial prediction (usually mean for regression, log-odds for classification)
2. For iteration 1 to N:
   a. Calculate residuals (errors) for current ensemble
   b. Fit a new tree to predict these residuals
   c. Add this tree to the ensemble (with learning rate)
   d. Update predictions
3. Final prediction = initial + sum of all trees
```

### Mathematical Intuition

**Objective:** Minimize loss function L(y, F(x))

**Gradient Descent in Function Space:**
```
F_m(x) = F_{m-1}(x) + η × h_m(x)

where:
- F_m(x) = ensemble after m trees
- η = learning rate (shrinkage)
- h_m(x) = new tree fitted to negative gradient of loss
```

**Key Insight:** Each new tree is fitted to the **negative gradient** of the loss function, hence "gradient" boosting.

### Step-by-Step Example (Regression)

**Data:**
```
X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # True values
```

**Iteration 0:** Initial prediction = mean(y) = 6
```
Predictions: [6, 6, 6, 6, 6]
Residuals:   [-4, -2, 0, 2, 4]
```

**Iteration 1:** Fit tree to predict residuals
```
Tree 1 predicts: [-3, -1, 0, 1, 3]
Updated predictions: [6, 6, 6, 6, 6] + 0.1 × [-3, -1, 0, 1, 3]
                   = [5.7, 5.9, 6.0, 6.1, 6.3]
New residuals: [-3.7, -1.9, 0, 1.9, 3.7]
```

**Iteration 2:** Fit tree to new residuals
```
Tree 2 predicts: [-3.5, -1.8, 0, 1.8, 3.5]
Updated predictions: [5.7, 5.9, 6.0, 6.1, 6.3] + 0.1 × [-3.5, -1.8, 0, 1.8, 3.5]
                   = [5.35, 5.72, 6.0, 6.28, 6.65]
```

**...continue until residuals are small or max iterations reached**

---

## The Big Three: XGBoost vs LightGBM vs CatBoost

### Comparison Table

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | ⚡⚡ Medium | ⚡⚡⚡ Fastest | ⚡⚡ Medium |
| **Memory** | High | Low ✅ | Medium |
| **Accuracy** | Excellent | Excellent | Excellent ✅ |
| **Categorical Handling** | Manual encoding | Manual encoding | Native ✅✅✅ |
| **Default Hyperparameters** | Need tuning | Need tuning | Good out-of-box ✅ |
| **Overfitting Risk** | Medium | Higher | Lower ✅ |
| **GPU Support** | ✅ | ✅ | ✅ |
| **When to Use** | General purpose | Large datasets, speed critical | Many categorical features |

### XGBoost (Extreme Gradient Boosting)

**Innovations:**
- Regularized learning objective (L1 + L2)
- Sparsity-aware algorithm (handles missing values)
- Parallel tree construction
- Cache-aware access patterns

**Best For:**
- General-purpose gradient boosting
- When you need mature ecosystem and community
- Standard benchmarking baseline

**Pros:**
- ✅ Most widely used (lots of resources/tutorials)
- ✅ Robust regularization
- ✅ Handles missing values well
- ✅ Good parallelization

**Cons:**
- ❌ Slower than LightGBM on large datasets
- ❌ Higher memory usage
- ❌ Categorical features need manual encoding

**Key Hyperparameters:**
```python
import xgboost as xgb

params = {
    'objective': 'reg:squarederror',  # or 'binary:logistic', 'multi:softmax'
    'max_depth': 6,                    # Tree depth (higher = more complex)
    'learning_rate': 0.1,              # Shrinkage (lower = more conservative)
    'n_estimators': 100,               # Number of trees
    'subsample': 0.8,                  # Row sampling (like bagging)
    'colsample_bytree': 0.8,           # Column sampling per tree
    'reg_alpha': 0,                    # L1 regularization
    'reg_lambda': 1,                   # L2 regularization
    'min_child_weight': 1,             # Minimum sum of instance weight in a child
    'gamma': 0,                        # Minimum loss reduction to split
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)
```

### LightGBM (Light Gradient Boosting Machine)

**Key Innovation:** **Leaf-wise growth** instead of level-wise

```
Level-wise (XGBoost):        Leaf-wise (LightGBM):
      Split all nodes              Split best leaf only
      at each level                (higher gain)

        ○                              ○
       / \                            / \
      ○   ○                          ○   ○
     /|   |\                        / \
    ○ ○   ○ ○                      ○   ○
                                  / \
                                 ○   ○
```

**Additional Innovations:**
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Histogram-based algorithms

**Best For:**
- Large datasets (>10K rows)
- When training speed is critical
- Low memory environments

**Pros:**
- ✅ Fastest training speed
- ✅ Lowest memory usage
- ✅ Handles large datasets efficiently
- ✅ Good accuracy

**Cons:**
- ❌ More prone to overfitting (leaf-wise growth)
- ❌ Sensitive to hyperparameters
- ❌ Categorical features still need encoding

**Key Hyperparameters:**
```python
import lightgbm as lgb

params = {
    'objective': 'regression',         # or 'binary', 'multiclass'
    'metric': 'rmse',                  # Evaluation metric
    'boosting_type': 'gbdt',           # or 'dart', 'goss'
    'num_leaves': 31,                  # Max leaves (2^max_depth - 1)
    'learning_rate': 0.05,
    'n_estimators': 100,
    'max_depth': -1,                   # -1 = no limit (use num_leaves)
    'subsample': 0.8,                  # Bagging fraction
    'colsample_bytree': 0.8,           # Feature fraction
    'reg_alpha': 0,                    # L1
    'reg_lambda': 0,                   # L2
    'min_child_samples': 20,           # Min data in leaf
    'verbose': -1
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
```

### CatBoost (Categorical Boosting)

**Key Innovation:** **Ordered boosting** + native categorical feature handling

**Ordered Boosting:**
- Reduces prediction shift problem in gradient boosting
- Uses different permutations of data for each tree
- More robust, less overfitting

**Categorical Features:**
- Automatic handling (no encoding needed!)
- Target-based encoding with regularization
- Combination of categorical features

**Best For:**
- Datasets with many categorical features
- When you want good results with minimal tuning
- When you want to avoid overfitting

**Pros:**
- ✅ Best categorical feature handling ⭐⭐⭐
- ✅ Great default hyperparameters
- ✅ Most robust to overfitting
- ✅ Good accuracy out-of-box
- ✅ Symmetric trees (faster prediction)

**Cons:**
- ❌ Slower training than LightGBM
- ❌ Less community adoption (fewer resources)

**Key Hyperparameters:**
```python
from catboost import CatBoostRegressor

# Specify categorical features
cat_features = ['category', 'brand', 'location']

params = {
    'iterations': 1000,                # Number of trees
    'learning_rate': 0.03,
    'depth': 6,                        # Tree depth
    'l2_leaf_reg': 3,                  # L2 regularization
    'loss_function': 'RMSE',           # or 'Logloss', 'MultiClass'
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50,
    'cat_features': cat_features       # ⭐ Native categorical support
}

model = CatBoostRegressor(**params)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

---

## When to Use Gradient Boosting

### ✅ Use Gradient Boosting When:

1. **Tabular data** (structured data in rows/columns)
2. **Need high accuracy** (competition, high-stakes predictions)
3. **Have mixed data types** (numerical + categorical)
4. **Non-linear relationships** present
5. **Feature interactions** important
6. **Missing values** in data
7. **Have sufficient data** (>1K samples typically)

### ❌ Don't Use Gradient Boosting When:

1. **Need real-time predictions** (<10ms latency required) → Use linear models or shallow trees
2. **Need perfect interpretability** → Use logistic regression or single decision tree
3. **Extremely small dataset** (<100 samples) → Use simpler models with regularization
4. **Image/audio/video data** → Use deep learning (CNNs, transformers)
5. **Sequential data** with long-range dependencies → Use RNNs, LSTMs, transformers
6. **Limited compute** for training → Consider Random Forest (faster)

---

## Practical Implementation Guide

### Complete Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Further split train into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")
print(f"Test: {X_test.shape}")

# ============================================
# 1. XGBoost
# ============================================
print("\n=== XGBoost ===")

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluate
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"RMSE: {xgb_rmse:.4f}")
print(f"R²: {xgb_r2:.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")

# ============================================
# 2. LightGBM
# ============================================
print("\n=== LightGBM ===")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42,
    'verbose': -1
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

lgb_pred = lgb_model.predict(X_test)

lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
lgb_r2 = r2_score(y_test, lgb_pred)

print(f"RMSE: {lgb_rmse:.4f}")
print(f"R²: {lgb_r2:.4f}")
print(f"Best iteration: {lgb_model.best_iteration_}")

# ============================================
# 3. CatBoost
# ============================================
print("\n=== CatBoost ===")

# Identify categorical features
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50
}

cat_model = CatBoostRegressor(**cat_params)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features
)

cat_pred = cat_model.predict(X_test)

cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
cat_r2 = r2_score(y_test, cat_pred)

print(f"RMSE: {cat_rmse:.4f}")
print(f"R²: {cat_r2:.4f}")
print(f"Best iteration: {cat_model.best_iteration_}")

# ============================================
# Compare Models
# ============================================
print("\n=== Model Comparison ===")
comparison = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
    'RMSE': [xgb_rmse, lgb_rmse, cat_rmse],
    'R²': [xgb_r2, lgb_r2, cat_r2]
})
print(comparison)

# ============================================
# Feature Importance
# ============================================
def plot_feature_importance(model, model_name, top_n=20):
    if model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'LightGBM':
        importance = model.feature_importances_
    elif model_name == 'CatBoost':
        importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title(f'{model_name} - Top {top_n} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_importance.png')
    plt.close()

# Plot for best model
plot_feature_importance(xgb_model, 'XGBoost')
```

### Hyperparameter Tuning with Optuna

```python
import optuna
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    """Optuna objective function for LightGBM hyperparameter tuning"""

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,

        # Hyperparameters to tune
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # Train with early stopping
    model = lgb.LGBMRegressor(**params, n_estimators=1000)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0),
            LightGBMPruningCallback(trial, 'rmse')
        ]
    )

    # Predict and evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)

print("Best trial:")
print(f"  RMSE: {study.best_trial.value:.4f}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best params
best_params = study.best_trial.params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_estimators': 1000
})

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting

**Symptoms:**
- Training error much lower than validation error
- Very deep trees (max_depth > 10)
- Too many iterations

**Solutions:**
```python
# 1. Reduce model complexity
params = {
    'max_depth': 3,              # Shallower trees (was 6-10)
    'num_leaves': 15,            # Fewer leaves (LightGBM)
    'min_child_samples': 50,     # More data per leaf (was 20)
}

# 2. Add regularization
params = {
    'reg_alpha': 1.0,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
}

# 3. Use sampling
params = {
    'subsample': 0.7,            # Row sampling
    'colsample_bytree': 0.7,     # Column sampling
}

# 4. Lower learning rate + more trees
params = {
    'learning_rate': 0.01,       # Slower learning (was 0.1)
    'n_estimators': 5000,        # More trees to compensate
    'early_stopping_rounds': 100 # Stop if no improvement
}
```

### Pitfall 2: Slow Training

**Solutions:**
```python
# 1. Use LightGBM (fastest)
import lightgbm as lgb

# 2. Reduce tree complexity
params = {
    'max_depth': 5,              # Shallower (was 8-10)
    'num_leaves': 31,            # Fewer leaves (was 63-127)
}

# 3. Use histogram-based methods
params = {
    'max_bin': 255,              # Fewer bins (was 255 default)
}

# 4. Use GPU
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

# 5. Reduce data
X_sample = X_train.sample(frac=0.5, random_state=42)  # Use 50% for tuning
```

### Pitfall 3: Poor Performance Despite Tuning

**Diagnostic Questions:**
1. **Is the problem learnable?**
   - Check if target correlates with features
   - Try simpler model (logistic regression) as sanity check

2. **Data quality issues?**
   - Check for data leakage
   - Verify train/test split is appropriate
   - Look for outliers, missing values

3. **Feature engineering needed?**
   - Gradient boosting is good but not magic
   - Create domain-specific features
   - Try polynomial features, interactions

4. **Wrong metric?**
   - Optimizing RMSE but should optimize MAE?
   - Classification with imbalanced classes?

**Solutions:**
```python
# 1. Feature engineering
X['feature_interaction'] = X['feature1'] * X['feature2']
X['feature_ratio'] = X['feature1'] / (X['feature2'] + 1e-5)
X['feature_squared'] = X['feature1'] ** 2

# 2. Target transformation (for regression)
y_train_log = np.log1p(y_train)  # Log transform target
# Train model on y_train_log, then inverse: np.expm1(predictions)

# 3. Custom objective function (advanced)
def custom_objective(y_true, y_pred):
    # Define custom loss
    grad = # ... gradient
    hess = # ... hessian
    return grad, hess

# 4. Ensemble multiple models
final_pred = 0.5 * xgb_pred + 0.3 * lgb_pred + 0.2 * cat_pred
```

### Pitfall 4: Memory Issues

**Solutions:**
```python
# 1. Use LightGBM (lowest memory)
import lightgbm as lgb

# 2. Reduce tree size
params = {
    'max_depth': 5,              # Shallower trees
    'max_bin': 128,              # Fewer bins (was 255)
}

# 3. Use smaller data types
X = X.astype('float32')          # Instead of float64
# For categorical: use 'category' dtype

# 4. Process in chunks (for very large data)
import dask.dataframe as dd
ddf = dd.read_csv('large_file.csv')

# 5. Use histogram-based methods
params = {
    'tree_method': 'hist',       # XGBoost histogram method
}
```

---

## Interview Questions

### Q1: "Explain gradient boosting to a non-technical person"

**Answer:**
"Imagine you're taking a test and keep getting questions wrong. After each test:
1. You identify which questions you got wrong
2. You study those specific topics
3. You take another test, focusing on fixing those mistakes

Gradient boosting works similarly: it builds a series of simple models, where each new model focuses on correcting the errors of the previous models. By combining all these models, we get a final prediction that's much more accurate than any single model."

### Q2: "What's the difference between Random Forest and Gradient Boosting?"

**Answer:**
```
Random Forest (Bagging):
- Builds trees in parallel (independent)
- Each tree sees random subset of data and features
- Trees are deep (high variance, low bias)
- Final prediction = average of all trees
- Reduces variance through averaging
- Less prone to overfitting
- Faster to train (parallel)

Gradient Boosting:
- Builds trees sequentially (dependent)
- Each tree corrects previous tree's errors
- Trees are shallow (low variance, high bias)
- Final prediction = weighted sum of all trees
- Reduces bias through sequential correction
- Can overfit if not regularized
- Slower to train (sequential)

Use Random Forest for: Quick baseline, less tuning needed
Use Gradient Boosting for: Maximum accuracy, willing to tune
```

### Q3: "XGBoost vs LightGBM vs CatBoost - which to choose?"

**Answer:**
"It depends on the problem:

**LightGBM:**
- Best for: Large datasets (>100K rows), speed is critical
- Trade-off: Slightly more prone to overfitting

**CatBoost:**
- Best for: Many categorical features, want good defaults
- Trade-off: Slower training than LightGBM

**XGBoost:**
- Best for: General purpose, when you want most mature ecosystem
- Trade-off: Slower and more memory than LightGBM

**In practice:** I'd try all three and choose based on validation performance. LightGBM often wins on speed/performance, CatBoost is great for categorical-heavy data, XGBoost is the safe default."

### Q4: "How do you prevent overfitting in gradient boosting?"

**Answer:**
"Multiple strategies:

1. **Reduce model complexity:**
   - Shallower trees (max_depth = 3-6)
   - Fewer leaves (num_leaves = 15-31)
   - More data per leaf (min_child_samples = 50+)

2. **Regularization:**
   - L1/L2 penalties (reg_alpha, reg_lambda)
   - Learning rate (0.01-0.1, lower is more conservative)

3. **Sampling:**
   - Row sampling (subsample = 0.7-0.9)
   - Column sampling (colsample_bytree = 0.7-0.9)

4. **Early stopping:**
   - Monitor validation error
   - Stop if no improvement for N rounds

5. **Cross-validation:**
   - Use k-fold CV to ensure generalization

I'd start with early stopping (easiest), then adjust learning rate and tree depth, finally add regularization if needed."

### Q5: "Can gradient boosting handle categorical features?"

**Answer:**
"Depends on the implementation:

**CatBoost:**
- ✅ Native support (best option)
- Handles automatically with target-based encoding + regularization
- Just pass `cat_features` parameter

**XGBoost/LightGBM:**
- ❌ No native support (must encode manually)
- Options:
  - One-hot encoding (for low cardinality <10 categories)
  - Label encoding (preserves ordinal if exists)
  - Target encoding (careful of leakage!)
  - Frequency encoding
  - Feature hashing (for high cardinality)

**Recommendation:**
If many categorical features, use CatBoost. Otherwise, manually encode and use XGBoost/LightGBM."

### Q6: "How do you interpret gradient boosting models?"

**Answer:**
"Gradient boosting is less interpretable than logistic regression or single decision trees, but we have tools:

**1. Feature Importance:**
```python
importance = model.feature_importances_
```
Shows which features contribute most (gain-based, split-based, etc.)

**2. SHAP (SHapley Additive exPlanations):**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```
- Shows feature contributions for each prediction
- Local (instance-level) and global interpretations
- Better than feature importance for understanding

**3. Partial Dependence Plots:**
Shows relationship between feature and prediction

**4. LIME (Local Interpretable Model-agnostic Explanations):**
Approximates model locally with interpretable model

For high-stakes decisions (healthcare, legal), I'd use SHAP for detailed explanations."

---

## Key Takeaways

1. **Gradient boosting builds sequentially** - each tree corrects previous errors
2. **LightGBM for speed** (large datasets), **CatBoost for categoricals**, **XGBoost for general use**
3. **Overfitting is common** - use early stopping, regularization, shallow trees
4. **Feature engineering still matters** - GB is powerful but not magic
5. **Hyperparameter tuning important** - use cross-validation and automated tuning (Optuna)
6. **Best for tabular data** - not for images, text, audio
7. **Interpretation tools exist** - SHAP, feature importance, partial dependence
8. **Monitor validation performance** - don't trust training error alone

---

**Next:** [Random Forest](./random_forest.md) | **Back:** [README](./README.md)
