# Evaluation Metrics Taxonomy

## Why This Matters

**Choosing the right metric is one of the most common interview questions.** Interviewers want to see that you:
- Understand the business problem
- Know the tradeoffs between different metrics
- Can justify your choices
- Recognize edge cases and limitations

The wrong metric can lead to models that optimize the wrong thing, even if technically "accurate."

## The Golden Rule

> **Your metric should align with your business objective.**

A model with 99% accuracy might be worthless if it never predicts the rare but critical events you care about.

---

## Classification Metrics

### 1. Confusion Matrix (Foundation)

```
                    Predicted
                 Positive  Negative
Actual Positive     TP        FN
Actual Negative     FP        TN
```

**Definitions:**
- **True Positive (TP)**: Correctly predicted positive
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **True Negative (TN)**: Correctly predicted negative
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

### 2. Accuracy

**Formula:** `(TP + TN) / (TP + TN + FP + FN)`

**Intuition:** "What percentage of predictions are correct?"

**When to use:**
- Balanced datasets
- Equal cost for FP and FN errors
- Multi-class problems where all classes matter equally

**When NOT to use:**
- Imbalanced datasets (e.g., fraud detection with 0.1% fraud rate)
- When error types have different costs

**Example:**
```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)  # 0.833
```

**Common Interview Trap:**
"We achieved 99% accuracy on fraud detection!"
- If only 1% of transactions are fraud, predicting "no fraud" for everything gives 99% accuracy
- This model is useless

### 3. Precision

**Formula:** `TP / (TP + FP)`

**Intuition:** "Of all positive predictions, how many were actually positive?"
- Also called: Positive Predictive Value (PPV)

**When to use:**
- When **false positives are costly**
- Example: Email spam detection (don't want to mark important emails as spam)
- Example: Medical diagnosis where false positives cause expensive follow-up tests

**Real-world scenario (Logistics):**
*Predicting which packages will be damaged in transit:*
- High precision means when you flag a package for special handling, it really needs it
- You don't want to waste resources on unnecessary special handling (false positives)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
```

### 4. Recall (Sensitivity, True Positive Rate)

**Formula:** `TP / (TP + FN)`

**Intuition:** "Of all actual positives, how many did we correctly identify?"

**When to use:**
- When **false negatives are costly**
- Example: Cancer screening (can't afford to miss cancer cases)
- Example: Fraud detection (better to investigate non-fraud than miss fraud)

**Real-world scenario (Supply Chain):**
*Predicting stock-outs:*
- High recall means you catch most potential stock-outs
- Missing a stock-out (false negative) causes lost sales and customer dissatisfaction

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
```

**The Precision-Recall Tradeoff:**
- Increasing threshold → Higher precision, lower recall
- Decreasing threshold → Higher recall, lower precision
- You can't maximize both simultaneously

### 5. F1-Score

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`
- Harmonic mean of precision and recall

**Intuition:** "Balanced measure of precision and recall"

**When to use:**
- When you need a single metric combining precision and recall
- When FP and FN costs are roughly equal
- For imbalanced datasets (better than accuracy)

**When NOT to use:**
- When precision and recall have very different importance
- Use F-beta score instead if you want to weight one higher

**F-Beta Score:**
- `F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)`
- β > 1: Favors recall
- β < 1: Favors precision
- β = 2 (F2): Common choice when recall is more important

```python
from sklearn.metrics import f1_score, fbeta_score

f1 = f1_score(y_true, y_pred)
f2 = fbeta_score(y_true, y_pred, beta=2)  # Weighs recall higher
```

### 6. ROC Curve & AUC

**ROC (Receiver Operating Characteristic) Curve:**
- Plots True Positive Rate (Recall) vs False Positive Rate
- FPR = FP / (FP + TN) = "What fraction of negatives were incorrectly classified as positive?"

**AUC (Area Under Curve):**
- Single number summarizing ROC curve
- Range: 0.5 (random) to 1.0 (perfect)
- **AUC = 0.5**: Random guessing
- **AUC > 0.7**: Acceptable
- **AUC > 0.8**: Good
- **AUC > 0.9**: Excellent

**Intuition:** "Probability that the model ranks a random positive example higher than a random negative example"

**When to use:**
- Comparing models across different thresholds
- When you care about ranking quality
- Binary classification with balanced importance of classes

**When NOT to use:**
- Severely imbalanced datasets (use PR-AUC instead)
- When you have a fixed operating point (use metrics at that threshold)

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_true, y_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

### 7. Precision-Recall Curve & PR-AUC

**PR Curve:** Plots Precision vs Recall at different thresholds

**When to use PR-AUC instead of ROC-AUC:**
- **Highly imbalanced datasets** (fraud, disease, defects)
- When you care more about positive class performance

**Why PR-AUC is better for imbalanced data:**
- ROC-AUC can be misleadingly optimistic
- Large number of true negatives makes FPR look good
- PR-AUC focuses on positive class performance

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)

plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

### 8. Log Loss (Binary Cross-Entropy)

**Formula:** `-1/N × Σ[y log(p) + (1-y) log(1-p)]`

**Intuition:** "Penalizes confident wrong predictions heavily"

**When to use:**
- When you need calibrated probabilities (not just classifications)
- When model confidence matters
- For probabilistic predictions

**When NOT to use:**
- When you only care about final classification
- When interpretability is key (not intuitive for stakeholders)

```python
from sklearn.metrics import log_loss

y_proba = model.predict_proba(X_test)
logloss = log_loss(y_true, y_proba)
```

### 9. Matthews Correlation Coefficient (MCC)

**Formula:** `(TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`

**Range:** -1 (complete disagreement) to +1 (perfect prediction)

**Intuition:** "Correlation between predictions and ground truth"

**Advantage:**
- Works well with imbalanced datasets
- Considers all four confusion matrix values
- More informative than F1 for imbalanced data

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
```

### 10. Cohen's Kappa

**Measures agreement while accounting for chance:**
- κ = (Observed accuracy - Expected accuracy) / (1 - Expected accuracy)

**Interpretation:**
- < 0: No agreement (worse than random)
- 0-0.20: Slight agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

**When to use:**
- Multi-rater scenarios
- Imbalanced datasets
- When you want to account for random agreement

---

## Regression Metrics

### 1. Mean Absolute Error (MAE)

**Formula:** `1/N × Σ|y_true - y_pred|`

**Intuition:** "Average absolute difference between predictions and actuals"

**Advantages:**
- Intuitive and interpretable (same units as target)
- Robust to outliers (compared to MSE)
- All errors weighted equally

**When to use:**
- When outliers shouldn't dominate the metric
- When you want interpretability
- When all errors have equal business impact

**Real-world scenario (Logistics):**
*Predicting delivery time:*
- MAE = 15 minutes means predictions are off by 15 minutes on average
- Easy to communicate to business stakeholders

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

### 2. Mean Squared Error (MSE)

**Formula:** `1/N × Σ(y_true - y_pred)²`

**Intuition:** "Average squared difference" - penalizes large errors heavily

**Advantages:**
- Smooth gradient (good for optimization)
- Heavily penalizes large errors

**Disadvantages:**
- Not in same units as target (use RMSE for interpretability)
- Very sensitive to outliers

**When to use:**
- When large errors are disproportionately bad
- During model training (smooth gradients)
- When outliers are genuine and important

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

### 3. Root Mean Squared Error (RMSE)

**Formula:** `sqrt(MSE)`

**Intuition:** "Square root of MSE" - back to original units

**When to use:**
- Same as MSE, but when you need interpretability
- More common for reporting than MSE

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### 4. Mean Absolute Percentage Error (MAPE)

**Formula:** `100/N × Σ|y_true - y_pred| / |y_true|`

**Intuition:** "Average percentage error"

**Advantages:**
- Scale-independent (compare across different scales)
- Interpretable as percentage

**Disadvantages:**
- **Undefined for y_true = 0**
- **Biased toward underestimates** (penalizes overestimates more)
- Not symmetric

**When to use:**
- Comparing models across different scales
- Business reporting (percentages are intuitive)

**When NOT to use:**
- Data with zeros or near-zeros
- When overestimates and underestimates should be treated equally

**Real-world scenario (Forecasting):**
*Demand forecasting:*
- MAPE = 10% means predictions are off by 10% on average
- Easy for business to understand: "We're within ±10%"

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### 5. Symmetric Mean Absolute Percentage Error (SMAPE)

**Formula:** `100/N × Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)`

**Advantages over MAPE:**
- Symmetric (treats over/under-prediction equally)
- Bounded (0% to 200%)

**Still has issues:**
- Undefined when both y_true and y_pred are 0
- Complex interpretation

```python
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
```

### 6. R² (Coefficient of Determination)

**Formula:** `1 - (SS_res / SS_tot)`
- SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
- SS_tot = Σ(y_true - y_mean)²  (total sum of squares)

**Intuition:** "Proportion of variance explained by the model"

**Interpretation:**
- **R² = 1**: Perfect predictions
- **R² = 0**: Model is no better than predicting the mean
- **R² < 0**: Model is worse than predicting the mean

**Advantages:**
- Scale-free
- Measures goodness of fit

**Disadvantages:**
- Always increases with more features (use adjusted R²)
- Can be negative on test set
- Sensitive to outliers

**When to use:**
- Comparing models on the same dataset
- Understanding model fit quality

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

### 7. Adjusted R²

**Formula:** `1 - (1 - R²) × (n - 1) / (n - p - 1)`
- n = number of samples
- p = number of features

**Advantage:** Penalizes unnecessary features

### 8. Huber Loss

**Combines MAE and MSE:**
- Behaves like MSE for small errors
- Behaves like MAE for large errors (robust to outliers)

**When to use:**
- When you have outliers but still want smooth gradients
- Robust regression

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()
```

---

## Ranking Metrics

### 1. Mean Average Precision (MAP)

**Used for:** Information retrieval, recommendation systems

**Intuition:** "How good is the ranking of relevant items?"

**When to use:**
- Search engines
- Recommendation systems
- When you care about ordering of top results

### 2. Normalized Discounted Cumulative Gain (NDCG)

**Accounts for:**
- Position of relevant items
- Graded relevance (not just binary)

**When to use:**
- Search ranking
- Recommendations with relevance scores
- When position matters (top items more important)

---

## Multi-Class Metrics

### 1. Macro-Average

**Compute metric for each class, then average**

**When to use:**
- All classes equally important
- Small classes should have equal weight

**Example:**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Macro-average
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')
```

### 2. Micro-Average

**Aggregate contributions of all classes, then compute metric**

**When to use:**
- Classes have very different sizes
- Want to weight by class frequency

```python
# Micro-average
precision_micro = precision_score(y_true, y_pred, average='micro')
recall_micro = recall_score(y_true, y_pred, average='micro')
f1_micro = f1_score(y_true, y_pred, average='micro')
```

### 3. Weighted-Average

**Weighted by class support (number of samples)**

**When to use:**
- Default compromise between macro and micro
- When larger classes are more important

```python
# Weighted-average
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

---

## Time Series Metrics

### Specific Considerations

**Traditional metrics (MAE, RMSE, MAPE) still apply, but also consider:**

1. **Directional Accuracy**: Did we predict the right trend direction?
2. **Forecast Bias**: Do we systematically over or underpredict?
3. **Seasonal Accuracy**: How well do we capture seasonality?

### Additional Metrics

**Mean Absolute Scaled Error (MASE):**
- Scales error relative to naive forecast
- Works well for seasonal data
- Independent of data scale

**Weighted MAPE (WMAPE):**
- Better than MAPE for data with zeros
- `WMAPE = Σ|y_true - y_pred| / Σ|y_true|`

---

## Clustering Metrics

### 1. Silhouette Score

**Range:** -1 to +1

**Interpretation:**
- +1: Perfect clustering
- 0: Overlapping clusters
- -1: Wrong cluster assignments

**When to use:**
- Evaluating k-means or hierarchical clustering
- Determining optimal number of clusters

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, cluster_labels)
```

### 2. Davies-Bouldin Index

**Lower is better**

**Measures:** Ratio of within-cluster to between-cluster distances

### 3. Calinski-Harabasz Index (Variance Ratio Criterion)

**Higher is better**

**Measures:** Ratio of between-cluster variance to within-cluster variance

---

## Business-Specific Metrics

### Cost-Sensitive Metrics

**Real-world costs are often asymmetric:**

```python
def business_cost(y_true, y_pred, cost_fp=10, cost_fn=100):
    """
    Custom cost function for business problems

    Example: Fraud detection
    - cost_fp=10: Cost of investigating false positive
    - cost_fn=100: Cost of missing fraud
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost
```

### Expected Value Framework

**For decision-making problems:**

```python
def expected_value(y_true, y_proba, value_tp=1000, cost_fp=50):
    """
    Example: Marketing campaign
    - value_tp=1000: Revenue from converting a customer
    - cost_fp=50: Cost of contacting non-responder
    """
    threshold = 0.5  # Can be optimized
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    expected_value = (tp * value_tp) - (fp * cost_fp)
    return expected_value
```

---

## Metric Selection Decision Tree

```
START
│
├─ What type of problem?
│  │
│  ├─ BINARY CLASSIFICATION
│  │  │
│  │  ├─ Balanced classes?
│  │  │  ├─ Yes → Accuracy, F1, ROC-AUC
│  │  │  └─ No → (Next question)
│  │  │
│  │  ├─ Which error is worse?
│  │  │  ├─ FP worse → Optimize Precision
│  │  │  ├─ FN worse → Optimize Recall
│  │  │  └─ Both bad → F1 or PR-AUC
│  │  │
│  │  ├─ Need probability calibration?
│  │  │  └─ Yes → Log Loss, Brier Score
│  │  │
│  │  └─ Severe imbalance (>1:100)?
│  │     └─ Yes → PR-AUC, MCC
│  │
│  ├─ MULTI-CLASS
│  │  │
│  │  ├─ All classes equal importance?
│  │  │  ├─ Yes → Macro-averaged F1
│  │  │  └─ No → Weighted F1, Micro F1
│  │  │
│  │  └─ Need per-class analysis?
│  │     └─ Yes → Confusion matrix, per-class F1
│  │
│  ├─ REGRESSION
│  │  │
│  │  ├─ Outliers present?
│  │  │  ├─ Yes → MAE, Huber Loss
│  │  │  └─ No → RMSE
│  │  │
│  │  ├─ Need scale-independence?
│  │  │  └─ Yes → MAPE (if no zeros), R²
│  │  │
│  │  └─ Large errors particularly bad?
│  │     └─ Yes → RMSE, Huber Loss
│  │
│  ├─ RANKING
│  │  └─ Use MAP, NDCG
│  │
│  └─ CLUSTERING
│     └─ Use Silhouette, Davies-Bouldin
│
END
```

---

## Common Interview Questions

### Q1: "Why not use accuracy for fraud detection?"

**Answer:**
"In fraud detection, fraud cases are typically <1% of transactions. If I predict 'no fraud' for every transaction, I get >99% accuracy, but the model is useless because it never catches fraud. This is why we need metrics that focus on the positive class performance like precision, recall, and PR-AUC. Specifically, we'd optimize for high recall (catching most fraud) while maintaining acceptable precision (not investigating too many false alarms)."

### Q2: "Should we use ROC-AUC or PR-AUC?"

**Answer:**
"It depends on class balance:
- **Balanced data (1:10 or better)**: ROC-AUC is fine
- **Imbalanced data (1:100 or worse)**: PR-AUC is better

ROC-AUC can be misleadingly optimistic with imbalanced data because the large number of true negatives makes the false positive rate look good. PR-AUC focuses on positive class performance, which is what we actually care about in imbalanced scenarios."

### Q3: "Why is MAPE problematic?"

**Answer:**
"MAPE has three key issues:
1. **Undefined for zeros**: Dividing by y_true causes problems
2. **Asymmetric**: Penalizes overestimates more than underestimates (e.g., 50% over vs 50% under are not equivalent)
3. **Scale sensitivity**: Better for values far from zero

Alternatives: SMAPE (more symmetric), MAE (absolute), or WMAPE (handles zeros better)."

### Q4: "What metric for imbalanced multi-class?"

**Answer:**
"Use **macro-averaged F1** if all classes are equally important (gives equal weight to rare classes), or **weighted F1** if larger classes are more important. Also examine the confusion matrix to understand per-class performance. For critical rare classes, I'd also track per-class recall explicitly."

### Q5: "How do you choose the classification threshold?"

**Answer:**
"The threshold depends on the business problem:
1. **Calculate business cost** for FP and FN
2. **Plot precision-recall or cost curve** at different thresholds
3. **Choose threshold** that optimizes business metric

Example (fraud):
- If investigating false positive costs $10 and missing fraud costs $1000, we'd set a low threshold (high recall, accept more FPs).
- If investigation capacity is limited, we'd increase threshold (higher precision)."

---

## Practical Sklearn Example

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import numpy as np

def evaluate_classification_model(y_true, y_pred, y_proba=None):
    """Comprehensive classification evaluation"""

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n=== Metrics Summary ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='binary'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='binary'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='binary'):.4f}")

    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
        print(f"PR-AUC:    {average_precision_score(y_true, y_proba):.4f}")

def evaluate_regression_model(y_true, y_pred):
    """Comprehensive regression evaluation"""
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score
    )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Custom MAPE (handle zeros)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print("=== Regression Metrics ===")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"MAPE:  {mape:.2f}%")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
```

---

## Key Takeaways

1. **No universal "best" metric** - choice depends on business problem
2. **Imbalanced data** requires special attention (F1, PR-AUC, MCC)
3. **Always use multiple metrics** for complete picture
4. **Align metric with business objective** - technical metric should map to business goal
5. **Consider error costs** - FP and FN often have different business impacts
6. **Beware of pitfalls** - accuracy paradox, MAPE with zeros, etc.

---

**Next:** [Data Leakage](./data_leakage.md) | **Back:** [README](./README.md)
