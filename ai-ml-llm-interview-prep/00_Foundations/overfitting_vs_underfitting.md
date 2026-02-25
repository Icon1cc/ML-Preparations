# Overfitting vs Underfitting

## The Core Problem (Plain English)

Imagine you're studying for a test:

**Underfitting (Understudy):**
- You barely studied
- You don't understand the material
- You fail both practice tests AND the real test
- **Problem: Too simple understanding**

**Good Fit:**
- You studied the right amount
- You understand concepts (not just memorization)
- You do well on practice AND real test
- **Sweet spot!**

**Overfitting (Overstudy/Memorize):**
- You memorized every practice question exactly
- You ace practice tests
- But real test has slightly different questions → you fail
- **Problem: Memorized specifics, didn't learn general concepts**

---

## Visual Understanding

```
Underfitting               Good Fit                Overfitting
────────────────────────────────────────────────────────────────
Data: ●  ●   ●  ●   ●      Data: ●  ●   ●  ●   ●   Data: ●  ●   ●  ●   ●
Model: ─────────────       Model: ╱‾‾╲___╱‾╲      Model: ╱╲╱╲╱╲╱╲╱╲
       (line)                     (curve)                (wiggly)

Miss pattern              Capture pattern         Memorize noise
High bias                 Balanced                High variance
Low complexity            Right complexity        Too complex
```

---

## Formal Definitions

### Underfitting
**Model is too simple to capture the underlying pattern**

**Characteristics:**
- Poor performance on training data
- Poor performance on test data
- High bias, low variance
- Model assumptions too restrictive

**Example:**
```python
# Linear model for non-linear data
# y = x² (true relationship)
model = LinearRegression()  # Tries to fit y = a + b×x
# Will underfit because it can't capture the curve
```

### Overfitting
**Model captures noise in training data as if it were real pattern**

**Characteristics:**
- Excellent performance on training data
- Poor performance on test data
- Low bias, high variance
- Model too complex for amount of data

**Example:**
```python
# Polynomial degree 20 for simple data
model = PolynomialFeatures(degree=20)
# Will overfit because it memorizes training points
```

### Good Fit
**Model captures true underlying pattern without noise**

**Characteristics:**
- Good performance on training data
- Similar performance on test data
- Balanced bias-variance
- Generalizes well to new data

---

## Diagnosis

### How to Detect Underfitting

**Symptoms:**
```
Training accuracy: 65%
Validation accuracy: 64%
Test accuracy: 63%

All poor → Underfitting
```

**Learning curve shows:**
- Training and validation error both high
- Error plateaus early
- Gap between errors is small

```
Error ↑
    │
    │  ─ ─ ─ ─ ─ ─   Validation (high)
    │
    │  ───────────   Training (also high)
    │
    └──────────────→ Training examples

Both high, close together = Underfitting
```

**Checklist:**
- [ ] Training error > threshold
- [ ] Validation error ≈ training error
- [ ] Simple model (e.g., linear for non-linear data)
- [ ] Few features
- [ ] Low model capacity

### How to Detect Overfitting

**Symptoms:**
```
Training accuracy: 99%
Validation accuracy: 72%
Test accuracy: 70%

Large gap → Overfitting
```

**Learning curve shows:**
- Training error very low
- Validation error much higher
- Large gap between errors

```
Error ↑
    │
    │  ─ ─ ─ ─ ─ ─   Validation (high)
    │
    │
    │  ───────────   Training (very low)
    │
    └──────────────→ Training examples

Large gap = Overfitting
```

**Checklist:**
- [ ] Training error << validation error
- [ ] Gap > 15-20%
- [ ] Complex model (many parameters)
- [ ] Small dataset
- [ ] Training too long

### How to Detect Good Fit

**Symptoms:**
```
Training accuracy: 85%
Validation accuracy: 83%
Test accuracy: 82%

All good, small gap → Good fit
```

**Learning curve shows:**
- Training error reasonably low
- Validation error close to training error
- Both converge to acceptable level

```
Error ↑
    │
    │          ─ ─ ─ ─ ─   Validation (low)
    │
    │         ───────────   Training (low)
    │
    └──────────────────────→ Training examples

Both low, close together = Good fit
```

---

## Solutions

### Fixing Underfitting

| Solution | How it helps | Example |
|----------|--------------|---------|
| **Increase model complexity** | Capture more patterns | Linear → Polynomial, shallow → deep |
| **Add more features** | Give model more information | Feature engineering, interactions |
| **Reduce regularization** | Allow model more flexibility | Lower λ in Ridge/Lasso |
| **Train longer** | Let model learn more | More epochs, lower learning rate |
| **Use more powerful model** | Better capacity | Decision tree → Random Forest → XGBoost |
| **Remove feature constraints** | More freedom | Remove feature selection, use all features |

**Code Examples:**

```python
# 1. Increase polynomial degree
from sklearn.preprocessing import PolynomialFeatures

# Before (underfitting)
model = LinearRegression()  # degree=1
model.fit(X, y)

# After
poly = PolynomialFeatures(degree=3)  # Add squared, cubic terms
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# 2. Use more complex model
# Before
model = LogisticRegression()

# After
model = RandomForestClassifier(n_estimators=100, max_depth=10)
# Or
model = xgb.XGBClassifier()

# 3. Reduce regularization
# Before
model = Ridge(alpha=10.0)  # High regularization

# After
model = Ridge(alpha=0.1)  # Lower regularization

# 4. Add features
# Create interaction features
X['feature1_x_feature2'] = X['feature1'] * X['feature2']
X['feature1_squared'] = X['feature1'] ** 2

# 5. Train longer (neural networks)
# Before
model.fit(X, y, epochs=10)

# After
model.fit(X, y, epochs=100)
```

### Fixing Overfitting

| Solution | How it helps | Example |
|----------|--------------|---------|
| **Get more data** | Harder to memorize | Collect more samples, data augmentation |
| **Reduce model complexity** | Fewer parameters to overfit | Simpler architecture, fewer features |
| **Add regularization** | Penalize complexity | L1, L2, Dropout, Early stopping |
| **Cross-validation** | Better performance estimate | K-fold CV |
| **Feature selection** | Remove irrelevant features | Correlation analysis, feature importance |
| **Ensemble methods** | Average out variance | Bagging (Random Forest) |
| **Data augmentation** | Create more diverse examples | Rotation, flip, noise for images |

**Code Examples:**

```python
# 1. Add L2 regularization (Ridge)
# Before (no regularization)
model = LinearRegression()

# After
model = Ridge(alpha=1.0)  # Add L2 penalty

# 2. Add L1 regularization (Lasso) - also does feature selection
model = Lasso(alpha=1.0)

# 3. Reduce polynomial degree
# Before
poly = PolynomialFeatures(degree=15)  # Too complex

# After
poly = PolynomialFeatures(degree=2)

# 4. Reduce tree depth
# Before
model = DecisionTreeClassifier(max_depth=None)  # No limit

# After
model = DecisionTreeClassifier(max_depth=5)

# 5. Early stopping (neural networks)
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop if no improvement for 10 epochs
    restore_best_weights=True
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=1000,
          callbacks=[early_stop])

# 6. Dropout (neural networks)
from keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),  # Randomly drop 50% of neurons during training
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# 7. Data augmentation (images)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 8. Get more data
# If not possible, try synthetic data or SMOTE for tabular data
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 9. Feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)  # Keep top 20 features
X_selected = selector.fit_transform(X_train, y_train)

# 10. Cross-validation for better estimate
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Complete Diagnostic Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def diagnose_model(model, X, y):
    """
    Diagnose if model is underfitting, overfitting, or good fit
    """

    # 1. Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train and evaluate
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    print(f"Gap: {train_score - test_score:.3f}")

    # 3. Diagnose
    if train_score < 0.7 and abs(train_score - test_score) < 0.1:
        diagnosis = "UNDERFITTING"
        recommendation = [
            "- Use more complex model",
            "- Add more features",
            "- Reduce regularization",
            "- Train longer"
        ]
    elif train_score > 0.9 and (train_score - test_score) > 0.15:
        diagnosis = "OVERFITTING"
        recommendation = [
            "- Get more training data",
            "- Add regularization (L1/L2/Dropout)",
            "- Reduce model complexity",
            "- Use cross-validation",
            "- Early stopping"
        ]
    else:
        diagnosis = "GOOD FIT"
        recommendation = ["- Model looks good!",
                        "- Maybe try slight tuning for improvement"]

    print(f"\nDiagnosis: {diagnosis}")
    print("\nRecommendations:")
    for rec in recommendation:
        print(rec)

    # 4. Plot learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2' if hasattr(model, 'predict') else 'accuracy'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.2)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {diagnosis}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return diagnosis

# Example usage
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=10)
model = LinearRegression()
diagnose_model(model, X, y)
```

---

## Real-World Example

### Problem: Predicting House Prices

**Dataset:** 100 houses with features [size, bedrooms, age]

#### Attempt 1: Underfitting

```python
# Model: Always predict mean price
mean_price = np.mean(train_prices)
predictions = np.full(len(test_prices), mean_price)

# Result:
# Train RMSE: $50K
# Test RMSE: $51K
# Diagnosis: Underfitting (both errors high)
```

#### Attempt 2: Good Fit

```python
# Model: Linear regression with all features
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Result:
# Train RMSE: $15K
# Test RMSE: $18K
# Diagnosis: Good fit (both errors acceptable, small gap)
```

#### Attempt 3: Overfitting

```python
# Model: Polynomial degree 15 with 100 samples
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_poly, y_train)

# Result:
# Train RMSE: $1K  (too good!)
# Test RMSE: $60K (terrible!)
# Diagnosis: Overfitting (memorized training data)
```

#### Solution: Regularized Polynomial

```python
# Balanced: Polynomial degree 3 with Ridge regularization
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = Ridge(alpha=10.0)  # Regularization prevents overfitting
model.fit(X_poly_train, y_train)

# Result:
# Train RMSE: $12K
# Test RMSE: $14K
# Diagnosis: Good fit (complexity + regularization balanced)
```

---

## Interview Questions

### Q1: "How do you know if your model is overfitting?"

**Answer:**
"I check for these signs:

1. **Performance gap:** Training accuracy much higher than validation (e.g., 99% vs 75%)
2. **Learning curves:** Large gap between training and validation error
3. **Model complexity:** Very deep trees, high polynomial degrees, many parameters
4. **Validation performance:** Accuracy decreases as training continues (validation curve goes up)

To confirm, I'd use cross-validation. If CV score is much lower than training score, it's overfitting."

### Q2: "Your model has 60% train and 59% validation accuracy. What's wrong?"

**Answer:**
"This is underfitting - both scores are low and similar. The model is too simple to capture the pattern.

**I'd try:**
1. More complex model (tree → forest, linear → polynomial)
2. Feature engineering (interactions, polynomial features)
3. Reduce regularization if using any
4. Check if problem is learnable (do features correlate with target?)
5. Add more relevant features

I'd start with #2 (feature engineering) as it often gives biggest gains."

### Q3: "What's the difference between high bias and high variance?"

**Answer:**
```
High Bias (Underfitting):
- Systematic error
- Model too simple
- Misses patterns in data
- Fix: Increase complexity

High Variance (Overfitting):
- Sensitivity to training data
- Model too complex
- Captures noise as pattern
- Fix: Decrease complexity or get more data

Both are bad, but for different reasons.
```

### Q4: "Can a model have both high bias and high variance?"

**Answer:**
"In practice, rare but possible:
- Wrong model for problem type (e.g., linear model for hierarchical data)
- Bad initialization in neural networks
- Extremely noisy data with insufficient preprocessing

Usually one dominates. Use learning curves to diagnose which is the primary issue."

### Q5: "How does regularization prevent overfitting?"

**Answer:**
"Regularization adds a penalty for model complexity:

**L2 (Ridge):** Penalty = λ × Σ(weights²)
- Discourages large weights
- Smooths decision boundary
- Model less sensitive to individual training examples

**L1 (Lasso):** Penalty = λ × Σ|weights|
- Drives some weights to exactly zero
- Automatic feature selection
- Simpler model

**Effect:** Model must balance fitting training data vs keeping weights small. This prevents memorizing noise."

---

## Key Takeaways

1. **Underfitting = too simple** (high bias) → Fix: increase complexity
2. **Overfitting = too complex** (high variance) → Fix: decrease complexity or more data
3. **Diagnose with learning curves** - visualize train vs validation error
4. **Gap > 15-20%** between train and validation suggests overfitting
5. **Both errors high** suggests underfitting
6. **Regularization is key** - L1, L2, Dropout, Early Stopping
7. **More data almost always helps** overfitting
8. **Cross-validation** gives better estimate of true performance

---

## Quick Reference

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Train: 65%, Val: 64% | Underfitting | Increase complexity |
| Train: 99%, Val: 72% | Overfitting | Decrease complexity, add regularization |
| Train: 85%, Val: 83% | Good Fit | Fine-tune if needed |
| Both improving slowly | Need more training | Train longer |
| Train good, Val getting worse | Overfitting starting | Early stopping |

---

**Next:** [Feature Engineering Fundamentals](./feature_engineering_fundamentals.md) | **Back:** [README](./README.md)
