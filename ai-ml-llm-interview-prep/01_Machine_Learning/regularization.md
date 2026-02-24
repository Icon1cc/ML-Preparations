# Regularization - Complete Interview Guide

> Regularization is the fundamental tool for controlling the bias-variance tradeoff. A deep understanding of L1 and L2 regularization—geometrically, analytically, and from a Bayesian perspective—is expected at the senior level.

---

## Table of Contents
1. [The Overfitting Problem](#overfitting)
2. [L2 Regularization (Ridge)](#ridge)
3. [L1 Regularization (Lasso)](#lasso)
4. [Why L1 Produces Sparsity: Geometric Proof](#geometric-proof)
5. [ElasticNet: Combining L1 and L2](#elasticnet)
6. [Bayesian Interpretation](#bayesian)
7. [Comparison Table: When to Use Each](#comparison)
8. [Hyperparameter Tuning for Lambda](#tuning)
9. [Group Lasso and Sparse Group Lasso](#group-lasso)
10. [Regularization in Neural Networks](#neural-networks)
11. [Code Examples](#code)
12. [Interview Questions](#interview-questions)

---

## 1. The Overfitting Problem {#overfitting}

**Overfitting** occurs when a model learns the training data too well, capturing noise rather than signal, and generalizes poorly to new data.

### Formal Definition in Terms of Bias-Variance

For a model's expected test error:
```
E[(y - f̂(x))²] = Bias²(f̂(x)) + Var(f̂(x)) + σ²
```

Where:
- **Bias** = `E[f̂(x)] - f(x)`: systematic error from wrong assumptions
- **Variance** = `E[(f̂(x) - E[f̂(x)])²]`: sensitivity to training data fluctuations
- **σ²** = irreducible noise

**Overfitting = high variance:** The model is too sensitive to the specific training sample. Slightly different training data → very different model.

**Underfitting = high bias:** The model is too simple to capture the true pattern.

### Why High-Degree Polynomial Models Overfit

A degree-15 polynomial fit to 20 points will pass through every training point (interpolation) but oscillate wildly between them. The model has 16 free parameters for 20 observations — not enough degrees of freedom to avoid overfitting.

**Regularization adds a penalty on model complexity** to the loss function, forcing parameters to stay small (less capacity to memorize noise):

```
Total Loss = Data Fit Term + λ × Complexity Penalty
```

The parameter λ controls the tradeoff:
- λ = 0: Pure data fit (may overfit)
- λ → ∞: All coefficients shrink to zero (underfit)
- Optimal λ: Determined by cross-validation

---

## 2. L2 Regularization (Ridge) {#ridge}

### Formulation

**Ridge regression** adds the L2 penalty on coefficients:

```
β̂_ridge = argmin_β { ||y - Xβ||² + λ||β||² }
         = argmin_β { Σᵢ(yᵢ - xᵢ'β)² + λΣⱼβⱼ² }
```

### Closed-Form Solution

**Taking the gradient and setting to zero:**
```
∂/∂β [||y-Xβ||² + λ||β||²]
= -2X'(y - Xβ) + 2λβ = 0
X'y = (X'X + λI)β
β̂_ridge = (X'X + λI)⁻¹X'y
```

**Key insight:** Adding λI to X'X ensures the matrix is **always invertible** (positive definite), even with perfect multicollinearity! The eigenvalues of `X'X + λI` are `{μⱼ + λ}` where `{μⱼ}` are eigenvalues of X'X. Since λ > 0, all eigenvalues are positive.

### Bias and Variance of Ridge

Let the OLS solution be `β̂ = (X'X)⁻¹X'y` and the Ridge solution `β̂_R = (X'X + λI)⁻¹X'y`. Define `A_λ = (X'X + λI)⁻¹X'X`.

**Bias:**
```
E[β̂_R] = A_λβ ≠ β    (biased, unless λ=0)
Bias = E[β̂_R] - β = (A_λ - I)β = -λ(X'X + λI)⁻¹β
```

Bias is proportional to λ and to β — larger λ → more shrinkage.

**Variance:**
```
Var(β̂_R) = σ²(X'X + λI)⁻¹X'X(X'X + λI)⁻¹
```

This is strictly smaller than `Var(β̂_OLS) = σ²(X'X)⁻¹`.

**Ridge can have smaller MSE than OLS** when the variance reduction exceeds the introduced bias. This is the entire motivation for regularization.

### SVD Perspective

Using SVD: `X = UΣV'` where U, V are orthogonal and Σ = diag(d₁, ..., dₚ).

**OLS fitted values:** `ŷ_OLS = Σⱼ uⱼ (uⱼ'y) = Σⱼ uⱼ dⱼ²/(dⱼ²) (uⱼ'y)`

**Ridge fitted values:** `ŷ_Ridge = Σⱼ uⱼ dⱼ²/(dⱼ² + λ) (uⱼ'y)`

The shrinkage factor `dⱼ²/(dⱼ² + λ)`:
- ≈ 1 for large singular values (principal directions — well-determined)
- ≈ 0 for small singular values (noisy directions — shrunk)

Ridge is particularly effective at shrinking directions corresponding to small eigenvalues of X'X — exactly the directions where OLS is most unstable.

### Ridge Degrees of Freedom

```
df(λ) = trace(H_λ) = trace(X(X'X + λI)⁻¹X') = Σⱼ dⱼ²/(dⱼ² + λ)
```

- λ = 0: df = p (all freedom, OLS)
- λ → ∞: df → 0 (no freedom, predicts mean)

Ridge continuously reduces effective degrees of freedom.

---

## 3. L1 Regularization (Lasso) {#lasso}

### Formulation

**Lasso (Least Absolute Shrinkage and Selection Operator)** adds the L1 penalty:

```
β̂_lasso = argmin_β { ||y - Xβ||² + λΣⱼ|βⱼ| }
```

### No Closed-Form Solution

The L1 penalty `|βⱼ|` is **not differentiable at βⱼ = 0**. Subgradient at 0 is the interval `[-1, 1]`.

The KKT optimality conditions for variable j:
```
-2Xⱼ'(y - Xβ) + λ∂|βⱼ| ∋ 0
```

Where the subgradient `∂|βⱼ| = sign(βⱼ)` if βⱼ ≠ 0, and ∈ [-1, 1] if βⱼ = 0.

For orthogonal design (X'X = I), the Lasso solution is:

```
β̂_lasso_j = sign(β̂_OLS_j) × max(|β̂_OLS_j| - λ/2, 0)
```

This is the **soft-thresholding operator** `S(z, γ)`:
- If |z| > γ: `S(z, γ) = z - γ × sign(z)` (shrink toward 0 by γ)
- If |z| ≤ γ: `S(z, γ) = 0` (exactly zero)

**The soft thresholding produces exact zeros** — this is why Lasso performs feature selection!

### Coordinate Descent for Lasso

For the general (non-orthogonal) case, **coordinate descent** is the standard algorithm:

```
For each j ∈ {1, ..., p}:
    r_j = y - Σ_{k≠j} x_k β_k    (partial residual)
    β_j = S(x_j'r_j / ||x_j||², λ / (2||x_j||²))
```

Cycle through all j until convergence. sklearn implements this (LARS algorithm is an alternative for the full Lasso path).

### The Lasso Path

As λ increases from 0 to ∞, coefficients shrink and variables enter/leave the model. The **regularization path** shows which variables are selected at each λ. This is a form of embedded feature selection.

```mermaid
graph LR
    A["λ = 0 (OLS)"] --> B["λ small: All features active"]
    B --> C["λ increases: Weakest features hit zero"]
    C --> D["λ large: Only strongest features remain"]
    D --> E["λ = ∞: All features zero (predict mean)"]

    style A fill:#27ae60,color:#fff
    style E fill:#e74c3c,color:#fff
```

---

## 4. Why L1 Produces Sparsity but L2 Does Not: Geometric Proof {#geometric-proof}

This is a canonical interview question. There are two complementary explanations.

### Geometric Explanation (Constraint Form)

The Lagrangian form of regularized regression is equivalent to a **constrained optimization**:

**Ridge (L2):**
```
minimize ||y - Xβ||²  subject to  Σⱼβⱼ² ≤ t
```

The constraint region is a **sphere** (circle in 2D).

**Lasso (L1):**
```
minimize ||y - Xβ||²  subject to  Σⱼ|βⱼ| ≤ t
```

The constraint region is a **diamond** (L1 ball has corners on axes).

The OLS solution lies outside the constraint region (otherwise regularization wouldn't matter). We move the elliptical contours of the OLS objective (centered at β̂_OLS) inward until they touch the constraint region:

- **L2 sphere:** The first contact point can be anywhere on the sphere's smooth surface. Very unlikely to be exactly on an axis (where βⱼ = 0).
- **L1 diamond:** The first contact point is very likely to be at a **corner** of the diamond, which is located ON the axes (where one or more βⱼ = 0).

In high dimensions (p features), the L1 ball is a cross-polytope with `2p` vertices/corners, all on the coordinate axes. The probability of touching a corner (and thus setting some βⱼ = 0) grows with p.

```
            β₂
             |     ← L2 ball (circle)
             |  /
    ─────────+─────── β₁
          /  |
     ◇ ← L1 ball (diamond, corners at axes)
```

When the loss function contour first touches the L1 ball, it hits the diamond corner → β₁ = 0 exactly.

### Analytical Explanation (Subgradient)

**Ridge:** The penalty `λβⱼ²` has derivative `2λβⱼ`. For βⱼ to be zero at the optimum, we need: `|∂Loss/∂βⱼ| = 0`, but the Ridge derivative is `2λβⱼ` which can only be zero at βⱼ = 0 if there's no other force. In practice, the data gradient almost always prevents this.

**Lasso:** The penalty `λ|βⱼ|` has subgradient `λ × sign(βⱼ)` at βⱼ ≠ 0, and `[-λ, λ]` at βⱼ = 0. The optimality condition is:

```
∂RSS/∂βⱼ + λ × sign(βⱼ) = 0
```

If the data gradient `|∂RSS/∂βⱼ|` at βⱼ = 0 is **less than λ**, then the subgradient condition is satisfied at βⱼ = 0 → coefficient stays zero! L1 creates a "dead zone" where small gradients can't overcome the penalty → exact sparsity.

---

## 5. ElasticNet: Combining L1 and L2 {#elasticnet}

### Formulation

```
β̂_EN = argmin_β { ||y - Xβ||² + λ₁Σⱼ|βⱼ| + λ₂Σⱼβⱼ² }
```

Often parameterized as:
```
β̂_EN = argmin_β { ||y - Xβ||² + λ[α||β||₁ + (1-α)/2 ||β||²] }
```

Where:
- λ controls overall regularization strength
- α ∈ [0, 1] is the "mixing parameter" (α=1: pure Lasso, α=0: pure Ridge)

### Motivation

**Lasso's weakness:** When features are highly correlated, Lasso arbitrarily selects one and sets the others to zero. This is inconsistent (random subset selection among correlated features) and ignores group structure.

**ElasticNet's advantage:** The L2 component introduces a grouping effect — correlated features tend to be selected or excluded together. If features are correlated, their coefficients tend to be similar in magnitude.

### Properties

- Sparsity (from L1 component): Not as sparse as pure Lasso
- Stability (from L2 component): More stable than pure Lasso with correlated features
- Can select more than n variables (Lasso can select at most n variables)
- Preferred default when features are correlated and you still want some sparsity

---

## 6. Bayesian Interpretation {#bayesian}

Regularization has an elegant Bayesian interpretation. The regularized objective corresponds to **MAP (Maximum A Posteriori) estimation** with a specific prior on β.

```
MAP estimate = argmax_β P(β | data) = argmax_β [P(data | β) × P(β)]
```

Taking the log:
```
log P(β | data) ∝ log P(data | β) + log P(β)
= Data log-likelihood + Log prior
```

### Ridge as Gaussian Prior

If `βⱼ ~ N(0, σ²_β)` i.i.d., then:
```
log P(β) = -Σⱼ βⱼ² / (2σ²_β) + const
```

With Gaussian likelihood (OLS):
```
argmax_β [log-likelihood + log P(β)]
= argmin_β { ||y - Xβ||² + (σ²/σ²_β) Σⱼ βⱼ² }
= argmin_β { ||y - Xβ||² + λ||β||² }
```

Where `λ = σ²/σ²_β`. **Ridge regression = MAP with Gaussian prior on coefficients.**

- Small σ²_β (tight prior): Strong regularization, coefficients shrink heavily
- Large σ²_β (diffuse prior): Weak regularization, approaches OLS

### Lasso as Laplace Prior

If `βⱼ ~ Laplace(0, b)` i.i.d., then:
```
log P(β) = -Σⱼ |βⱼ| / b + const
```

This gives:
```
argmax_β [log-likelihood + log P(β)]
= argmin_β { ||y - Xβ||² + (2σ²/b) Σⱼ|βⱼ| }
= argmin_β { ||y - Xβ||² + λ||β||₁ }
```

**Lasso regression = MAP with Laplace prior on coefficients.**

The Laplace distribution has **heavier tails** than Gaussian and a **sharper peak at zero**. This is why it promotes sparsity: the prior assigns high probability mass to coefficients near zero AND allows occasional large coefficients (heavy tails). The Gaussian prior is smooth at zero, not actively promoting exact zeros.

### Why Laplace → Sparsity vs Gaussian → Shrinkage

```
Gaussian prior:  f(β) = (1/√2π σ) exp(-β²/2σ²)    # smooth at 0
Laplace prior:   f(β) = (1/2b) exp(-|β|/b)          # sharp peak at 0
```

The Laplace prior has a **non-differentiable peak at zero** that creates the dead zone in the MAP objective, producing exact zeros. The Gaussian prior is smooth at zero — no dead zone.

### Comparison

| Prior | Penalty | Effect | Shape |
|-------|---------|--------|-------|
| Gaussian `N(0, σ²)` | L2: `λ||β||²` | Shrinkage, never exactly zero | Smooth |
| Laplace `Laplace(0, b)` | L1: `λ||β||₁` | Sparsity, exact zeros | Sharp peak at 0 |
| Spike-and-Slab | L0 | True subset selection | Mixture |
| Horseshoe | Adaptive | Aggressive sparsity + little shrinkage on large coefs | Heavy-tailed |

---

## 7. Comparison Table: When to Use Each {#comparison}

| Property | OLS | Ridge (L2) | Lasso (L1) | ElasticNet |
|----------|-----|-----------|-----------|------------|
| Feature selection | No | No | Yes (exact zeros) | Partial |
| Handles multicollinearity | No | Yes | Partially (unstable) | Yes |
| Correlated feature groups | Arbitrary | Shrinks together | Arbitrarily picks one | Selects group |
| Closed-form solution | Yes | Yes | No | No |
| Number of nonzero coefs | p | p | ≤ min(n,p) | Between L1 and p |
| Computational cost | Low | Low | Medium | Medium |
| Interpretability | High | High | High (sparse) | High (sparse) |
| Best when | p << n, no multicollinearity | Multicollinearity, keep all features | Sparse ground truth, feature selection | Correlated features, want some sparsity |

---

## 8. Hyperparameter Tuning for Lambda {#tuning}

### Cross-Validation Path

The standard approach: fit model for a grid of λ values, evaluate each with k-fold CV:

```python
alphas = np.logspace(-4, 4, 100)  # 10^-4 to 10^4
# sklearn's RidgeCV / LassoCV does this efficiently
```

### The Regularization Path

For Lasso, the full **regularization path** (coefficients as a function of λ) can be computed via LARS (Least Angle Regression) in `O(p²n)` time — same cost as a single OLS fit.

For Ridge, closed-form at each λ.

### Hyperparameter Selection Criteria

| Method | Pros | Cons |
|--------|------|------|
| K-Fold CV | Unbiased estimate | Expensive for large grids |
| BIC | Consistent (selects true model as n→∞) | Assumes true model is in set |
| AIC | Optimizes predictive performance | Not consistent |
| EBIC | Good for high-dimensional selection | Complex |
| Stability selection | Robust feature selection | Requires two tuning parameters |

**Practical recommendation:** Use `LassoCV` or `RidgeCV` with `cv=5`. They implement efficient warm-start along the λ path.

---

## 9. Group Lasso and Sparse Group Lasso {#group-lasso}

### Group Lasso

When features naturally form groups (e.g., dummy variables for one categorical, lagged versions of one time series), we want to select or deselect **entire groups**:

```
β̂_GL = argmin_β { ||y - Xβ||² + λ Σ_g ||β_g||₂ }
```

Where `||β_g||₂ = √(Σⱼ∈g βⱼ²)` is the Euclidean norm of coefficients in group g.

Effect: All coefficients in a group go to zero together (group-level sparsity), or none go to zero.

### Sparse Group Lasso

Combines group-level sparsity (Group Lasso) with within-group sparsity (Lasso):

```
β̂_SGL = argmin_β { ||y - Xβ||² + λ₁ Σ_g ||β_g||₂ + λ₂ ||β||₁ }
```

Useful when you have groups of related features but expect only some features within each group to matter.

---

## 10. Regularization in Neural Networks {#neural-networks}

### Weight Decay (L2)

In neural networks, L2 regularization on weights is called **weight decay**:

```
Loss = Task Loss + λ Σ_{all weights w} w²
```

In modern frameworks (PyTorch), weight decay is a parameter of the optimizer:
```python
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
```

Weight decay prevents weights from growing large, keeps the effective function class simpler, and improves generalization.

**Note:** Weight decay is NOT equivalent to L2 regularization with adaptive optimizers (Adam). With SGD they're equivalent, but with Adam, they differ (this led to the AdamW fix).

### Dropout

**Dropout** randomly zeros out neurons with probability p during training:

```python
# During training: neuron output = 0 with prob p, else x/(1-p)
# During inference: use all neurons with weights scaled by (1-p)
```

**Connection to regularization:**
- Acts as an ensemble method: each forward pass is a different network
- Equivalent (approximately) to L2 regularization on the product of weights in linear networks
- Prevents co-adaptation: neurons can't rely on specific other neurons

### Batch Normalization

**Batch normalization** normalizes layer activations:

```
ẑ = (z - μ_batch) / σ_batch × γ + δ
```

While primarily for training stability, BatchNorm has regularization effects:
- Adds noise from batch statistics (acts like dropout)
- Allows higher learning rates
- Reduces sensitivity to initialization

### Early Stopping

**Early stopping** monitors validation loss and stops training when it begins to increase. This is implicit regularization: it limits the number of gradient updates (effective model capacity).

**Relationship to L2:** For linear models trained with gradient descent, early stopping is approximately equivalent to L2 regularization with `λ ∝ 1/(learning_rate × iterations)`.

---

## 11. Code Examples {#code}

### Ridge vs Lasso vs ElasticNet Comparison

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LinearRegression,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. Generate data with known sparsity + correlated features
# ──────────────────────────────────────────────
np.random.seed(42)
n, p = 200, 50

# Correlated feature structure
Sigma = np.eye(p)
for i in range(p):
    for j in range(p):
        Sigma[i, j] = 0.7 ** abs(i - j)  # AR(1) correlation

L = np.linalg.cholesky(Sigma)
X = np.random.randn(n, p) @ L.T

# True coefficients: only 10 of 50 are nonzero
true_beta = np.zeros(p)
true_beta[[0, 1, 5, 10, 20, 25, 30, 35, 40, 45]] = [3, -2, 1.5, -1, 2, 0.8, -1.2, 0.9, -0.7, 1.1]

y = X @ true_beta + np.random.normal(0, 1, n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ──────────────────────────────────────────────
# 2. Cross-validated model selection for each method
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# OLS
ols = LinearRegression().fit(X_train_scaled, y_train)
ols_rmse = np.sqrt(mean_squared_error(y_test, ols.predict(X_test_scaled)))

# Ridge with CV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_cv.predict(X_test_scaled)))

# Lasso with CV
lasso_cv = LassoCV(n_alphas=100, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_cv.predict(X_test_scaled)))

# ElasticNet with CV
en_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], n_alphas=50, cv=5)
en_cv.fit(X_train_scaled, y_train)
en_rmse = np.sqrt(mean_squared_error(y_test, en_cv.predict(X_test_scaled)))

print("Model Comparison:")
print(f"{'Model':<15} {'RMSE':<10} {'R²':<10} {'Nonzero Coefs':<15} {'Best λ':<10}")
print("-" * 60)

models = [
    ('OLS', ols, ols_rmse, None),
    ('Ridge', ridge_cv, ridge_rmse, ridge_cv.alpha_),
    ('Lasso', lasso_cv, lasso_rmse, lasso_cv.alpha_),
    ('ElasticNet', en_cv, en_rmse, en_cv.alpha_),
]

for name, model, rmse, alpha in models:
    coefs = model.coef_
    nonzero = np.sum(np.abs(coefs) > 1e-6)
    r2 = r2_score(y_test, model.predict(X_test_scaled))
    alpha_str = f"{alpha:.5f}" if alpha else "N/A"
    print(f"{name:<15} {rmse:<10.4f} {r2:<10.4f} {nonzero:<15} {alpha_str:<10}")

# True nonzero for reference
print(f"\nTrue nonzero coefficients: {(true_beta != 0).sum()}")

# ──────────────────────────────────────────────
# 3. Regularization path visualization
# ──────────────────────────────────────────────
from sklearn.linear_model import lasso_path, ridge_regression

alphas_lasso, coefs_lasso, _ = lasso_path(
    X_train_scaled, y_train,
    alphas=np.logspace(-3, 1, 100)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lasso path
ax = axes[0]
for i in range(p):
    ax.plot(np.log10(alphas_lasso), coefs_lasso[i], linewidth=0.8, alpha=0.7)
ax.set_xlabel('log10(α)')
ax.set_ylabel('Coefficient')
ax.set_title('Lasso Regularization Path\n(Features enter/leave as α increases)')
ax.axvline(x=np.log10(lasso_cv.alpha_), color='red', linestyle='--', label=f'CV-optimal α')
ax.legend()

# Ridge path
alphas_ridge = np.logspace(-3, 4, 100)
coefs_ridge = np.zeros((p, len(alphas_ridge)))
for i, a in enumerate(alphas_ridge):
    r = Ridge(alpha=a).fit(X_train_scaled, y_train)
    coefs_ridge[:, i] = r.coef_

ax = axes[1]
for i in range(p):
    ax.plot(np.log10(alphas_ridge), coefs_ridge[i], linewidth=0.8, alpha=0.7)
ax.set_xlabel('log10(α)')
ax.set_ylabel('Coefficient')
ax.set_title('Ridge Regularization Path\n(All coefficients shrink continuously, never zero)')
ax.axvline(x=np.log10(ridge_cv.alpha_), color='red', linestyle='--', label=f'CV-optimal α')
ax.legend()

plt.tight_layout()
plt.savefig('regularization_paths.png', dpi=150, bbox_inches='tight')

# ──────────────────────────────────────────────
# 4. Coefficient recovery comparison
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(p)
width = 0.2

ax.bar(x - 1.5*width, true_beta, width, label='True β', color='black', alpha=0.7)
ax.bar(x - 0.5*width, ols.coef_, width, label='OLS', color='blue', alpha=0.7)
ax.bar(x + 0.5*width, lasso_cv.coef_, width, label='Lasso', color='orange', alpha=0.7)
ax.bar(x + 1.5*width, ridge_cv.coef_, width, label='Ridge', color='green', alpha=0.7)

ax.set_xlabel('Feature Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Coefficient Recovery: True vs OLS vs Lasso vs Ridge')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('coefficient_recovery.png', dpi=150, bbox_inches='tight')
```

### Neural Network Regularization

```python
# ──────────────────────────────────────────────
# 5. Neural network regularization comparison (PyTorch-style pseudocode)
# ──────────────────────────────────────────────

# PyTorch weight decay (L2)
import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout_rate),          # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Training with weight decay (L2 on weights, NOT biases)
# Note: AdamW correctly decouples weight decay from gradient adaptation
def train_with_regularization():
    model = RegularizedNet(50, 128, 1)

    # AdamW: weight decay applied directly, not via gradient
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4   # L2 penalty on weights
    )

    # Alternative: SGD with momentum + weight decay
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
    #                              momentum=0.9, weight_decay=1e-4)

    # Early stopping callback
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(1000):
        model.train()
        # ... training loop ...

        model.eval()
        with torch.no_grad():
            val_loss = 0  # ... compute validation loss ...

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

# ──────────────────────────────────────────────
# 6. Comparing regularization strength effect
# ──────────────────────────────────────────────
from sklearn.preprocessing import PolynomialFeatures

# Generate 1D data to show overfitting
np.random.seed(42)
n_demo = 30
X_demo = np.sort(np.random.uniform(0, 10, n_demo)).reshape(-1, 1)
y_demo = np.sin(X_demo.ravel()) + np.random.normal(0, 0.3, n_demo)

# Degree-15 polynomial features
poly = PolynomialFeatures(degree=15, include_bias=False)
X_poly = poly.fit_transform(X_demo)
X_poly_scaled = StandardScaler().fit_transform(X_poly)

X_fine = np.linspace(0, 10, 300).reshape(-1, 1)
X_fine_poly = poly.transform(X_fine)
X_fine_scaled = StandardScaler().fit_transform(X_fine_poly)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
configs = [
    ('OLS (λ=0)', LinearRegression(), 'red'),
    ('Ridge (λ=0.1)', Ridge(alpha=0.1), 'orange'),
    ('Ridge (λ=10)', Ridge(alpha=10), 'green'),
]

for ax, (title, model, color) in zip(axes, configs):
    model.fit(X_poly_scaled, y_demo)
    y_fine_pred = model.predict(X_fine_scaled)

    ax.scatter(X_demo, y_demo, color='black', s=30, zorder=5, label='Training data')
    ax.plot(X_fine, y_fine_pred, color=color, linewidth=2, label=f'{title}')
    ax.plot(X_fine, np.sin(X_fine), 'k--', linewidth=1, alpha=0.5, label='True function')
    ax.set_ylim(-3, 3)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlabel('X')

plt.suptitle('Effect of L2 Regularization on Degree-15 Polynomial', y=1.02)
plt.tight_layout()
plt.savefig('regularization_effect.png', dpi=150, bbox_inches='tight')
```

---

## 12. Interview Questions {#interview-questions}

### Q1: "Why does L1 produce sparsity but L2 doesn't? Prove it geometrically."

**Answer:** (See Section 4 for full derivation, summary here)

The geometric argument: Regularized regression is equivalent to constrained optimization. L1 constrains to an L1 ball (diamond shape in 2D), L2 constrains to an L2 ball (sphere). When we shrink the loss function contours inward from the OLS solution, they hit:
- L1 ball: likely at a **corner** (on the coordinate axes where βⱼ = 0) → exact zeros
- L2 ball: likely at a **smooth surface point** (off the axes) → no exact zeros

Analytic proof: The L1 penalty creates a "dead zone" at βⱼ = 0. A coordinate is set to zero when the absolute value of its data gradient is less than λ (the subgradient condition). L2 penalty has zero subgradient ambiguity — the gradient is 2λβⱼ, which can always be matched exactly without forcing βⱼ to zero.

### Q2: "What is the Bayesian interpretation of Ridge regression?"

**Answer:** Ridge regression is equivalent to MAP estimation with a Gaussian prior `βⱼ ~ N(0, σ²_β)` on each coefficient. The posterior mode (MAP estimate) is:

```
β̂_MAP = argmin { -log P(y|X,β) - log P(β) }
       = argmin { ||y-Xβ||² / (2σ²) + Σⱼ βⱼ² / (2σ²_β) }
       = argmin { ||y-Xβ||² + (σ²/σ²_β) Σⱼ βⱼ² }
```

Where λ = σ²/σ²_β. A tighter prior (smaller σ²_β) → stronger regularization. Similarly, Lasso corresponds to a Laplace prior.

### Q3: "How do you choose the regularization parameter λ?"

**Answer:** Cross-validation is the gold standard. Use `LassoCV` or `RidgeCV` which efficiently compute the path across many λ values using warm-starting (the previous solution is the starting point for the next λ).

For Lasso, the "one-standard-error rule" is often preferred over the minimum-CV rule: choose the largest λ whose CV error is within one standard error of the minimum. This gives a more parsimonious model without significantly worse CV performance.

For neural networks, treat it as a standard hyperparameter with Bayesian optimization (Optuna) or random search.

### Q4: "When would you use ElasticNet over Lasso?"

**Answer:** Use ElasticNet when:
1. **Features are highly correlated:** Lasso arbitrarily selects one from a correlated group. ElasticNet's L2 component produces a grouping effect, giving similar coefficients to correlated features.
2. **p >> n:** Lasso selects at most n variables. ElasticNet can select more.
3. **You want stability in feature selection:** Lasso's selected features can change dramatically with small perturbations in data (high variance in variable selection). ElasticNet is more stable.

Example: In genomics with 50,000 SNPs and 200 subjects, Lasso picks at most 200 SNPs and ignores correlation structure. ElasticNet handles this better.

### Q5: "Explain dropout in neural networks and its connection to regularization."

**Answer:** During training, dropout randomly zeros activations with probability p, equivalent to sampling a different "thinned" network at each step. This forces redundant representations (no single neuron can rely on any specific other neuron) and effectively trains an ensemble of 2^n networks sharing weights.

Connection to regularization:
- For linear networks, dropout is equivalent to an adaptive L2 penalty where the regularization strength depends on the input magnitude.
- It reduces co-adaptation and complex interactions between neurons.
- Mathematically, training with dropout maximizes the lower bound on the log-likelihood of an ensemble of models.

At inference, all neurons are used with weights scaled by (1-p) to match expected activation from training.

### Q6: "What is the difference between L1 and L0 regularization, and why don't we use L0?"

**Answer:**
- **L0 "regularization":** Penalize by the number of nonzero coefficients `||β||₀ = #{j : βⱼ ≠ 0}`. This would give true best-subset selection — find the k-sparse model that best fits the data.
- **Problem:** L0 is combinatorially hard (NP-hard in general). For p features, there are `2^p` possible subsets. For p=50, that's 10^15 models.
- **L1 as convex relaxation of L0:** Lasso is the tightest convex relaxation of the L0 penalty. Under certain conditions (restricted isometry property, incoherence), Lasso recovers the same sparse solution as L0.

### Q7: "Does regularization always improve test performance?"

**Answer:** Not always. Regularization introduces bias. If the true model has no sparse structure (all features are relevant and of similar magnitude), L1 regularization will harm performance by forcing some coefficients to zero incorrectly. Ridge might still help by reducing variance if n/p is small.

The optimal strategy depends on:
1. True sparsity of β
2. Signal-to-noise ratio
3. Correlation structure
4. Sample size relative to features

With large n and small p, OLS is often fine. Regularization primarily helps when p is large relative to n.

---

## Summary Cheat Sheet

| Method | Penalty | Prior | Closed Form | Sparsity |
|--------|---------|-------|-------------|----------|
| OLS | None | Flat | Yes | No |
| Ridge | λ||β||² | Gaussian | Yes | No |
| Lasso | λ||β||₁ | Laplace | No | Yes |
| ElasticNet | λ(α||β||₁ + (1-α)||β||²) | Mixture | No | Partial |
| Group Lasso | λΣ_g||β_g||₂ | — | No | Group-level |

**Decision rule:**
- Low p, no multicollinearity → OLS
- Multicollinearity, interpretability → Ridge
- Suspect sparse signal, feature selection needed → Lasso
- Correlated features + sparsity → ElasticNet
- Group structure in features → Group Lasso

---

*Next: [Decision Trees](./decision_trees.md) - information-theoretic splitting*
