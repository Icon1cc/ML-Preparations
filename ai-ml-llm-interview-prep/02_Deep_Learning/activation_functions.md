# Activation Functions

> **Interview relevance:** Know each activation's formula, derivative, range, and when to use it. The most common question is "Why ReLU over sigmoid in hidden layers?" and "Why does BERT/GPT use GELU?" Be ready to discuss the dying ReLU problem and numerical stability of softmax.

---

## Table of Contents

1. [Sigmoid](#1-sigmoid)
2. [Tanh](#2-tanh)
3. [ReLU](#3-relu)
4. [Leaky ReLU](#4-leaky-relu)
5. [Parametric ReLU (PReLU)](#5-parametric-relu-prelu)
6. [ELU](#6-elu)
7. [GELU](#7-gelu)
8. [SiLU / Swish](#8-silu--swish)
9. [Mish](#9-mish)
10. [Softmax](#10-softmax)
11. [Output Layer Guidelines](#11-output-layer-guidelines)
12. [Comparison Table](#12-comparison-table)
13. [Interview Questions](#13-interview-questions)
14. [Code: Implement and Visualize All Activations](#14-code)

---

## 1. Sigmoid

### Formula

```
σ(z) = 1 / (1 + e^{-z})
```

### Derivative

```
σ'(z) = σ(z)(1 - σ(z))
```

This can be derived: `d/dz [1/(1+e^{-z})] = e^{-z}/(1+e^{-z})^2 = σ(z)(1-σ(z))`

### Properties

| Property | Value |
|---------|-------|
| Range | (0, 1) |
| Output at z=0 | 0.5 |
| Max derivative | 0.25 (at z=0) |
| Saturates? | Yes (both ends) |
| Zero-centered? | No (output always positive) |

### When to Use

- **Output layer for binary classification:** Output in (0,1) naturally interpreted as probability
- **Gates in LSTM/GRU:** Need values in (0,1) to act as soft gates
- **Attention mechanisms:** Historically (mostly replaced by softmax now)

### Limitations

1. **Gradient saturation:** Derivative approaches 0 for large |z| → vanishing gradients
2. **Not zero-centered:** All outputs positive → all-positive or all-negative gradients → zig-zag optimization
3. **Computationally expensive:** `exp()` is expensive compared to ReLU's `max(0, z)`

### The "Not Zero-Centered" Problem Explained

If all activations are positive, then the gradient `dL/dW^[l] = δ^[l] (a^{l-1})^T` will have all elements of the same sign (same sign as `δ^[l]`). This means all weights must move in the same direction simultaneously, causing "zig-zagging" optimization that is slower than zero-centered activations.

---

## 2. Tanh

### Formula

```
tanh(z) = (e^z - e^{-z}) / (e^z + e^{-z})
         = 2σ(2z) - 1    (relationship to sigmoid)
```

### Derivative

```
tanh'(z) = 1 - tanh^2(z)
```

### Properties

| Property | Value |
|---------|-------|
| Range | (-1, 1) |
| Output at z=0 | 0 |
| Max derivative | 1 (at z=0) |
| Saturates? | Yes (both ends) |
| Zero-centered? | Yes |

### Improvement Over Sigmoid

- Zero-centered output eliminates the zig-zag optimization problem
- Stronger gradient (max 1 vs 0.25 for sigmoid)
- Still widely used in RNN/LSTM cells for the cell state and hidden state

### Remaining Limitation

Still saturates for large |z|, causing vanishing gradients. For deep feedforward networks, replaced by ReLU variants.

---

## 3. ReLU (Rectified Linear Unit)

### Formula

```
ReLU(z) = max(0, z) = z * 1_{z > 0}
```

### Derivative

```
ReLU'(z) = {1  if z > 0
            {0  if z < 0
            {undefined at z=0 (typically set to 0)
```

### Properties

| Property | Value |
|---------|-------|
| Range | [0, +∞) |
| Max derivative | 1 (for all z > 0) |
| Saturates? | Only on negative side |
| Zero-centered? | No |
| Computationally cheap? | Yes (single comparison) |

### Why ReLU Dominates in Hidden Layers

1. **No gradient saturation for positive inputs:** Derivative is exactly 1, not decaying
2. **Sparsity:** ~50% of neurons have zero output → sparse representations are efficient
3. **Computational simplicity:** `max(0, z)` is trivially fast
4. **Empirically works well:** Especially for image tasks

### The Dying ReLU Problem

**Problem:** If a neuron's pre-activation `z` is consistently negative (e.g., due to large negative bias), its gradient is 0. The weight update is: `ΔW = -lr * (0 * upstream)` = 0. The neuron can never recover.

**Causes:**
- High learning rates that cause large weight updates
- Poor initialization (large negative biases)
- Batch normalization misconfiguration

**Diagnosis:** After training, if many neurons have zero output for all training examples, you have dead neurons.

**Solutions:** Leaky ReLU, ELU, initialize biases to small positive values, lower learning rate.

---

## 4. Leaky ReLU

### Formula

```
LeakyReLU(z) = {z          if z > 0
               {α * z      if z ≤ 0
```

Where `α` is a small constant, typically `0.01`.

### Derivative

```
LeakyReLU'(z) = {1   if z > 0
                {α   if z ≤ 0
```

### Why It Fixes Dying ReLU

For negative inputs, the gradient is `α > 0` instead of 0. Dead neurons now receive a small gradient signal and can recover. Even with `α = 0.01`, learning can resume.

### Tradeoff

- Introduces a hyperparameter `α`
- The small negative slope means negative inputs are not truly "dead" but also don't contribute meaningfully to positive outputs
- Not always better than ReLU empirically — depends on task

---

## 5. Parametric ReLU (PReLU)

### Formula

```
PReLU(z; α) = {z      if z > 0
              {α * z  if z ≤ 0
```

The difference from Leaky ReLU: **`α` is a learned parameter**, not a fixed constant.

### Gradient w.r.t. α

```
dL/dα = Σ_{z_i < 0} (dL/dz_i) * z_i
```

This gradient is backpropagated like any other parameter.

### When to Use

- He et al. (2015) introduced PReLU for very deep image classification networks
- Can provide a modest improvement over fixed Leaky ReLU
- Adds a few parameters (one per channel typically, not per neuron)

---

## 6. ELU (Exponential Linear Unit)

### Formula

```
ELU(z; α) = {z               if z > 0
            {α(e^z - 1)      if z ≤ 0
```

Typical value: `α = 1.0`

### Derivative

```
ELU'(z; α) = {1               if z > 0
             {α * e^z         if z ≤ 0
             = ELU(z; α) + α  for z ≤ 0   (convenient form)
```

### Properties

| Property | Value |
|---------|-------|
| Range | (-α, +∞) |
| Zero-centered? | Approximately yes (negative outputs possible) |
| Differentiable? | Yes, everywhere (smooth at z=0) |
| Saturates? | Soft saturation at `-α` for large negative z |

### Advantages

1. **Smooth everywhere:** No kink at z=0 (unlike ReLU). Theoretically better optimization landscape.
2. **Negative outputs:** The mean activation can be closer to 0, helping with batch normalization.
3. **No dying neurons:** Negative inputs get nonzero (but small) gradient from `α*e^z`.

### Disadvantage

`exp()` is computationally expensive for negative inputs.

---

## 7. GELU (Gaussian Error Linear Unit)

### Formula

```
GELU(z) = z * Φ(z)
```

Where `Φ(z)` is the CDF of the standard normal distribution:
```
Φ(z) = P(X ≤ z),  X ~ N(0,1) = (1/2)[1 + erf(z/√2)]
```

### Approximation (used in practice)

```
GELU(z) ≈ 0.5z * (1 + tanh[√(2/π) * (z + 0.044715z^3)])
```

Or the simpler sigmoid approximation:
```
GELU(z) ≈ z * σ(1.702z)
```

### Derivative

```
GELU'(z) = Φ(z) + z * φ(z)
```

Where `φ(z) = dΦ/dz` is the standard normal PDF.

### Intuition

GELU is a "stochastic" activation: it applies the input `z` scaled by the probability that a standard Gaussian random variable is less than `z`. Intuitively, inputs are "dropped" proportionally to how negative they are, but smoothly (unlike ReLU's hard threshold).

- For large positive z: `Φ(z) ≈ 1`, so `GELU(z) ≈ z` (like linear/ReLU)
- For large negative z: `Φ(z) ≈ 0`, so `GELU(z) ≈ 0` (like ReLU "off")
- At z=0: `GELU(0) = 0`, `GELU'(0) ≈ 0.5`

### Why Transformers Use GELU

1. **Smooth everywhere:** Unlike ReLU, GELU has no discontinuity in derivative. This produces smoother loss landscapes.
2. **Non-monotonic at small scales:** GELU slightly dips below zero near z ≈ -0.17. This provides a subtle regularization effect.
3. **Works well empirically:** BERT, GPT-2, GPT-3, and most modern transformers use GELU. The difference vs ReLU is small but consistently positive in NLP tasks.
4. **Stochastic interpretation:** GELU can be interpreted as a smooth version of dropout applied during the activation itself.

---

## 8. SiLU / Swish

### Formula

```
SiLU(z) = z * σ(z) = z / (1 + e^{-z})
```

Also called **Swish** (Ramachandran et al., 2017, Google Brain). SiLU is the specific case `β=1` of the more general `x * σ(βx)`.

### Derivative

```
SiLU'(z) = σ(z) + z * σ(z)(1 - σ(z))
          = σ(z)(1 + z(1 - σ(z)))
          = SiLU(z)/z + σ(z)(1 - SiLU(z)/z)
```

### Properties

| Property | Value |
|---------|-------|
| Range | (-0.278..., +∞) |
| Smooth? | Yes (C∞) |
| Non-monotonic? | Yes (slight dip below 0) |
| Self-gated? | Yes (the gate σ(z) is derived from the input itself) |

### Self-Gating Intuition

SiLU multiplies the input by its own sigmoid, which acts as a "self-gate." Positive inputs close to 0 are slightly suppressed; large positive inputs are amplified approximately linearly; negative inputs are suppressed.

### Used In

EfficientNet, MobileNetV3, CLIP, and many modern computer vision models. Often competitive with or better than GELU for vision tasks.

---

## 9. Mish

### Formula

```
Mish(z) = z * tanh(softplus(z))
         = z * tanh(ln(1 + e^z))
```

### Derivative

```
Mish'(z) = tanh(softplus(z)) + z * sech^2(softplus(z)) * σ(z)
```

Where `sech^2 = 1 - tanh^2`.

### Properties

- Smooth, non-monotonic
- Unbounded above, bounded below (≈ -0.31)
- Self-regularizing: slightly negative values are preserved

### When to Use

Mish was shown to outperform ReLU and Swish on several image classification benchmarks (Misra, 2019). Used in some versions of YOLOv4/v5. Less mainstream than GELU/SiLU.

---

## 10. Softmax

### Formula

For a vector `z ∈ R^K`:

```
softmax(z)_k = e^{z_k} / Σ_{j=1}^{K} e^{z_j}
```

### Derivative (Jacobian)

```
d softmax(z)_i / d z_j = softmax(z)_i * (δ_{ij} - softmax(z)_j)
```

In matrix form, the Jacobian `J ∈ R^{K×K}`:
```
J = diag(s) - s s^T    where s = softmax(z)
```

### Numerical Stability: The Log-Sum-Exp Trick

Direct computation of `e^{z_k}` overflows for large z (e.g., `e^{1000} = inf`).

**Solution:** Subtract the maximum value before computing:

```
softmax(z)_k = e^{z_k - max(z)} / Σ_j e^{z_j - max(z)}
```

This is mathematically equivalent:
```
e^{z_k - c} / Σ_j e^{z_j - c} = e^{z_k} * e^{-c} / (Σ_j e^{z_j} * e^{-c}) = e^{z_k} / Σ_j e^{z_j}
```

After subtraction, the maximum exponent is 0 (largest value is 1), preventing overflow.

### Temperature Scaling

```
softmax(z / T)_k = e^{z_k/T} / Σ_j e^{z_j/T}
```

- **T → 0:** Approaches argmax (one-hot distribution)
- **T = 1:** Standard softmax
- **T → ∞:** Approaches uniform distribution

Temperature scaling is used in knowledge distillation, language model sampling, and attention mechanisms.

### Softmax vs Sigmoid for Multi-Label

- **Softmax:** Probabilities sum to 1. Use for mutually exclusive classes.
- **Sigmoid (elementwise):** Each output independent probability. Use for multi-label classification (an image can be both "cat" AND "dog").

---

## 11. Output Layer Guidelines

The output layer activation should match the task's output distribution:

| Task | Output Layer | Loss Function |
|------|-------------|---------------|
| Binary classification | Sigmoid (single unit) | Binary Cross-Entropy |
| Multi-class classification | Softmax | Categorical Cross-Entropy |
| Multi-label classification | Sigmoid (per class) | Binary Cross-Entropy per class |
| Regression (unbounded) | Linear (no activation) | MSE or MAE |
| Regression (bounded [0,1]) | Sigmoid | MSE |
| Regression (positive only) | Softplus or ReLU | MSE |
| Sequence probability | Softmax | Cross-Entropy |
| Variational inference (mean) | Linear | — |
| Variational inference (variance) | Softplus | — |

---

## 12. Comparison Table

| Activation | Range | Differentiable | Saturates | Zero-Centered | Used In |
|-----------|-------|---------------|-----------|--------------|---------|
| Sigmoid | (0, 1) | Yes | Both ends | No | Binary output, LSTM gates |
| Tanh | (-1, 1) | Yes | Both ends | Yes | LSTM/GRU hidden state |
| ReLU | [0, ∞) | Almost | Negative side | No | CNNs, MLP hidden layers |
| Leaky ReLU | (-∞, ∞) | Almost | No | No | CNNs when dying ReLU is a problem |
| PReLU | (-∞, ∞) | Almost | No | No | Deep image models |
| ELU | (-α, ∞) | Yes | Soft negative | Approximately | Deep networks needing stability |
| GELU | ≈(-0.17, ∞) | Yes | Soft | No | BERT, GPT, most transformers |
| SiLU/Swish | ≈(-0.28, ∞) | Yes | Soft | No | EfficientNet, MobileNet |
| Mish | ≈(-0.31, ∞) | Yes | Soft | No | YOLOv4, some CNNs |
| Softmax | (0, 1)^K, sum=1 | Yes | No | No | Multi-class output |

---

## 13. Interview Questions

### Q1: Why is ReLU preferred over sigmoid in hidden layers?

**Strong answer:**

> Three main reasons:
>
> 1. **No gradient saturation for positive inputs:** ReLU's derivative is 1 for z > 0, so gradients flow without decay. Sigmoid's max derivative is 0.25, and it decays toward 0 for large |z|.
>
> 2. **Computational efficiency:** `max(0, z)` is a single comparison, vastly cheaper than `1/(1 + e^{-z})` which requires an exponential.
>
> 3. **Sparse representations:** Approximately 50% of ReLU neurons output 0, creating sparse activations. Sparse representations are often more robust and easier to learn from.
>
> The tradeoff is the dying ReLU problem — neurons with consistently negative pre-activations have zero gradient and cannot recover. Solutions include Leaky ReLU, ELU, or careful initialization.

### Q2: What is GELU and why is it used in transformers?

**Strong answer:**

> GELU (Gaussian Error Linear Unit) is `z * Φ(z)`, where Φ is the standard normal CDF. It can be interpreted as the expected value of a random variable that equals z with probability Φ(z) and 0 otherwise — effectively a smooth, stochastic version of ReLU.
>
> Transformers use GELU because: (1) it is smooth everywhere, which produces smoother loss landscapes; (2) it is slightly non-monotonic near the origin (a subtle regularization effect); (3) it empirically outperforms ReLU on NLP tasks, and this advantage compounds across many transformer layers.
>
> In practice, the tanh approximation `0.5z(1 + tanh[√(2/π)(z + 0.044715z³)])` is used for efficiency.

### Q3: What is the dying ReLU problem? How do you diagnose and fix it?

**Strong answer:**

> A ReLU neuron "dies" when its pre-activation is always negative. The gradient is 0 for z < 0, so the weights never update, and the neuron is permanently inactive.
>
> **Diagnosis:** During or after training, check what fraction of neurons have zero output for all training examples. A healthy network should have 30-60% sparse activations, but dead neurons are completely inactive for ALL inputs.
>
> **Fixes:**
> - **Leaky ReLU/ELU:** Provide non-zero gradient for negative inputs
> - **Lower learning rate:** Prevent large weight updates that cause neurons to always be negative
> - **He initialization with small positive bias:** Biases initialized to small positive values (e.g., 0.01) help neurons start active
> - **Batch normalization:** Keeps pre-activations near zero, reducing saturation chance

### Q4: Why is the numerical stability trick needed for softmax?

**Strong answer:**

> Without the trick, computing `e^{z_k}` for large z_k causes floating-point overflow. For example, `e^{1000}` is infinity in float32.
>
> The trick subtracts the maximum value: `softmax(z)_k = e^{z_k - max(z)} / Σ_j e^{z_j - max(z)}`. This is mathematically identical because the `e^{-max(z)}` factor cancels in numerator and denominator. After subtraction, the largest exponent is 0 (so max value is 1.0), preventing overflow.
>
> This is also how `log_softmax` is implemented stably: `log_softmax(z)_k = z_k - max(z) - log(Σ_j e^{z_j - max(z)})`.

---

## 14. Code: Implement and Visualize All Activations

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf

# ============================================================
# NumPy Implementations
# ============================================================

class ActivationFunctions:
    """Pure NumPy implementations for educational purposes."""

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_grad(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_grad(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_grad(z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)

    @staticmethod
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))

    @staticmethod
    def elu_grad(z, alpha=1.0):
        return np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))

    @staticmethod
    def gelu(z):
        """GELU using the tanh approximation (as used in GPT-2)."""
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

    @staticmethod
    def gelu_exact(z):
        """Exact GELU using the error function."""
        return z * 0.5 * (1 + erf(z / np.sqrt(2)))

    @staticmethod
    def gelu_grad(z):
        """Approximate gradient of GELU."""
        tanh_arg = np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
        tanh_val = np.tanh(tanh_arg)
        sech2 = 1 - tanh_val**2
        dtanh_dz = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z**2) * sech2
        return 0.5 * (1 + tanh_val) + 0.5 * z * dtanh_dz

    @staticmethod
    def silu(z):
        """SiLU = Swish(1) = z * sigmoid(z)"""
        return z * ActivationFunctions.sigmoid(z)

    @staticmethod
    def silu_grad(z):
        s = ActivationFunctions.sigmoid(z)
        return s + z * s * (1 - s)

    @staticmethod
    def mish(z):
        """Mish = z * tanh(softplus(z))"""
        softplus = np.log1p(np.exp(np.clip(z, -500, 500)))
        return z * np.tanh(softplus)

    @staticmethod
    def softmax(z, axis=-1):
        """Numerically stable softmax."""
        z_shifted = z - np.max(z, axis=axis, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

    @staticmethod
    def softmax_temperature(z, temperature=1.0, axis=-1):
        """Softmax with temperature scaling."""
        return ActivationFunctions.softmax(z / temperature, axis=axis)

    @staticmethod
    def log_softmax(z, axis=-1):
        """Numerically stable log-softmax."""
        z_shifted = z - np.max(z, axis=axis, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(z_shifted), axis=axis, keepdims=True))
        return z_shifted - log_sum_exp


# ============================================================
# Visualization
# ============================================================

def plot_activations_and_gradients(save_path='activations_comparison.png'):
    """Plot all activation functions and their derivatives."""
    af = ActivationFunctions()
    z = np.linspace(-4, 4, 1000)

    activations = {
        'Sigmoid':     (af.sigmoid(z),     af.sigmoid_grad(z)),
        'Tanh':        (af.tanh(z),        af.tanh_grad(z)),
        'ReLU':        (af.relu(z),        af.relu_grad(z)),
        'Leaky ReLU':  (af.leaky_relu(z),  af.leaky_relu_grad(z)),
        'ELU':         (af.elu(z),         af.elu_grad(z)),
        'GELU':        (af.gelu(z),        af.gelu_grad(z)),
        'SiLU/Swish':  (af.silu(z),        af.silu_grad(z)),
        'Mish':        (af.mish(z),        None),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
              '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']

    for idx, (name, (vals, grads)) in enumerate(activations.items()):
        ax = axes[idx]
        ax.plot(z, vals, color=colors[idx], linewidth=2.5, label=f'{name}')
        if grads is not None:
            ax.plot(z, grads, color=colors[idx], linewidth=2, linestyle='--',
                   alpha=0.7, label=f"{name}'")
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 3.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('z', fontsize=11)

    plt.suptitle('Activation Functions and Their Derivatives', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def demonstrate_gradient_saturation():
    """Show how sigmoid kills gradients vs ReLU preserves them."""
    af = ActivationFunctions()
    np.random.seed(42)

    z_values = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

    print("=== Gradient Comparison at Various Pre-Activations ===")
    print(f"{'z':>8} {'sigmoid':>12} {'sigmoid_grad':>15} {'relu_grad':>12}")
    print("-" * 50)
    for z in z_values:
        s = af.sigmoid(z)
        s_grad = af.sigmoid_grad(z)
        r_grad = af.relu_grad(z)
        print(f"{z:>8.1f} {s:>12.4f} {s_grad:>15.6f} {r_grad:>12.1f}")

    # Demonstrate gradient decay through 10 sigmoid layers
    print("\n=== Gradient Decay Through N Sigmoid Layers ===")
    print("(Starting gradient = 1.0, assuming z=0 which gives max sigmoid gradient)")
    grad = 1.0
    for layer in range(1, 11):
        # At z=0, sigmoid_grad = 0.25 (best case!)
        grad *= 0.25
        print(f"  After layer {layer:2d}: gradient = {grad:.2e}")

    print(f"\n  After 10 layers: gradient = {(0.25)**10:.2e}  (effectively zero!)")


# ============================================================
# PyTorch Implementations
# ============================================================

class CustomGELU(nn.Module):
    """Custom GELU implementation matching GPT-2 style."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))


class CustomSiLU(nn.Module):
    """SiLU = x * sigmoid(x), also available as nn.SiLU() in PyTorch >= 1.7"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class CustomMish(nn.Module):
    """Mish = x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def numerical_gradient_check():
    """Verify custom activation gradients using numerical differentiation."""
    print("=== Numerical Gradient Check for Custom Activations ===")
    eps = 1e-5

    activations = {
        'GELU':    CustomGELU(),
        'SiLU':    CustomSiLU(),
        'Mish':    CustomMish(),
        'PyTorch GELU': nn.GELU(),
        'PyTorch SiLU': nn.SiLU(),
    }

    x = torch.linspace(-3, 3, 100).double()

    for name, act in activations.items():
        act = act.double()
        x_req = x.clone().requires_grad_(True)
        out = act(x_req).sum()
        out.backward()
        analytical_grad = x_req.grad.clone()

        # Numerical gradient
        with torch.no_grad():
            numerical_grad = (act(x + eps) - act(x - eps)) / (2 * eps)

        max_diff = (analytical_grad - numerical_grad).abs().max().item()
        print(f"  {name:<20}: max gradient diff = {max_diff:.2e} ({'OK' if max_diff < 1e-4 else 'FAIL'})")


def demonstrate_temperature_softmax():
    """Show effect of temperature on softmax distribution."""
    print("\n=== Temperature Effect on Softmax ===")
    z = np.array([2.0, 1.0, 0.5, -0.5])
    print(f"Logits: {z}")

    af = ActivationFunctions()
    for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
        probs = af.softmax_temperature(z, T)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"  T={T:5.1f}: probs={probs.round(3)}, entropy={entropy:.3f}")


def compare_activations_on_training():
    """Compare how different activations affect training a simple network."""
    import torch.optim as optim
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cpu')
    print("\n=== Activation Function Comparison on Training ===")

    # Simple 5-layer network to stress-test gradient flow
    def make_network(activation):
        layers = []
        sizes = [2, 32, 32, 32, 32, 1]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)

    activations_to_test = {
        'Sigmoid':   nn.Sigmoid,
        'Tanh':      nn.Tanh,
        'ReLU':      nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'ELU':       nn.ELU,
        'GELU':      nn.GELU,
        'SiLU':      nn.SiLU,
    }

    # Spirals dataset (non-linearly separable)
    def make_spirals(n=200):
        theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
        r_a = theta + np.pi
        r_b = -theta - np.pi
        X = np.stack([
            np.concatenate([r_a*np.cos(theta), r_b*np.cos(theta)]),
            np.concatenate([r_a*np.sin(theta), r_b*np.sin(theta)])
        ], axis=1).astype(np.float32)
        y = np.array([0]*n + [1]*n, dtype=np.float32)
        return X / X.std(), y

    X_np, y_np = make_spirals()
    X = torch.tensor(X_np)
    y = torch.tensor(y_np).unsqueeze(1)

    results = {}
    for name, act_class in activations_to_test.items():
        model = make_network(act_class)
        # Initialize with small weights to test activation behavior
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        final_loss = float('inf')
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if torch.isnan(loss):
                print(f"  {name}: NaN loss at epoch {epoch}")
                break
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        preds = (torch.sigmoid(model(X)) > 0.5).float()
        acc = (preds == y).float().mean().item()
        results[name] = (final_loss, acc)
        print(f"  {name:<15}: Final Loss = {final_loss:.4f}, Accuracy = {acc:.3f}")


if __name__ == '__main__':
    print("1. Plotting activations...")
    plot_activations_and_gradients()

    print("\n2. Demonstrating gradient saturation...")
    demonstrate_gradient_saturation()

    print("\n3. Numerical gradient check...")
    numerical_gradient_check()

    print("\n4. Temperature softmax demo...")
    demonstrate_temperature_softmax()

    print("\n5. Training comparison...")
    compare_activations_on_training()
```

---

## Key Formulas Quick Reference

```
Sigmoid:      σ(z) = 1/(1+e^{-z})          σ'(z) = σ(z)(1-σ(z)) ≤ 0.25
Tanh:         tanh(z) = (e^z-e^{-z})/(e^z+e^{-z})  tanh'(z) = 1-tanh²(z) ≤ 1
ReLU:         max(0,z)                       ReLU'(z) = 1 if z>0 else 0
Leaky ReLU:   max(αz, z), α≪1              grad = 1 if z>0 else α
ELU:          z if z>0; α(e^z-1) if z≤0    smooth everywhere
GELU:         z·Φ(z)  ≈ 0.5z(1+tanh[...])  smooth, stochastic interpretation
SiLU:         z·σ(z)                        self-gated
Softmax:      e^{z_k}/Σe^{z_j}             subtract max for numerical stability
```

---

*Next: [Loss Functions](./loss_functions.md) - The objective functions that define what we're learning*
