# Linear Algebra for ML

## Why Linear Algebra Matters for ML

**Almost all machine learning is linear algebra at its core:**
- Data is stored in matrices
- Neural networks = matrix multiplications
- Optimization uses gradients (vectors)
- Dimensionality reduction uses eigenvectors
- Even "non-linear" models use linear algebra internally

**You don't need to be a mathematician, but you need to understand the basics.**

---

## Scalars, Vectors, Matrices, and Tensors

### Scalars

**A single number:** `x = 5`

### Vectors

**A 1D array of numbers:**
```
v = [1, 2, 3, 4]
```

**In ML:**
- A single data sample (feature vector)
- Model weights
- Gradients

**Python:**
```python
import numpy as np
v = np.array([1, 2, 3, 4])
print(v.shape)  # (4,)
```

### Matrices

**A 2D array of numbers:**
```
M = [[1, 2, 3],
     [4, 5, 6]]
```

**In ML:**
- Dataset (rows = samples, columns = features)
- Weight matrices in neural networks
- Covariance matrix

**Python:**
```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)  # (2, 3) - 2 rows, 3 columns
```

### Tensors

**Multi-dimensional arrays:**
- 3D: Image (height × width × channels)
- 4D: Batch of images (batch × height × width × channels)

**Python:**
```python
# Batch of 32 RGB images, each 224x224
images = np.random.rand(32, 224, 224, 3)
print(images.shape)  # (32, 224, 224, 3)
```

---

## Essential Matrix Operations

### 1. Matrix Addition

**Element-wise addition:**
```
A + B = [[1, 2],  +  [[5, 6],  =  [[6, 8],
         [3, 4]]      [7, 8]]      [10, 12]]
```

**Requirement:** Same dimensions

**Python:**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B
```

### 2. Scalar Multiplication

**Multiply every element by scalar:**
```
2 × [[1, 2],  =  [[2, 4],
     [3, 4]]      [6, 8]]
```

**Python:**
```python
A = np.array([[1, 2], [3, 4]])
C = 2 * A
```

### 3. Matrix Multiplication (Most Important!)

**Not element-wise!**

**Rule:** (m × n) × (n × p) = (m × p)

**Example:**
```
[[1, 2],     [[5, 6],     [[1×5+2×7, 1×6+2×8],     [[19, 22],
 [3, 4]]  ×   [7, 8]]  =   [3×5+4×7, 3×6+4×8]]  =   [43, 50]]

(2×2)    ×    (2×2)    =         (2×2)
```

**Process:**
- Row of first matrix × Column of second matrix
- Element-wise multiply, then sum

**Python:**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.matmul(A, B)  # or A @ B
```

**Why it matters for ML:**
```python
# Neural network forward pass
X = input_data        # (batch_size, features)
W = weights           # (features, hidden_units)
output = X @ W        # (batch_size, hidden_units)
```

### 4. Transpose

**Flip rows and columns:**
```
A = [[1, 2, 3],      A^T = [[1, 4],
     [4, 5, 6]]             [2, 5],
                            [3, 6]]

(2×3) → (3×2)
```

**Python:**
```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A_T = A.T
print(A_T.shape)  # (3, 2)
```

**Why it matters:**
- Computing covariance matrices
- Backpropagation in neural networks

### 5. Dot Product

**Two vectors:**
```
a · b = [1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32
```

**Python:**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 32
```

**Why it matters:**
- Similarity measure
- Projection
- Neural network computations

---

## Key Concepts for ML

### 1. Identity Matrix (I)

**Diagonal of 1s, rest 0s:**
```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```

**Property:** A × I = A (like multiplying by 1)

**Python:**
```python
I = np.eye(3)  # 3×3 identity matrix
```

### 2. Inverse Matrix (A⁻¹)

**Definition:** A × A⁻¹ = I

**Used in:**
- Solving linear equations
- Computing optimal weights in linear regression: w = (X^T X)⁻¹ X^T y

**Python:**
```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # Approximately identity matrix
```

**Important:** Not all matrices have inverses (singular matrices)

### 3. Norm (Vector Length)

**L2 norm (Euclidean):**
```
||v||₂ = sqrt(v₁² + v₂² + ... + vₙ²)

||[3, 4]||₂ = sqrt(3² + 4²) = sqrt(25) = 5
```

**L1 norm (Manhattan):**
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|

||[3, 4]||₁ = |3| + |4| = 7
```

**Python:**
```python
v = np.array([3, 4])
l2_norm = np.linalg.norm(v, 2)  # 5.0
l1_norm = np.linalg.norm(v, 1)  # 7.0
```

**Why it matters:**
- Regularization (L1, L2)
- Distance metrics
- Gradient clipping

### 4. Eigenvalues and Eigenvectors

**Definition:**
```
A × v = λ × v

where:
- v is eigenvector
- λ is eigenvalue
```

**Intuition:** Eigenvectors are directions that don't change when transformed by matrix A, only scaled by λ

**Why it matters:**
- **PCA (Principal Component Analysis):** Eigenvectors of covariance matrix are principal components
- Understanding matrix transformations
- Stability analysis

**Python:**
```python
A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [5., 2.]
print(eigenvectors)  # Corresponding eigenvectors
```

### 5. Matrix Rank

**Number of linearly independent rows/columns**

**Full rank:** All rows/columns are independent
**Rank deficient:** Some rows/columns are linear combinations of others

**Why it matters:**
- Feature redundancy detection
- Invertibility (only full-rank square matrices are invertible)

**Python:**
```python
A = np.array([[1, 2], [2, 4]])  # Row 2 = 2 × Row 1
rank = np.linalg.matrix_rank(A)  # 1 (not full rank)
```

---

## ML Applications

### 1. Linear Regression

**Goal:** Find weights w such that y = X × w

**Solution (Normal Equation):**
```
w = (X^T × X)⁻¹ × X^T × y
```

**Python:**
```python
# X: (n_samples, n_features)
# y: (n_samples,)
w = np.linalg.inv(X.T @ X) @ X.T @ y
predictions = X @ w
```

### 2. Neural Network Forward Pass

**Single layer:**
```
output = activation(X × W + b)

where:
- X: input (batch_size, input_dim)
- W: weights (input_dim, output_dim)
- b: bias (output_dim,)
```

**Python:**
```python
X = np.random.rand(32, 10)    # 32 samples, 10 features
W = np.random.rand(10, 5)     # 10 input, 5 hidden units
b = np.random.rand(5)         # 5 biases

hidden = X @ W + b            # (32, 5)
output = np.maximum(0, hidden) # ReLU activation
```

### 3. PCA (Principal Component Analysis)

**Steps:**
1. Center data: X_centered = X - mean(X)
2. Compute covariance: C = (X_centered^T × X_centered) / n
3. Find eigenvectors of C
4. Project data: X_pca = X_centered × eigenvectors

**Python:**
```python
from sklearn.decomposition import PCA

# Manually
X_centered = X - X.mean(axis=0)
cov_matrix = (X_centered.T @ X_centered) / len(X)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Top 2 components
X_pca = X_centered @ eigenvectors[:, :2]

# Or use sklearn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

### 4. Cosine Similarity

**Measure similarity between vectors:**
```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

Range: [-1, 1]
- 1: Identical direction
- 0: Orthogonal
- -1: Opposite direction
```

**Python:**
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Or
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([a], [b])[0, 0]
```

**Why it matters:**
- Text similarity (word embeddings)
- Recommendation systems
- RAG retrieval

### 5. Gradient Descent

**Update rule:**
```
w_new = w_old - learning_rate × gradient

where gradient = ∇L (vector of partial derivatives)
```

**Python:**
```python
# Simple gradient descent
learning_rate = 0.01
for epoch in range(1000):
    predictions = X @ w
    error = predictions - y
    gradient = X.T @ error / len(X)
    w = w - learning_rate * gradient
```

---

## Common Operations in Code

### Broadcasting

**NumPy automatically expands dimensions:**
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # (2, 3)
b = np.array([10, 20, 30])  # (3,)

C = A + b  # b is broadcast to (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Reshaping

```python
X = np.array([1, 2, 3, 4, 5, 6])  # (6,)

X_2d = X.reshape(2, 3)  # (2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

X_3d = X.reshape(2, 3, 1)  # (2, 3, 1)

# Flatten back
X_flat = X_2d.flatten()  # (6,)
```

### Axis Operations

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])

# Sum along axis 0 (down rows)
col_sums = X.sum(axis=0)  # [5, 7, 9]

# Sum along axis 1 (across columns)
row_sums = X.sum(axis=1)  # [6, 15]

# Mean, max, min work similarly
col_means = X.mean(axis=0)
```

---

## Interview Questions

### Q1: "What's the difference between element-wise and matrix multiplication?"

**Answer:**
```
Element-wise (Hadamard product):
[[1, 2],  ⊙  [[5, 6],  =  [[1×5, 2×6],  =  [[5, 12],
 [3, 4]]      [7, 8]]      [3×7, 4×8]]      [21, 32]]

Matrix multiplication:
[[1, 2],  ×  [[5, 6],  =  [[1×5+2×7, 1×6+2×8],  =  [[19, 22],
 [3, 4]]      [7, 8]]      [3×5+4×7, 3×6+4×8]]      [43, 50]]

Element-wise: Same position, multiply
Matrix: Row × Column, sum
```

### Q2: "How is linear algebra used in neural networks?"

**Answer:**
"Neural networks are essentially stacked matrix multiplications with non-linearities:

1. **Forward pass:** output = activation(X @ W + b)
2. **Backpropagation:** Compute gradients using chain rule (matrix calculus)
3. **Weight updates:** W = W - lr × gradient

Each layer transforms data via matrix multiplication, allowing the network to learn complex representations."

### Q3: "What is the transpose used for in ML?"

**Answer:**
"Common uses:

1. **Computing gradients:** ∇w = X^T @ error
2. **Covariance matrix:** cov(X) = X^T @ X
3. **Changing shapes:** (m, n) → (n, m) for matrix multiplication
4. **Inner product:** a^T @ b (dot product)

Example in linear regression:
```python
# Normal equation: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X.T @ X) @ X.T @ y
```"

### Q4: "Why can't some matrices be inverted?"

**Answer:**
"A matrix is singular (non-invertible) if:

1. **Not full rank:** Rows/columns are linearly dependent
2. **Determinant = 0:** No unique solution

Example:
```python
A = [[1, 2],
     [2, 4]]  # Row 2 = 2 × Row 1

# Rank = 1 (not full rank)
# Cannot invert
```

**ML implications:**
- X^T X might be singular if features are perfectly correlated
- Solution: Add regularization (Ridge regression) or remove correlated features"

### Q5: "Explain eigenvalues/eigenvectors simply"

**Answer:**
"Eigenvectors are special directions that don't change when a matrix transforms them, only stretch/shrink:

```
A @ v = λ @ v

where:
- v: eigenvector (direction)
- λ: eigenvalue (scaling factor)
```

**Intuition:** If you apply transformation A to eigenvector v, it just scales v by λ.

**ML application:** In PCA, eigenvectors of covariance matrix point in directions of maximum variance. We keep top eigenvectors to reduce dimensions while preserving information."

---

## Key Takeaways

1. **Matrix multiplication is core to ML** - Neural networks, linear models, transformers
2. **Transpose is everywhere** - Gradients, covariance, reshaping
3. **Eigenvalues/vectors power PCA** - Dimensionality reduction
4. **Norms measure size** - Regularization, distance metrics
5. **Broadcasting simplifies code** - No need for explicit loops
6. **Shape matters** - Always check dimensions for matrix operations

---

## Practice Problems

**1. Matrix shapes:**
```python
A: (10, 5)
B: (5, 3)
C = A @ B  # What shape?
```
<details>
<summary>Answer</summary>
(10, 3) - outer dimensions
</details>

**2. Why does this fail?**
```python
A = np.array([[1, 2]])     # (1, 2)
B = np.array([[3], [4]])   # (2, 1)
C = B @ A  # Works
D = A @ B  # Also works
E = A @ A  # Fails!
```
<details>
<summary>Answer</summary>
A @ A tries (1, 2) @ (1, 2) - middle dimensions don't match (2 ≠ 1)
</details>

**3. Compute by hand:**
```
[[1, 2],  @  [[1],  = ?
 [3, 4]]      [2]]
```
<details>
<summary>Answer</summary>
[[1×1 + 2×2],  =  [[5],
 [3×1 + 4×2]]      [11]]
</details>

---

**Next:** [Probability Theory](./probability_theory.md) | **Back:** [README](./README.md)
