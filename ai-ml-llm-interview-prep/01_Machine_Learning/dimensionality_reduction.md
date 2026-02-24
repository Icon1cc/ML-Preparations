# Dimensionality Reduction: PCA, t-SNE, UMAP, and Practical Use

## Why reduce dimensions?
High dimensions cause sparse neighborhoods, noisy distances, and slower models (curse of dimensionality).

## PCA
Principal Component Analysis finds orthogonal directions maximizing variance.
Steps:
1. Center data.
2. Compute covariance matrix.
3. Eigendecompose covariance.
4. Project onto top components.

`X_reduced = X_centered * W_k`

### What first PC means
Direction of maximum variance in data. Not necessarily causal importance.

### Use cases
- Compression and de-noising.
- Preprocessing for linear models.
- Visualization (2D/3D) with caution.

## t-SNE
- Non-linear neighborhood-preserving embedding for visualization.
- Great local structure, weak global geometry.
- Non-parametric: cannot directly map new points without retraining.

## UMAP
- Topology-inspired manifold learning.
- Faster and often more structure-preserving than t-SNE.
- Has transform mode for new data (implementation dependent).

## Comparison

| Method | Strength | Weakness | Typical use |
|---|---|---|---|
| PCA | fast, stable, interpretable linear components | linear only | preprocessing + compression |
| t-SNE | excellent local visualization | slow, unstable global layout | exploratory 2D plots |
| UMAP | fast, scalable, often better global/local balance | hyperparameter sensitivity | visualization + embedding reduction |

## When each fails
- PCA fails for highly nonlinear manifolds.
- t-SNE fails for large-scale production transforms.
- UMAP may distort if parameters poorly tuned.

## Autoencoder preview
Neural network learns compressed latent representation and reconstruction.
Useful for nonlinear reduction and anomaly detection.

## Interview questions
1. Why can’t t-SNE be used for production feature transform?
2. What does explained variance ratio represent in PCA?
3. PCA vs UMAP for embedding compression?

## Code: PCA, t-SNE, UMAP
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# X assumed standardized
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)
print('Explained variance (50 PCs):', pca.explained_variance_ratio_.sum())

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_pca)

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)
```

## Practical logistics example
Reducing 1536-d text embeddings from support tickets to 256 dimensions before vector indexing can reduce storage and latency, but validate retrieval recall tradeoff.
