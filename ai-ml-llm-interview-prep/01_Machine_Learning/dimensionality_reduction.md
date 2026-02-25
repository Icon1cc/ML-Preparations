# Dimensionality Reduction

As the number of features in a dataset increases, the volume of the mathematical space increases exponentially, making data sparse and causing algorithms to overfit and slow down. This is the "Curse of Dimensionality." Dimensionality reduction solves this by compressing high-dimensional data into a lower-dimensional space while retaining the most important information.

---

## 1. Feature Selection vs. Feature Extraction
*   **Feature Selection:** Keeping the most important original columns and deleting the rest. (e.g., keeping `Income` and `Age`, dropping `Shoe_Size`). The meaning of the features is perfectly preserved.
*   **Feature Extraction (Dimensionality Reduction):** Creating entirely *new* mathematical columns that are combinations of the original columns. (e.g., creating `Component_1` which is a mix of `Income`, `Age`, and `Credit_Score`). You lose human interpretability, but you capture more total variance in fewer columns.

## 2. PCA (Principal Component Analysis)
The undisputed king of linear dimensionality reduction.
*   **The Goal:** Find the axes (directions) in the dataset that maximize variance (spread of the data).
*   **Mechanism:**
    1. Center the data (mean=0).
    2. Calculate the Covariance Matrix of the features.
    3. Calculate the Eigenvectors and Eigenvalues of that matrix.
    4. The Eigenvectors are the new "Principal Components." They are strictly orthogonal (perpendicular/uncorrelated) to each other.
    5. `PC1` captures the absolute maximum amount of variance possible in a single line. `PC2` captures the maximum remaining variance, and so on.
*   **Use Case:** You have 100 features. PCA reveals that the first 10 Principal Components capture 95% of the total variance in the data. You can drop the other 90 components, reducing your dataset size by 90% while keeping 95% of the information.
*   **Trap:** PCA is strictly a *linear* algorithm. If your data lies on a complex curved manifold, PCA will crush and destroy the structural relationships. **Data must be Standard Scaled before PCA.**

## 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)
A highly advanced non-linear technique used almost exclusively for **Data Visualization**.
*   **Mechanism:** It calculates the probability that two points are neighbors in the high-dimensional space, and tries to arrange them in a 2D or 3D space so that those probabilities are preserved. It uses a Student's t-distribution to push dissimilar points further away from each other, creating beautiful, distinct visual clusters.
*   **Use Case:** Taking 768-dimensional Word Embeddings (from BERT) and compressing them to 2D so you can plot them on a graph to physically "see" the semantic clusters (e.g., seeing all the animal words clumped together).
*   **Cons:** It is incredibly computationally expensive. It is non-deterministic (running it twice gives different looking plots). **You cannot use t-SNE as a preprocessing step for a supervised model** because there is no `.transform()` method to apply the exact same mapping to new, unseen test data.

## 4. UMAP (Uniform Manifold Approximation and Projection)
The modern successor to t-SNE.
*   **Pros:** It is significantly faster than t-SNE, scales better to larger datasets, and preserves the *global* structure of the data better (t-SNE tends to destroy global relationships to perfect local clusters).
*   **Use Case:** The current state-of-the-art for visualizing high-dimensional LLM embeddings or genetic data.

## 5. Autoencoders
A deep learning approach to dimensionality reduction.
*   A neural network trained to recreate its own input. It compresses data down into a tiny "bottleneck" hidden layer, and then decompresses it. That tiny bottleneck layer becomes your lower-dimensional representation. Capable of learning highly complex non-linear manifolds.