# Anomaly Detection

Anomaly detection (Outlier detection) is the identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

---

## 1. The Core Challenge: Lack of Labels
In a perfect world, we would frame fraud or failure as a standard Binary Classification problem (Predict 1 for Anomaly, 0 for Normal). 
However, anomalies are so rare that we often have zero labeled examples of them. Therefore, Anomaly Detection is usually framed as an **Unsupervised Learning** problem: "Learn what 'normal' looks like, and flag anything that doesn't fit."

## 2. Statistical / Distance-Based Methods

### Z-Score (The Baseline)
*   Calculate the Mean and Standard Deviation of a feature.
*   If a point is more than 3 standard deviations away from the mean (Z-score > 3 or < -3), flag it.
*   *Flaw:* Assumes data is perfectly normally distributed. Fails on multi-variate anomalies (where `Age=20` is normal, and `Income=$150k` is normal, but the combination of a 20-year-old making $150k is an anomaly).

### KNN (K-Nearest Neighbors) Outlier Detection
*   Calculate the distance from a point to its $K$-th nearest neighbor. If that distance is massive compared to the average distance for other points, it is out in the middle of nowhere (an anomaly).

## 3. Isolation Forest (The Industry Standard)
An ensemble tree-based model specifically designed for anomaly detection.
*   **The Concept:** Anomalies are "few and different." Therefore, they are easier to isolate from the rest of the data.
*   **Mechanism:**
    1. The algorithm randomly selects a feature and randomly selects a split value between the min and max.
    2. It recursively splits the data like a Decision Tree until every single point is isolated in its own leaf node.
    3. **The Magic:** Normal points, packed tightly together in the center of the data mass, require *many* random splits to finally isolate. Anomalies, sitting far away on the edges, will be isolated very quickly in just 1 or 2 splits.
*   **Scoring:** The "Anomaly Score" is simply the path length from the root to the leaf. Shorter path = Higher probability of being an anomaly.
*   **Pros:** Extremely fast, handles highly dimensional data perfectly, handles non-linear distributions.

## 4. One-Class SVM
*   Standard SVM tries to draw a line *between* two classes. One-Class SVM is trained on only one class (Normal data).
*   It tries to draw a boundary (a hyper-sphere) that neatly encompasses all the normal data points. Any new data point that falls outside that boundary is flagged as an anomaly.
*   **Pros:** Very strict boundaries.
*   **Cons:** Computationally heavy, highly sensitive to hyperparameter tuning (`nu` and `gamma`).

## 5. Autoencoders (Deep Learning)
Used for complex data (images, massive time-series).
*   **Mechanism:** Train a neural network to compress the data into a tiny bottleneck, and then reconstruct it back to the original input. Train it *only* on normal data.
*   **Detection:** When you pass an anomaly through the network, the bottleneck won't know how to represent it, and the reconstruction will be garbled. You calculate the **Reconstruction Error** (MSE between input and output). If the error is unusually high, flag it as an anomaly.

## Interview Strategy
"For a production system, I always start with an **Isolation Forest**. It is mathematically elegant, scales to millions of rows, and handles high-dimensional tabular data without the heavy compute requirements of Deep Learning. I would output the anomaly score and set an operational threshold based on the business's tolerance for False Positives."