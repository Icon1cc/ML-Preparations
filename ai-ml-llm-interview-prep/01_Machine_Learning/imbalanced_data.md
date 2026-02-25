# Handling Imbalanced Data

Imbalanced data occurs when the target class of interest is vastly outnumbered by the other class. It is the default state of reality for high-stakes ML: Fraud Detection (0.1% fraud), Disease Diagnosis (1% sick), Ad Clicks (2% click).

---

## 1. The Accuracy Trap
*   **The Trap:** If a dataset has 99% legitimate transactions and 1% fraud, a model that simply hardcodes `return "Legitimate"` for every single input will achieve 99% Accuracy.
*   **The Solution:** Never use Accuracy for imbalanced data.
    *   **Precision:** Out of all the ones we flagged as fraud, how many were actually fraud? (Minimizes False Positives).
    *   **Recall:** Out of all the actual fraud cases in the real world, how many did we catch? (Minimizes False Negatives).
    *   **F1-Score:** The harmonic mean of Precision and Recall.
    *   **PR-AUC (Precision-Recall Area Under Curve):** The best overall metric for highly skewed binary classification.

## 2. Algorithmic Solutions (Cost-Sensitive Learning)
The best way to handle imbalance is often to not change the data at all, but to change how the algorithm learns.

*   **Class Weights:** In libraries like Scikit-Learn or XGBoost, you can set `class_weight='balanced'`. 
*   **Mechanism:** Normally, an algorithm treats a mistake on a fraud case and a mistake on a legit case equally (Error = 1). By adjusting class weights, you tell the loss function: "If you misclassify a fraud case, the penalty is 100x larger than if you misclassify a legit case." The optimizer is forced to pay attention to the minority class.
*   **Pros:** Requires no data manipulation, very fast, highly effective.

## 3. Data-Level Solutions (Resampling)

### A. Under-sampling the Majority Class
*   Randomly delete rows from the 99% majority class until it matches the 1% minority class.
*   **Pros:** Makes training extremely fast.
*   **Cons:** You are literally throwing away 98% of your valid data and information. The model might miss important nuances of the majority class.

### B. Over-sampling the Minority Class
*   Randomly duplicate rows from the 1% minority class until it matches the majority.
*   **Pros:** No information lost.
*   **Cons:** Because you are just copy-pasting the exact same rows, the model tends to severely overfit to those specific examples.

### C. SMOTE (Synthetic Minority Over-sampling Technique)
*   Instead of copying exact rows, SMOTE creates *fake, synthetic* data points.
*   **Mechanism:** It finds a minority point, finds its K-nearest minority neighbors, and draws lines between them in high-dimensional space. It then generates new data points randomly along those lines.
*   **Cons:** It assumes that the space between two minority points is also a valid minority point. If the classes are heavily overlapping (noisy data), SMOTE will generate fake fraud cases deep inside legit territory, ruining the decision boundary.

## 4. Anomaly Detection (The Extreme Imbalance)
If the minority class is less than 0.01% (e.g., predicting when a specific server rack will physically catch fire), standard classification fails entirely.
*   You must switch from Supervised Learning to Unsupervised Anomaly Detection (like **Isolation Forests** or **Autoencoders**). You train the model *only* on the normal data, and flag anything that heavily deviates from that learned normal distribution.

## Interview Strategy
"When dealing with a highly imbalanced dataset like fraud detection, my first step is to discard Accuracy and track **PR-AUC**. I avoid SMOTE and synthetic data manipulation because it can introduce artificial noise. Instead, I rely on **Cost-Sensitive Learning** by passing `scale_pos_weight` directly into an algorithm like XGBoost, forcing the model to heavily penalize false negatives natively during gradient descent."