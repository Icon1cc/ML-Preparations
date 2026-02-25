# Loss Functions

The Loss Function (or Cost Function / Objective Function) is the mathematical compass of a machine learning model. It quantifies how "wrong" the model's predictions are compared to the true labels. The optimizer's sole job is to minimize this number.

---

## 1. Regression Loss Functions (Predicting Continuous Values)

### A. MSE (Mean Squared Error) / L2 Loss
*   **Formula:** $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$
*   **Characteristics:** Squaring the error heavily penalizes large errors. If your prediction is off by 10, the penalty is 100.
*   **When to use:** The default for regression. Use when you want to strongly penalize massive outliers.
*   **Drawback:** Highly sensitive to anomalies. A single bad data point can heavily skew the entire model.

### B. MAE (Mean Absolute Error) / L1 Loss
*   **Formula:** $\frac{1}{n} \sum |y_i - \hat{y}_i|$
*   **Characteristics:** Treats all errors linearly. An error of 10 incurs a penalty of 10.
*   **When to use:** When your dataset has outliers that you want the model to ignore (e.g., predicting median house prices).
*   **Drawback:** The gradient is constant everywhere (even very close to the minimum), which can cause the optimizer to bounce around the minimum instead of settling.

### C. Huber Loss
*   **Characteristics:** The best of both worlds. It acts like MSE when the error is small (providing stable gradients near the minimum) and acts like MAE when the error is large (making it robust to outliers).
*   **When to use:** Production forecasting systems (like ETA prediction or demand forecasting) where extreme outliers exist but shouldn't break the model.

## 2. Classification Loss Functions (Predicting Categories)

### A. Binary Cross-Entropy (BCE) / Log Loss
*   **Formula:** $- \frac{1}{n} \sum [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$
*   **When to use:** Binary classification (e.g., Spam vs. Not Spam).
*   **Intuition:** It penalizes *confident wrong answers* astronomically. If the true label is 1, and the model predicts 0.99, the loss is near zero. If the model predicts 0.01, the loss shoots toward infinity.

### B. Categorical Cross-Entropy (CCE)
*   **When to use:** Multi-class classification where each sample belongs to exactly ONE class (e.g., classifying an image as a Cat, Dog, or Bird).
*   **Requirement:** The final layer of the network must use a **Softmax** activation function to output a valid probability distribution.

### C. Focal Loss
*   **Characteristics:** A modification of Cross-Entropy designed to handle **severe class imbalance** (e.g., object detection, where 99% of the image is "background" and 1% is the "object").
*   **How it works:** It mathematically down-weights the loss assigned to easy-to-classify examples (the background) and forces the model to focus heavily on hard-to-classify examples.

## 3. Generative AI Loss Functions (LLMs)

### Causal Language Modeling Loss (Next-Token Prediction)
*   At its core, LLM pre-training uses standard **Categorical Cross-Entropy**.
*   If the vocabulary size is 50,000 words, the problem is framed as a massive 50,000-class classification problem. Given the context "The cat sat on the", the model outputs a probability distribution across all 50,000 words. The true label "mat" is converted to a one-hot vector. Cross-Entropy calculates the loss.

## Interview Tip: "Softmax vs. Sigmoid with Cross-Entropy"
*   **Multi-Class (Mutually Exclusive):** Is this image a Cat, Dog, OR Bird? Use **Softmax** + **Categorical Cross Entropy**.
*   **Multi-Label (Non-Exclusive):** Does this movie contain Action, Comedy, AND/OR Romance? Use multiple **Sigmoid** outputs + **Binary Cross Entropy** calculated independently for each label.