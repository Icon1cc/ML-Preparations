# Cross-Validation Strategies

Evaluating a model on the exact same data it was trained on is the ultimate sin of machine learning. It guarantees an artificially high score due to overfitting. To know how a model will perform in the real world, we must use Cross-Validation.

---

## 1. The Basic Train-Test Split
*   **Mechanism:** Shuffle the dataset, take 80% for training and 20% for testing.
*   **The Flaw:** By random chance, your 20% test set might be unusually easy or unusually hard. You might tune your hyperparameters to do perfectly on this specific 20% split, which is still a form of indirect data leakage.

## 2. K-Fold Cross-Validation
The industry standard for general tabular data.
*   **Mechanism:** 
    1. Divide the dataset into $K$ equal-sized folds (usually $K=5$ or $K=10$).
    2. Train the model $K$ times. Each time, use $K-1$ folds for training and the remaining 1 fold for validation.
    3. The final performance score is the average of the $K$ validation scores.
*   **Why it's better:** Every single data point gets to be in the test set exactly once. It provides a highly robust estimate of the model's true generalization error.

## 3. Stratified K-Fold
Mandatory for **Classification** problems with imbalanced classes.
*   **The Problem:** If you are detecting fraud (1% of data) and use standard K-Fold, one of your folds might randomly contain zero fraud cases. The model will fail to evaluate properly.
*   **The Solution:** Stratified K-Fold guarantees that the ratio of classes (e.g., 99% legit, 1% fraud) is preserved exactly in *every single fold*. 

## 4. Time-Series Cross-Validation (Out-of-Time Validation)
**CRITICAL INTERVIEW TOPIC:** Never use standard K-Fold on time-series data!
*   **The Problem:** Standard K-Fold shuffles the data. If you are predicting tomorrow's stock price, K-Fold might put tomorrow's data in the training set and yesterday's data in the test set. The model will "see into the future," resulting in massive data leakage.
*   **The Solution (Sliding Window):** 
    *   Fold 1: Train on Jan, Test on Feb.
    *   Fold 2: Train on Jan+Feb, Test on Mar.
    *   Fold 3: Train on Jan+Feb+Mar, Test on Apr.
*   This strictly enforces that the model is only ever evaluated on data that occurred *chronologically after* its training data.

## 5. Group K-Fold
Mandatory when you have multiple records from the same entity.
*   **The Problem:** You have 10,000 medical scans from 500 patients (20 scans per patient). If you use standard K-Fold, Patient A's scan #1 might end up in the training set, and Patient A's scan #2 in the test set. The model will memorize Patient A's unique bone structure, not the actual disease.
*   **The Solution:** Group K-Fold ensures that all 20 scans for Patient A are kept strictly together—they are either *all* in the training set, or *all* in the test set.

## Interview Strategy
"When approaching model evaluation, I never rely on a single train-test split. My default is **Stratified 5-Fold Cross-Validation** to ensure stable metrics across different data distributions. However, because logistics problems—like demand forecasting—are inherently chronological, I would strictly implement a **Time-Series Expanding Window** split to prevent future data leakage."