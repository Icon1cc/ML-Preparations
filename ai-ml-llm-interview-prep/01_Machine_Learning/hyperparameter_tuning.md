# Hyperparameter Tuning

Hyperparameters are the dials and knobs of a machine learning algorithm that must be set *before* training begins (unlike model weights, which are learned *during* training). Tuning them correctly is the difference between a mediocre model and a state-of-the-art model.

---

## 1. Grid Search
The brute-force approach.
*   **Mechanism:** You define a grid of values. 
    `{learning_rate: [0.1, 0.01], max_depth: [3, 5, 7]}`
    The algorithm trains and evaluates a model for every single combination ($2 	imes 3 = 6$ models).
*   **Pros:** It is guaranteed to find the absolute best combination *within the defined grid*.
*   **Cons:** The "Curse of Dimensionality." If you want to tune 5 parameters with 5 values each, you must train $5^5 = 3,125$ models. It is computationally impossible for large datasets or deep learning.

## 2. Random Search
The smarter brute-force approach.
*   **Mechanism:** You define a statistical distribution for each parameter (e.g., a uniform distribution between 0.01 and 0.1). The algorithm randomly selects combinations for a set number of iterations (e.g., 50).
*   **Why it's better than Grid Search:** In Grid Search, if `max_depth` doesn't actually affect the model, you waste 50% of your compute re-testing `max_depth=3` with different learning rates. Random Search tests a unique value for *every* parameter on *every* iteration. It finds optimal zones much faster.

## 3. Bayesian Optimization (The Senior Approach)
Grid and Random search are "dumb"—they do not learn from their past mistakes. Bayesian Optimization is "smart."
*   **Mechanism:** It builds a probabilistic surrogate model (usually a Gaussian Process) of the objective function (the validation loss). 
    1. It tries a few random hyperparameter combinations.
    2. It updates its internal mathematical map of where it thinks the lowest loss is.
    3. It balances **Exploration** (trying completely new areas of the grid) with **Exploitation** (drilling down into areas that have shown low loss so far).
*   **Tools:** `Optuna`, `Hyperopt`, `Scikit-Optimize`.
*   **Use Case:** Mandatory for expensive models like XGBoost or Deep Neural Networks where you can only afford to run 30-50 training cycles.

## 4. Hyperband & Successive Halving
Advanced algorithms designed specifically for Deep Learning.
*   **Mechanism:** Instead of fully training 100 bad models to epoch 50, it starts training 100 models, evaluates them at epoch 5, and aggressively kills off the bottom 50%. It continues training the top 50% to epoch 10, kills half again, until only the best model remains to finish full training.

## 5. What to Tune? (The High ROI Dials)
Do not tune everything. Focus on the parameters that matter.

### XGBoost / LightGBM
1.  **Learning Rate (`eta`):** Start at 0.1. Lower it to 0.01 if you have time to add more trees.
2.  **Number of Trees (`n_estimators`):** Use Early Stopping instead of tuning this directly.
3.  **Tree Complexity (`max_depth`):** Usually 3 to 10. Deeper = more prone to overfit.
4.  **Regularization (`lambda`, `alpha`):** To prevent overfitting.

### Random Forest
1.  **`max_depth`:** The most important to prevent overfitting.
2.  **`max_features`:** Controls the randomness of the forest.

### Neural Networks (Adam Optimizer)
1.  **Learning Rate:** The single most important parameter in all of Deep Learning.
2.  **Batch Size:** Affects gradient noise and training speed.
3.  **Dropout Rate:** To control overfitting.