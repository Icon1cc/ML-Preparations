# Logistic Regression

Despite the name, Logistic Regression is a **classification** algorithm, not a regression algorithm. It is the fundamental building block for binary classification and the mathematical foundation for a single neuron in a neural network.

---

## 1. The Core Concept
Linear Regression predicts a continuous value (from $-\infty$ to $\infty$). For binary classification (e.g., Spam vs. Not Spam), we need a value bounded strictly between 0 and 1 to represent a probability.

Logistic regression achieves this by taking the output of a linear equation and passing it through the **Sigmoid (Logistic) Function**.

### The Math
1.  **Linear Combination:** $z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
2.  **Sigmoid Activation:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
3.  **Output:** The result is $\hat{y}$, which represents the probability that the instance belongs to the positive class: $P(Y=1|X)$.

### Decision Boundary
To make a final classification (0 or 1), we apply a threshold to the probability.
*   If $P(Y=1|X) \ge 0.5$, predict Class 1.
*   If $P(Y=1|X) < 0.5$, predict Class 0.
*   *Note:* The threshold doesn't have to be 0.5. In fraud detection, where catching fraud is more important than false alarms, you might lower the threshold to 0.1.

## 2. Loss Function: Log Loss (Cross-Entropy)
We cannot use Mean Squared Error (MSE) for Logistic Regression because the sigmoid function creates a non-convex (wavy) error surface with many local minima.

Instead, we use **Log Loss** (Binary Cross-Entropy):
$$L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

*   If the true class $y=1$ and the model predicts $\hat{y}=0.99$, the loss is near 0.
*   If the true class $y=1$ but the model predicts $\hat{y}=0.01$, the loss approaches infinity. It heavily penalizes confident but wrong predictions.
*   Log Loss guarantees a strictly convex error surface, meaning Gradient Descent will always find the global minimum.

## 3. Assumptions of Logistic Regression
It is a linear model, so it carries strict assumptions:
1.  **Linearity of Independent Variables and Log-Odds:** It assumes a linear relationship between the input features and the log-odds of the target. It cannot naturally capture complex, non-linear patterns (e.g., an XOR function) without manual feature engineering (like polynomial features).
2.  **No Multicollinearity:** Features should not be highly correlated with each other. If they are, the coefficients ($\beta$) become unstable and uninterpretable.
3.  **Independence of Observations:** Rows of data should not be dependent on each other (e.g., time-series data violates this).

## 4. Interpretability (Odds Ratio)
Logistic Regression is highly prized in industries like medicine and finance because it is perfectly interpretable.
*   The coefficients ($\beta$) represent the change in the **log-odds** of the target for a 1-unit increase in the predictor.
*   If you exponentiate the coefficient ($e^{\beta_1}$), you get the **Odds Ratio**.
*   *Example:* If predicting lung cancer, and the $e^{\beta}$ for the feature `smoker_flag` is $2.5$, you can explain to a stakeholder: "Being a smoker increases the odds of lung cancer by a factor of 2.5, holding all other variables constant."

## 5. Regularization
Logistic Regression is prone to overfitting if you have many features or highly correlated features.
*   **L1 (Lasso) Regularization:** Adds absolute value of coefficients to the loss. Forces less important features to exactly 0 (performing feature selection).
*   **L2 (Ridge) Regularization:** Adds squared value of coefficients to the loss. Shrinks all coefficients toward zero, preventing any single feature from dominating. (This is usually the default in Scikit-Learn).

## Interview Summary
*   **When to use:** Baseline model for binary classification, when interpretability is paramount, or when you have a small dataset where complex models would overfit.
*   **When to avoid:** When the decision boundary is highly non-linear, or when you have raw image/text data.