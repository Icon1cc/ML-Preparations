# Decision Trees

Decision Trees are non-parametric algorithms used for both classification and regression tasks. They are intuitive, highly interpretable, and serve as the foundational building blocks for powerful ensemble methods like Random Forests and XGBoost.

---

## 1. The Intuition
A decision tree breaks down a dataset into smaller and smaller subsets based on a series of Yes/No questions about the features.
*   **Root Node:** The top of the tree, represents the entire dataset.
*   **Internal Nodes:** A point where the data is split based on a feature condition (e.g., `Age > 30`).
*   **Leaf Nodes:** The final end points. In classification, they output a class label (e.g., "Survived"). In regression, they output a continuous value (usually the mean of the samples in that leaf).

## 2. How the Tree Splits (The Math)
The algorithm uses a greedy, top-down approach. At each node, it evaluates every possible split for every feature and chooses the one that creates the "purest" child nodes.

### Classification: Impurity Metrics
1.  **Gini Impurity (Default in Scikit-Learn):**
    Measures the probability that a randomly chosen element would be incorrectly classified if it were randomly labeled according to the distribution in the node.
    $$Gini = 1 - \sum (p_i)^2$$
    *Where $p_i$ is the probability of an item belonging to class $i$. A pure node (all one class) has a Gini of 0.*
2.  **Entropy & Information Gain:**
    Entropy measures the amount of disorder or uncertainty in a node.
    $$Entropy = -\sum p_i \log_2(p_i)$$
    The tree chooses the split that maximizes **Information Gain** (the reduction in Entropy from parent to children).

### Regression: Variance Reduction
For regression tasks (CART algorithm), the tree splits to minimize the Mean Squared Error (MSE) within the child nodes. It calculates the variance of the target variable in the parent, and chooses the split that results in the lowest weighted variance in the children.

## 3. Advantages and Disadvantages

### Pros:
*   **Interpretability:** You can literally print the tree and follow the logic with your finger. Easily understood by non-technical stakeholders.
*   **No Preprocessing:** They do not require feature scaling (standardization/normalization). They handle non-linear relationships naturally. They ignore irrelevant features.
*   **Handles Mixed Data:** They can handle both numerical and categorical data natively.

### Cons (The Big Problem): High Variance
*   **Severe Overfitting:** A decision tree left to its own devices will grow until every leaf is perfectly pure (containing 1 sample). It will perfectly memorize the training data and fail catastrophically on test data.
*   **Instability:** Changing a single row in the training data can result in a completely different tree structure.

## 4. Hyperparameter Tuning (Preventing Overfitting)
Because trees overfit so easily, you *must* constrain their growth.
*   `max_depth`: Limits how deep the tree can grow. The most important parameter.
*   `min_samples_split`: The minimum number of samples required to split an internal node. (e.g., don't split if the node only has 5 samples left).
*   `min_samples_leaf`: The minimum number of samples required to be at a leaf node. (Prevents leaves with a single sample).
*   `max_features`: The maximum number of features to consider when looking for the best split.

## Interview Strategy
*   Never propose a single Decision Tree as your final production model; it is too unstable.
*   Propose it as an **Exploratory Data Analysis (EDA)** tool or a **Baseline Model**. You can train a shallow tree and plot it to instantly see which 2 or 3 features are the most dominant splitters in your dataset, gaining massive intuition before building a more complex model.