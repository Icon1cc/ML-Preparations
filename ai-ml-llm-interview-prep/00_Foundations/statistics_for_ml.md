# Statistics for Machine Learning

While probability studies theoretical models, statistics deals with analyzing empirical data to estimate the parameters of those models and test hypotheses.

---

## 1. Descriptive Statistics
Summarizing and describing the features of a dataset.

### Measures of Central Tendency
*   **Mean:** The mathematical average. Highly sensitive to outliers.
*   **Median:** The middle value when sorted. Robust to outliers (Use this for salary or house prices).
*   **Mode:** The most frequent value.

### Measures of Dispersion (Spread)
*   **Variance:** Average squared deviation from the mean.
*   **Standard Deviation:** The square root of variance. It is in the same units as the original data, making it easier to interpret.
*   **IQR (Interquartile Range):** The difference between the 75th percentile and the 25th percentile. Used to detect outliers (e.g., values $> Q3 + 1.5 	imes IQR$).

## 2. Inferential Statistics
Drawing conclusions about a population based on a sample.

### Hypothesis Testing (The Core of A/B Testing)
Used to determine if a result is statistically significant or just due to random chance.
1.  **Null Hypothesis ($H_0$):** The default assumption that there is no effect or no difference. (e.g., "The new website UI does not increase conversion rates.")
2.  **Alternative Hypothesis ($H_1$):** What you are trying to prove. (e.g., "The new UI increases conversion rates.")
3.  **p-value:** The probability of observing the data (or something more extreme) assuming the Null Hypothesis is true.
4.  **Alpha ($\alpha$):** The significance level (usually 0.05). If $p < \alpha$, we reject the Null Hypothesis.

### Type I and Type II Errors
*   **Type I Error (False Positive):** Rejecting the null hypothesis when it is actually true. (Saying the new UI works when it actually doesn't). Alpha ($\alpha$) is the probability of a Type I error.
*   **Type II Error (False Negative):** Failing to reject the null hypothesis when it is actually false. (Missing a real effect). Beta ($\beta$) is the probability of a Type II error. Power is $1 - \beta$.

## 3. Correlation and Covariance

### Covariance
Measures the directional relationship between two variables.
*   Positive: As X goes up, Y goes up.
*   Negative: As X goes up, Y goes down.
*   *Problem:* The magnitude is unbounded and depends on the units of X and Y, making it hard to interpret "how strong" the relationship is.

### Pearson Correlation Coefficient ($r$)
A normalized version of covariance that is strictly bounded between -1 and 1.
*   $1$: Perfect positive linear relationship.
*   $0$: No linear relationship.
*   $-1$: Perfect negative linear relationship.
*   *Trap:* Pearson only measures *linear* relationships. Two variables can have a correlation of 0 but have a perfect quadratic relationship ($y = x^2$).

## 4. The Curse of Dimensionality
As the number of features (dimensions) increases, the volume of the space increases so fast that the available data becomes sparse.
*   **Impact:** In high dimensions, all points become almost equidistant from each other. Distance-based algorithms (like KNN or K-Means) lose their meaning and perform terribly.
*   **Solution:** Dimensionality reduction techniques (PCA, t-SNE) or feature selection.

## Interview Trap: "Correlation does not imply Causation"
If an interviewer presents a scenario where ice cream sales and shark attacks are highly correlated, do not build a model predicting shark attacks from ice cream sales. Point out the **Confounding Variable** (Temperature/Summer). Machine learning models generally learn correlation, not causal inference, unless specifically designed to do so (like A/B testing).