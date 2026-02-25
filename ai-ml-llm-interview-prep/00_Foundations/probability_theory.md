# Probability Theory for Machine Learning

Probability is the mathematical language for quantifying uncertainty. Machine learning is fundamentally about making predictions under uncertainty based on data.

---

## 1. Core Concepts

### Random Variables
A variable whose values depend on outcomes of a random phenomenon.
*   **Discrete:** Takes on distinct, separate values (e.g., a coin flip, number of packages shipped today). PMF (Probability Mass Function).
*   **Continuous:** Takes on any value within a range (e.g., the exact transit time of a truck). PDF (Probability Density Function).

### Joint, Marginal, and Conditional Probability
*   **Joint Probability $P(A, B)$:** The probability of events A and B happening at the same time.
*   **Marginal Probability $P(A)$:** The probability of an event A happening, regardless of other events. $P(A) = \sum_B P(A, B)$.
*   **Conditional Probability $P(A|B)$:** The probability of event A happening, given that event B has already happened. $P(A|B) = \frac{P(A, B)}{P(B)}$.

## 2. Bayes' Theorem
The most important theorem in ML. It describes how to update the probabilities of hypotheses when given evidence.

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

*   **$P(H)$ (Prior):** Initial degree of belief in hypothesis $H$.
*   **$P(E|H)$ (Likelihood):** The probability of observing the evidence $E$ given that hypothesis $H$ is true.
*   **$P(E)$ (Evidence/Marginal):** The total probability of observing evidence $E$ under all possible hypotheses.
*   **$P(H|E)$ (Posterior):** The updated probability of the hypothesis given the evidence.

**ML Context (Naive Bayes Classifier):** Given features (Evidence), what is the probability of a specific class (Hypothesis)?

## 3. Common Probability Distributions

### A. Bernoulli and Binomial
*   **Bernoulli:** A single trial with two possible outcomes (Success/Failure). e.g., Will this package be delayed? (Yes/No).
*   **Binomial:** The number of successes in $n$ independent Bernoulli trials. e.g., Out of 100 packages, how many will be delayed?

### B. Poisson
Models the probability of a given number of events occurring in a fixed interval of time or space.
*   **Use Case:** Predicting the number of customer support calls DHL receives per hour, or the number of trucks arriving at a warehouse dock in a day.

### C. Normal (Gaussian) Distribution
The "Bell Curve." It appears everywhere in nature due to the **Central Limit Theorem**.
*   **Formula Parameters:** Mean ($\mu$) controls the center, Variance ($\sigma^2$) controls the spread.
*   **Central Limit Theorem:** The sum of a large number of independent random variables will be approximately normally distributed, regardless of the underlying distribution. This is why we assume errors (residuals) in Linear Regression are normally distributed.

## 4. Expected Value and Variance
*   **Expected Value $E[X]$:** The long-run average value of a random variable. The mean.
*   **Variance $Var(X)$:** How much the values of the random variable spread out from the expected value. $Var(X) = E[(X - \mu)^2]$.
*   **Covariance:** A measure of the joint variability of two random variables. If greater than 0, they move in the same direction.

## Interview Tip: "Maximum Likelihood Estimation (MLE)"
Interviewers love asking about MLE.
*   **Concept:** Given a dataset, what are the parameters of the model that make this observed data the most probable?
*   **Example:** If you flip a coin 10 times and get 7 heads, the MLE for the probability of heads is $0.7$. In ML, when we minimize Mean Squared Error or Cross-Entropy Loss, we are actually performing Maximum Likelihood Estimation mathematically.