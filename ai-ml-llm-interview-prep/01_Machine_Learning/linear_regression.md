# Linear Regression

Linear Regression is the foundational algorithm of predictive modeling. While rarely used in complex modern AI systems alone, a deep understanding of its mechanics and assumptions is mandatory for any Data Science interview.

---

## 1. The Core Concept
Linear regression attempts to model the relationship between a continuous target variable ($y$) and one or more independent variables ($X$) by fitting a linear equation to the observed data.

### The Equation (Multiple Linear Regression)
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon$$

*   $y$: Target (Dependent variable)
*   $\beta_0$: Y-intercept
*   $\beta_1 \dots \beta_n$: Coefficients (Weights). How much $y$ changes for a 1-unit increase in $x$.
*   $x_1 \dots x_n$: Features (Independent variables)
*   $\epsilon$: The error term (residuals)

## 2. The Loss Function: OLS
How do we find the "best fitting" line? We use **Ordinary Least Squares (OLS)**. 
We calculate the difference between the actual $y$ values and our predicted $\hat{y}$ values (the residuals), square them, and sum them up. The algorithm finds the specific $\beta$ weights that minimize this sum.

$$MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$
*(Squaring penalizes large errors heavily and ensures negative and positive errors don't cancel each other out).*

## 3. The 4 Strict Assumptions (LINE)
Interviewers love asking about this. If these assumptions are violated, the model's coefficients are untrustworthy.

1.  **L - Linearity:** The relationship between the features and the target must be linear. (Check via scatter plots).
2.  **I - Independence:** The observations must be independent of each other. (No time-series autocorrelation).
3.  **N - Normality of Residuals:** The *errors* ($\epsilon$) should be normally distributed. It does not mean the features $X$ must be normally distributed. (Check via Q-Q plots).
4.  **E - Equal Variance (Homoscedasticity):** The variance of the errors should be constant across all values of $X$. If the errors get wider as predictions get larger (a cone shape on a residual plot), you have Heteroscedasticity. (Fix by log-transforming the target variable).

## 4. Multicollinearity
A massive red flag in Linear Regression. Multicollinearity occurs when two or more independent variables are highly correlated with *each other* (e.g., predicting house price using both `square_footage` and `number_of_rooms`, which move together).

*   **The Problem:** The algorithm cannot figure out which feature is actually responsible for the change in the target. The coefficients ($\beta$) become wildly unstable—a tiny change in data can flip a coefficient from positive to negative. It destroys interpretability.
*   **How to detect:** Correlation Matrix or calculating the VIF (Variance Inflation Factor).
*   **How to fix:** Drop one of the correlated columns, use PCA to combine them, or use L2 (Ridge) Regularization.

## 5. Evaluation Metrics
*   **$R^2$ (Coefficient of Determination):** Ranges from 0 to 1. Represents the proportion of the variance in the target variable that is explained by the model. ($R^2 = 0.8$ means your features explain 80% of the movement in $y$).
*   **Adjusted $R^2$:** Standard $R^2$ artificially goes up every time you add a new feature, even if the feature is garbage. Adjusted $R^2$ penalizes you for adding useless features.
*   **RMSE (Root Mean Squared Error):** Gives the average error in the exact same units as the target variable (e.g., "Predictions are off by an average of $5,000").

## Interview Tip
"Linear regression is highly interpretable, but it is too rigid for complex datasets. If I plot the residuals and see a clear curve (violating the Linearity assumption), my first step would be to apply a non-linear transformation to the features (like taking the log or polynomial features) or switch to a tree-based model."