# Data Preprocessing Patterns

Preprocessing is the mechanical step of cleaning and scaling data before it enters a machine learning model. Unlike Feature Engineering (which is about creating *new* information), preprocessing is about standardizing *existing* information.

---

## 1. Handling Missing Data (Imputation)
Real-world data is messy and full of Nulls/NaNs. You cannot pass NaNs to most models (except advanced trees like XGBoost).

*   **Deletion:** Dropping rows with missing values. *Only acceptable if missing data is <5% and is Missing Completely at Random (MCAR).* Dropping data can introduce massive bias.
*   **Simple Imputation:**
    *   *Mean/Median:* Fill with the column average. (Use Median if the data is skewed with outliers).
    *   *Mode:* Fill with the most frequent category for categorical data.
    *   *Constant:* Fill with a sentinel value (e.g., `-999` or `"Unknown"`) to let the model explicitly learn that the data was missing.
*   **Advanced Imputation:**
    *   *KNN Imputation:* Finds the $K$ most similar rows and averages their values to fill the missing spot.
    *   *Iterative Imputation (MICE):* Treats the missing column as a target variable and trains a machine learning model on the other columns to predict the missing value. Highly accurate but computationally expensive.

## 2. Feature Scaling
Models based on distance (KNN, SVM, K-Means) or gradient descent (Neural Networks, Linear/Logistic Regression) are highly sensitive to the scale of the input features. If `Age` ranges from 0-100 and `Salary` ranges from 0-100,000, the model will mistakenly assume `Salary` is 1000x more important.

*   **Standardization (Z-score scaling):** Transforms data to have a mean of 0 and a standard deviation of 1.
    *   $x_{scaled} = \frac{x - \mu}{\sigma}$
    *   *When to use:* The default choice for Logistic Regression, SVMs, and Neural Networks. Does not bound the data to a specific range, so it handles outliers better than Min-Max.
*   **Normalization (Min-Max Scaling):** Squeezes all data into a fixed range, usually [0, 1].
    *   $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
    *   *When to use:* Image processing (pixel values 0-255 scaled to 0-1). *Warning:* Highly sensitive to outliers. If one massive outlier exists, it crushes all normal data into a tiny range.
*   **Robust Scaler:** Uses the Median and the Interquartile Range (IQR) instead of Mean and Standard Deviation.
    *   *When to use:* When your dataset is heavily contaminated with extreme outliers.

### Do Tree-Based Models Need Scaling?
**No.** Decision Trees, Random Forests, and XGBoost split nodes based on the order of values, not their absolute magnitude. Scaling features will have absolutely zero effect on the predictions of a Random Forest.

## 3. Handling Outliers
Outliers can destroy linear models by pulling the line of best fit away from the true trend.
*   **Trimming:** Simply removing rows in the top/bottom 1st percentile.
*   **Winsorization (Capping):** Setting a hard cap. Any value above the 99th percentile is simply overwritten with the 99th percentile value. This keeps the data point but reduces its extreme influence.
*   **Log Transformation:** (See Feature Engineering file).

## 4. The Golden Rule of Preprocessing: Avoid Leakage
**Data Leakage** occurs when information from the test/validation set leaks into the training process.
*   *The Trap:* If you calculate the Mean of the `Salary` column on your *entire dataset* before splitting it into Train/Test, your training data now contains information (the global mean) influenced by the test data.
*   *The Solution:* Always use **Pipelines** (e.g., Scikit-Learn's `Pipeline`). You must `.fit()` the scaler/imputer ONLY on the Training set, and then `.transform()` the Test set using those exact same learned parameters.