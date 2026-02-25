# Feature Engineering Fundamentals

Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data that make machine learning algorithms work better. It is often the most time-consuming and impactful part of the ML lifecycle.

---

## 1. Why Feature Engineering?
"Garbage In, Garbage Out." A simple model (like Logistic Regression) with excellent, carefully crafted features will almost always outperform a complex deep neural network fed with raw, noisy data.

## 2. Handling Categorical Variables
Machine learning models (except some tree implementations like LightGBM) require numerical inputs.
*   **One-Hot Encoding (OHE):** Converts a category (e.g., "Color: Red, Blue, Green") into binary columns (`is_red`, `is_blue`, `is_green`).
    *   *Danger:* The "Dummy Variable Trap" (multicollinearity) for linear models. You must drop one column. Also causes the "Curse of Dimensionality" if a feature has thousands of unique values (high cardinality).
*   **Label/Ordinal Encoding:** Converts categories to integers (Small=1, Medium=2, Large=3). *Only use this if there is a natural mathematical order.* Do not use this for unordered categories (like City Names).
*   **Target Encoding:** Replaces a categorical value with the average of the target variable for that category (e.g., replace "New York" with the average house price in New York).
    *   *Danger:* Massive risk of Data Leakage. You MUST calculate target encodings using cross-validation or out-of-fold techniques.

## 3. Handling Numerical Variables
*   **Binning (Discretization):** Converting continuous variables into categories. (e.g., Age 0-18 -> "Child", 19-65 -> "Adult"). Useful when the relationship between the feature and the target is highly non-linear.
*   **Log Transformation:** Taking the logarithm of a feature ($log(x)$). Crucial for handling highly skewed data (like income, prices, or transit delays) to make the distribution more normal, which linear models prefer.
*   **Scaling:** (See Data Preprocessing Patterns file).

## 4. Date and Time Features
Raw timestamps (`2026-02-25 14:30:00`) are useless to models. You must extract cyclic and categorical components:
*   *Categorical:* Day of Week, Month, Is_Weekend, Is_Holiday.
*   *Cyclic Encoding:* Time is a circle (23:59 is 1 minute away from 00:00). If you encode hours as 0-23, the model thinks 23 and 0 are far apart. Encode time using Sine and Cosine transformations to preserve the cyclical nature.

## 5. Domain-Specific Transformations (Logistics Examples)
The best features come from understanding the business.
*   *Instead of:* Raw GPS coordinates (Latitude, Longitude).
*   *Create:* Distance to destination (Haversine formula), Distance to nearest highway, Binary flag for "Is in high traffic zone".
*   *Cross-Features:* Creating new features by combining existing ones. (e.g., `Volume = length * width * height`, or `Density = weight / volume`).

## 6. Feature Selection
Having too many features introduces noise, increases training time, and causes overfitting.
*   **Filter Methods:** Statistical tests (e.g., Chi-Square, ANOVA) to filter out features that have low correlation with the target.
*   **Wrapper Methods:** Recursive Feature Elimination (RFE). Train a model, drop the least important feature, repeat. Very slow but accurate.
*   **Embedded Methods:** L1 Regularization (Lasso) automatically forces the weights of useless features to exactly zero. Tree-based models naturally perform feature selection during splitting.

## Interview Strategy
When asked to design a system (e.g., "Predict ETA"), spend at least 40% of the time discussing Feature Engineering. Explicitly mention creating interaction features, handling dates properly, and the risks of target encoding leakage.