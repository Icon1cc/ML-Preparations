# 01 Machine Learning

## Overview

This section covers **classical machine learning algorithms** - the foundation of data science and a critical part of every ML interview. Even in the LLM era, these algorithms remain essential for many production systems.

## 📚 Contents

### Supervised Learning - Regression
1. [Linear Regression](./linear_regression.md)
2. [Regularization (Ridge, Lasso, ElasticNet)](./regularization.md)

### Supervised Learning - Classification
3. [Logistic Regression](./logistic_regression.md)
4. [Decision Trees](./decision_trees.md)
5. [Random Forest](./random_forest.md)
6. [Gradient Boosting](./gradient_boosting.md) - XGBoost, LightGBM, CatBoost
7. [Support Vector Machines (SVM)](./support_vector_machines.md)

### Unsupervised Learning
8. [K-Means and Clustering](./clustering.md)
9. [Dimensionality Reduction](./dimensionality_reduction.md) - PCA, t-SNE, UMAP
10. [Anomaly Detection](./anomaly_detection.md)

### Time Series
11. [Time Series Basics](./time_series_basics.md)

### Practical Topics
12. [Scikit-Learn Workflows](./sklearn_workflows.md)
13. [Cross-Validation Strategies](./cross_validation.md)
14. [Hyperparameter Tuning](./hyperparameter_tuning.md)
15. [Handling Imbalanced Data](./imbalanced_data.md)
16. [Model Interpretability](./model_interpretability.md) - SHAP, LIME

## 🎯 Learning Objectives

After this section, you should be able to:

- Explain how each algorithm works (intuition + math)
- Choose the right algorithm for a given problem
- Implement and tune models using scikit-learn
- Interpret model predictions and debug issues
- Handle real-world challenges (imbalanced data, missing values, etc.)
- Answer interview questions confidently

## ⏱️ Time Estimate

**Total: 12-15 hours**

**High Priority (Interview Critical):**
- Decision Trees + Random Forest: 2 hours ⭐⭐⭐
- Gradient Boosting: 2 hours ⭐⭐⭐
- Logistic Regression: 1.5 hours ⭐⭐⭐
- Regularization: 1 hour ⭐⭐
- Model Interpretability: 1 hour ⭐⭐

**Medium Priority:**
- Linear Regression: 1 hour
- SVM: 1 hour
- Cross-Validation: 1 hour
- Hyperparameter Tuning: 1 hour

**Lower Priority (but useful):**
- Clustering: 1 hour
- Dimensionality Reduction: 1 hour
- Time Series: 1 hour

## 🔥 Interview Focus Areas

### Most Common Interview Questions

1. **"Explain Random Forest vs Gradient Boosting"**
   - See [Random Forest](./random_forest.md) and [Gradient Boosting](./gradient_boosting.md)

2. **"How do you handle imbalanced data?"**
   - See [Imbalanced Data](./imbalanced_data.md)

3. **"Walk me through your machine learning pipeline"**
   - See [Scikit-Learn Workflows](./sklearn_workflows.md)

4. **"How do you prevent overfitting?"**
   - See [Regularization](./regularization.md), [Cross-Validation](./cross_validation.md)

5. **"Explain L1 vs L2 regularization"**
   - See [Regularization](./regularization.md)

6. **"How do you interpret a black-box model?"**
   - See [Model Interpretability](./model_interpretability.md)

### Algorithm Selection Guide (Quick Reference)

| Problem Type | Best Algorithms | Why? |
|--------------|----------------|------|
| **Tabular data, interpretability needed** | Logistic Regression, Decision Trees | Simple, interpretable |
| **Tabular data, max performance** | XGBoost, LightGBM, CatBoost | SOTA for structured data |
| **High-dimensional data** | Logistic + L1, SVM | Handle many features |
| **Need probabilities** | Logistic Regression, Calibrated RF/XGBoost | Direct probability estimates |
| **Non-linear boundaries** | SVM (RBF kernel), Random Forest, XGBoost | Capture complex patterns |
| **Imbalanced data** | XGBoost/LightGBM + class weights, Logistic + SMOTE | Handle class imbalance |
| **Categorical features** | CatBoost, LightGBM | Native categorical support |
| **Fast training needed** | Logistic Regression, Decision Tree | Quick to train |
| **Fast prediction needed** | Logistic Regression, Small Decision Tree | Low latency inference |
| **Small dataset** | Logistic Regression, SVM | Less prone to overfitting |
| **Very large dataset** | LightGBM, SGD-based methods | Scales well |

## 🚀 Quick Start Paths

### Path 1: I Need the Fastest ROI (4-5 hours)

**Goal:** Cover the most interview-relevant topics

1. [Gradient Boosting](./gradient_boosting.md) - 2 hours ⚡ **START HERE**
2. [Random Forest](./random_forest.md) - 1.5 hours
3. [Regularization](./regularization.md) - 1 hour
4. [Model Interpretability](./model_interpretability.md) - 1 hour

### Path 2: I'm New to ML (Full Coverage, 12-15 hours)

**Goal:** Build from basics to advanced

**Week 1: Supervised Learning Basics**
1. [Linear Regression](./linear_regression.md) - 1 hour
2. [Regularization](./regularization.md) - 1 hour
3. [Logistic Regression](./logistic_regression.md) - 1.5 hours
4. [Cross-Validation](./cross_validation.md) - 1 hour

**Week 2: Tree-Based Methods**
5. [Decision Trees](./decision_trees.md) - 1 hour
6. [Random Forest](./random_forest.md) - 1.5 hours
7. [Gradient Boosting](./gradient_boosting.md) - 2 hours

**Week 3: Production & Advanced**
8. [Scikit-Learn Workflows](./sklearn_workflows.md) - 1 hour
9. [Hyperparameter Tuning](./hyperparameter_tuning.md) - 1 hour
10. [Imbalanced Data](./imbalanced_data.md) - 1 hour
11. [Model Interpretability](./model_interpretability.md) - 1 hour

### Path 3: I Have ML Experience (Refresher, 3-4 hours)

**Goal:** Review and fill gaps

1. Skim all algorithm files, focus on "When to use" sections - 1.5 hours
2. [Gradient Boosting](./gradient_boosting.md) deep dive - 1 hour
3. [Model Interpretability](./model_interpretability.md) - 1 hour
4. [Imbalanced Data](./imbalanced_data.md) - 30 mins

## 🎓 Key Concepts to Master

### 1. Bias-Variance Tradeoff
- **High bias** = underfitting (too simple)
- **High variance** = overfitting (too complex)
- Every algorithm has this tradeoff

### 2. Train-Test-Validation Split
```
Data → Train (60-70%) → Model Training
    → Validation (15-20%) → Hyperparameter Tuning
    → Test (15-20%) → Final Evaluation
```

### 3. Cross-Validation
- **K-Fold CV**: Split data into K folds, train K times
- **Stratified CV**: Maintains class distribution
- **Time Series CV**: Respects temporal order

### 4. Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression**: MAE, RMSE, R², MAPE
- Choose based on business objective!

### 5. Feature Engineering
- More important than algorithm choice for tabular data
- Domain knowledge > fancy algorithms

## 📊 When to Use What

### Decision Tree
```
✅ Need interpretability
✅ Non-linear relationships
✅ Mixed feature types (numerical + categorical)
❌ Prone to overfitting (use ensemble instead)
❌ Unstable (small data changes → big tree changes)
```

### Random Forest
```
✅ Reduces overfitting compared to single tree
✅ Handles high-dimensional data
✅ Built-in feature importance
✅ Robust to outliers
❌ Less interpretable than single tree
❌ Slower than simpler models
❌ Not great for very high cardinality features
```

### Gradient Boosting (XGBoost/LightGBM)
```
✅ BEST for tabular data (Kaggle winner)
✅ Handles missing values
✅ Built-in regularization
✅ Feature importance
✅ High performance
❌ Longer training time
❌ More hyperparameters to tune
❌ Can overfit if not tuned properly
```

### Logistic Regression
```
✅ Fast training and prediction
✅ Interpretable coefficients
✅ Probabilistic output
✅ Works well with limited data
❌ Assumes linear decision boundary
❌ Requires feature engineering for non-linear patterns
```

### SVM
```
✅ Effective in high-dimensional spaces
✅ Works well with clear margin of separation
✅ Kernel trick for non-linear boundaries
❌ Slow on large datasets
❌ Sensitive to feature scaling
❌ Hard to interpret
❌ Memory intensive
```

## 💡 Pro Tips for Interviews

### 1. Always Start Simple
```python
# Interview approach:
# 1. Baseline: Logistic Regression / Simple Tree
# 2. Ensemble: Random Forest / XGBoost
# 3. Tune and validate

"I'd start with a logistic regression baseline to understand
linear relationships, then try XGBoost to capture non-linear
interactions, comparing via cross-validation."
```

### 2. Discuss Tradeoffs
Never say "XGBoost is always better." Say:
```
"XGBoost typically gives better performance on tabular data,
but requires more tuning and longer training. If we need
fast iteration or interpretability, I'd start with logistic
regression or a decision tree."
```

### 3. Connect to Business
```
"For this fraud detection problem, false negatives
(missed fraud) are more costly than false positives.
I'd optimize for recall and use PR-AUC as the primary metric.
XGBoost with class weights would be my first choice."
```

### 4. Show You Think About Production
```
"While XGBoost gives best accuracy, we need to consider:
- Training time (how often do we retrain?)
- Inference latency (real-time vs batch?)
- Model size (deployment constraints?)
- Interpretability (regulatory requirements?)

These factors might push us toward a simpler model."
```

## 🔗 Connections to Other Sections

- **00 Foundations**: Evaluation metrics, bias-variance - prerequisite for this section
- **02 Deep Learning**: Neural nets vs classical ML - when to use each
- **06 MLOps**: Deploying ML models in production
- **07 System Design**: Building ML systems at scale
- **08 Case Studies**: Applying these algorithms to real problems

## 📈 Success Metrics

You've mastered this section when you can:

- [ ] Explain each major algorithm in 2 minutes (intuition + math basics)
- [ ] Choose appropriate algorithm given problem constraints
- [ ] Implement full ML pipeline in scikit-learn
- [ ] Debug common issues (overfitting, poor metrics, slow training)
- [ ] Answer "Why did you choose this algorithm?" confidently
- [ ] Interpret model predictions using SHAP or LIME
- [ ] Handle imbalanced data appropriately

---

**Ready to start?** → [Gradient Boosting](./gradient_boosting.md) (Highest interview ROI) or [Linear Regression](./linear_regression.md) (If starting from basics)

**Next Section:** [02 Deep Learning](../02_Deep_Learning/README.md)
