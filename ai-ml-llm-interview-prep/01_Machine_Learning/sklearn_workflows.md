# Scikit-Learn Workflows: Pipelines and Production Code

In coding interviews or take-home assignments, writing "script-kiddie" ML code (doing `fit_transform` on individual columns manually) is a massive red flag. Senior engineers use **Pipelines** to prevent data leakage and write modular, reproducible code.

---

## 1. The Core Problem: Data Leakage
Data leakage occurs when information from the validation or test set is used to train the model.

**The Red Flag Approach (Leakage!):**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all) # BAD! The scaler saw the test data distribution!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all)
model.fit(X_train, y_train)
```

**The Correct Approach:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Scaler learns ONLY from training data
X_test_scaled = scaler.transform(X_test)       # Scaler applies those rules to test data
model.fit(X_train_scaled, y_train)
```

## 2. Using `Pipeline`
A `Pipeline` strings together multiple data transformation steps ending with an estimator (model). When you call `pipeline.fit(X_train)`, it automatically calls `fit_transform` on the preprocessing steps and `fit` on the model, guaranteeing no leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

# Clean, safe, one-liner training
pipe.fit(X_train, y_train)

# Clean, safe inference
predictions = pipe.predict(X_test)
```

## 3. Using `ColumnTransformer`
Real-world datasets have mixed data types. You need to apply One-Hot Encoding to categorical columns and Standard Scaling to numerical columns. `ColumnTransformer` handles this elegantly without pandas spaghetti code.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['age', 'income']
categorical_features = ['city', 'job_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Combine with Pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)
```
*Tip:* Always use `handle_unknown='ignore'` in OneHotEncoder for production, so the model doesn't crash if it sees a new category during inference.

## 4. Hyperparameter Tuning with `GridSearchCV`
You can tune the parameters of *both* the model and the preprocessing steps directly through the pipeline.

```python
from sklearn.model_selection import GridSearchCV

# Note the syntax: stepname__parametername (double underscore)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

# Cross-validation is safe here because the pipeline ensures the
# scaler is re-fit ONLY on the 4 training folds, never the 1 validation fold!
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
```

## Why Interviewers Look for This
1.  **Safety:** Prevents silent data leakage bugs.
2.  **Deployment:** You can `joblib.dump()` the *entire* `full_pipeline` as a single artifact. In production, you just load the pipeline and call `.predict()` on raw incoming JSON data. If you didn't use a pipeline, you'd have to rewrite all your scaling and encoding logic in your API backend.