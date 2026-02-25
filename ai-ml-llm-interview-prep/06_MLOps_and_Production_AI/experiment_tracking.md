# Experiment Tracking

In traditional software, `git` tracks changes to code. In machine learning, code is only one piece of the puzzle. The output of an ML model depends on Code + Data + Hyperparameters + Environment. Experiment tracking is the systematic recording of all these variables.

---

## 1. The Chaos of Un-Tracked ML (Level 0)
Without tracking, data scientists often resort to naming conventions like:
*   `model_final.pkl`
*   `model_final_v2_use_this.pkl`
*   `model_xgboost_learning_rate_01_no_outliers.pkl`

**The Problem:** Six months later, when the production model breaks, nobody knows exactly which dataset it was trained on, what the random seed was, or what the exact `learning_rate` was. It is completely irreproducible.

## 2. Core Components of Experiment Tracking

A modern tracking system (like MLflow or Weights & Biases) logs four main pillars for every single training run:

1.  **Parameters (Inputs):**
    *   Hyperparameters (e.g., `learning_rate=0.01`, `max_depth=5`, `epochs=100`).
    *   Configuration variables (e.g., `batch_size=32`, `optimizer=Adam`).
2.  **Metrics (Outputs):**
    *   Training loss and validation loss over time (creates learning curves).
    *   Final evaluation metrics (e.g., `F1-Score=0.92`, `RMSE=4.5`).
3.  **Artifacts (Files):**
    *   The actual serialized model weights (e.g., `model.onnx` or `model.pt`).
    *   Charts (e.g., ROC curve images, SHAP feature importance plots).
    *   Requirements files (`requirements.txt` or `conda.yaml`) to capture the exact library versions.
4.  **Metadata (Context):**
    *   Git commit hash of the code used for the run.
    *   Data version hash (linking to DVC or lakehouse snapshot).
    *   Who ran the experiment and when.

## 3. Popular Tools

### A. MLflow (Open Source / Databricks)
The industry standard for enterprise.
*   **Tracking Server:** A centralized database/UI to log and compare all experiments.
*   **Model Registry:** Acts as a centralized repository to transition models through stages (`Staging` -> `Production` -> `Archived`).
*   **MLflow Models:** A standard format for packaging models that allows them to be deployed uniformly, regardless of whether they were built in scikit-learn, PyTorch, or TensorFlow.

### B. Weights & Biases (W&B)
Highly popular in Deep Learning and LLM research.
*   Provides exceptional, real-time visualization of metrics and system hardware usage (GPU memory/temp) during massive distributed training runs.

## 4. The Workflow
```python
import mlflow
import mlflow.sklearn

# 1. Start a tracking run
with mlflow.start_run():
    
    # 2. Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)
    
    # Evaluate
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # 3. Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # 4. Log the model artifact
    mlflow.sklearn.log_model(rf, "random_forest_model")
```

## Interview Strategy
"A critical part of any ML system design is reproducibility. I would enforce that no model can be deployed to production unless it is registered in **MLflow**. The registry must contain the exact Git commit of the training script, the hyperparameters used, and the version hash of the dataset. This ensures that if the model catastrophically fails, we can instantly rollback to a previous version and perfectly reproduce the failed environment for debugging."