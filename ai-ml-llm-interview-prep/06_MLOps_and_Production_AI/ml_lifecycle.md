# The ML Lifecycle (MLOps)

MLOps (Machine Learning Operations) is the discipline of bridging the gap between a Data Scientist's Jupyter Notebook and a reliable, scalable production system. It brings DevOps principles (CI/CD, monitoring, testing) to machine learning.

---

## The 4 Phases of the ML Lifecycle

### Phase 1: Data Engineering & Feature Extraction
*   **Data Ingestion:** Extracting data from source systems (SQL, Data Lakes, Kafka streams).
*   **Data Validation:** Using tools like Great Expectations to ensure the data schema hasn't changed (e.g., checking that `Age` is an integer between 0 and 120).
*   **Feature Engineering:** Creating ML-ready features.
*   **Feature Store:** A centralized repository (e.g., Feast, Hopsworks) that stores calculated features. This ensures the exact same feature code is used during training (batch) and inference (real-time), preventing train-serving skew.

### Phase 2: Model Development & Experimentation
*   **Experiment Tracking:** Data scientists run hundreds of experiments changing hyperparameters. Tools like **MLflow** or **Weights & Biases (W&B)** automatically log the parameters, metrics (RMSE, Accuracy), and model artifacts for every run.
*   **Model Registry:** Once a successful model is trained, it is placed in the Model Registry (a GitHub-like repository for models) and tagged as `Staging`.

### Phase 3: Continuous Integration & Deployment (CI/CD for ML)
Unlike traditional software where CI/CD only tests code, ML CI/CD must test Code + Data + Model.
*   **Continuous Integration (CI):**
    *   Linting and unit testing the Python code.
    *   Testing the model artifact (e.g., does it load correctly? Is the inference latency under 100ms on a CPU?).
*   **Continuous Deployment (CD):**
    *   Containerizing the model (Docker).
    *   Deploying it as a REST/gRPC API using servers like **TorchServe**, **Triton**, or **Seldon Core** on Kubernetes.
    *   *Release Strategies:* Shadow Deployment (running in parallel with the old model without affecting users), Canary Release (routing 5% of traffic to the new model).

### Phase 4: Continuous Monitoring & Retraining (CT)
Deploying is day one. Models instantly begin to degrade in the real world.
*   **Monitoring:** Tracking System Latency, Data Drift, and Concept Drift (See `monitoring_and_drift_detection.md`).
*   **Continuous Training (CT):** This is the Holy Grail of MLOps. When drift is detected, an automated pipeline (e.g., via Airflow or Kubeflow) is triggered. It fetches the latest data, retrains the model, evaluates it against the current production model, and automatically deploys the new model if it is statistically better.

## MLOps Maturity Levels (Google Framework)
Interviewers often ask where a system falls on this scale.
*   **Level 0 (Manual):** Script-driven, interactive notebooks. No CI/CD. The data scientist hands a `.pkl` file to engineering. (Massive technical debt).
*   **Level 1 (ML Pipeline Automation):** Automated training pipelines. Data validation and experiment tracking are in place. Deployment is automated.
*   **Level 2 (CI/CD Pipeline Automation):** The CI/CD system doesn't just deploy models; it automatically builds, tests, and deploys the *pipelines* that train the models. Fully automated feedback loops.

## Key Terminology
*   **Train-Serving Skew:** When the model's performance in production is significantly worse than in training because the data pipelines differ. (e.g., using Pandas `fillna` in training, but the production Java backend handles Nulls differently). Solved by Feature Stores.
*   **Data Provenance/Lineage:** Tracking exactly which dataset version was used to train which model version, crucial for compliance and debugging.