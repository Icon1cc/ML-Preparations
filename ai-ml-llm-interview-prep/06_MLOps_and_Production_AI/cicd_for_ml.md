# CI/CD for Machine Learning (CT)

Continuous Integration and Continuous Deployment (CI/CD) in software engineering ensures that code changes are automatically tested and safely deployed. In MLOps, the concept is expanded to **Continuous Training (CT)**, because ML systems fail not just when code breaks, but when *data* changes.

---

## 1. The Three Pillars of ML CI/CD
In traditional software, the artifact is a compiled binary. In ML, the artifact is a trained model. It depends on three moving parts:
1.  **Code:** The Python logic for feature engineering, model architecture, and inference.
2.  **Data:** The actual dataset used for training.
3.  **Model:** The serialized weights and hyperparameters.

## 2. Continuous Integration (CI) in ML
Triggered when a Data Scientist opens a Pull Request (PR) to merge new code (e.g., adding a new feature column).
*   **Linting & Formatting:** Standard `flake8`, `black`, `mypy`.
*   **Unit Tests:** Does `calculate_distance()` work?
*   **Data Validation Tests:** Does the data loader correctly handle NaNs? Are all expected columns present?
*   **Training Pipeline Test:** Run the entire training script on a tiny, mocked dataset (e.g., 100 rows) for 2 epochs. The goal isn't accuracy; the goal is to ensure the code doesn't crash with `Out Of Memory` or matrix dimension mismatch errors.

## 3. Continuous Training (CT) - The ML Differentiator
Triggered either by a Schedule (e.g., every Sunday at 2 AM) or by a Trigger (e.g., a monitoring system detects Data Drift).
*   **Data Ingestion:** The pipeline (e.g., managed by Apache Airflow or Kubeflow) fetches the freshest data from the Data Warehouse.
*   **Full Training:** Spins up cloud GPU instances and trains the model from scratch on the new data.
*   **Model Evaluation:** The new model is tested against a hold-out set. 
    *   *The Gate:* If the new model's RMSE is worse than the model currently in production, the pipeline **fails** and aborts. It does not deploy a degraded model.
*   **Registration:** If the model beats the baseline, it is logged into the Model Registry (e.g., MLflow) and tagged as `Staging`.

## 4. Continuous Deployment (CD) in ML
Triggered when a model in the Registry is promoted to `Production`.
*   **Packaging:** The CD system pulls the model artifact and packages it into a Docker container alongside an API server (like FastAPI, TorchServe, or Triton).
*   **Deployment:** The container is pushed to a Kubernetes cluster or a serverless environment (AWS SageMaker).
*   **Safe Rollout Strategies:**
    *   **Shadow Mode:** The new model receives live production traffic and makes predictions, but those predictions are *discarded*. They are only logged to a database to compare against the old model's predictions in real-time. No user impact.
    *   **Canary Release:** 5% of user traffic is routed to the new model. If error rates (500s) or business metrics (CTR) don't crash, it gradually scales to 100%.
    *   **A/B Testing:** Routing traffic 50/50 to measure actual business impact.

## 5. Standard Tooling Stack
*   **Orchestration:** Jenkins, GitHub Actions, GitLab CI (for the code logic).
*   **Pipeline DAGs:** Apache Airflow, Kubeflow Pipelines, Vertex AI Pipelines (for the data/training logic).
*   **Infrastructure:** Terraform / Infrastructure as Code (IaC).

## Interview Strategy
"The biggest mistake companies make is treating ML deployment like software deployment. If an ML engineer manually trains a model in a notebook and hands a `.pkl` file to DevOps, that is a brittle Level 0 system. I design CI/CD pipelines that automate the *training process itself*. We deploy the pipeline, and the pipeline continuously deploys the models."