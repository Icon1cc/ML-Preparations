# Model Versioning and Registries

Training a model is easy. Managing 50 different models, each with 10 different versions, across Dev, Staging, and Production environments is a massive engineering challenge. This is solved by a **Model Registry**.

---

## 1. What is a Model Registry?
A model registry is a centralized, access-controlled repository for managing the lifecycle of ML models. It is the "GitHub for Models." (MLflow Model Registry is the most common example).

It separates the concept of a *Run* (an experiment) from a *Registered Model* (a candidate for production).

## 2. Core Features of a Registry

### A. Lineage and Traceability
Every model in the registry is linked back to the exact experiment run that created it.
If you look at `Demand_Forecaster_v4` in the registry, you can click a link to see its exact hyperparameters, the Git commit of the code, and the Data version used to train it.

### B. Lifecycle Stages
Models don't just exist; they have states. Standard stages include:
*   **None/Development:** The model was just registered.
*   **Staging:** The model is currently undergoing integration testing or A/B testing.
*   **Production:** The model is actively serving live traffic.
*   **Archived:** The model has been deprecated.

### C. Model Signatures
The registry explicitly defines the expected input/output schema of the model.
*   *Input:* `{"distance": float, "is_hazardous": bool}`
*   *Output:* `{"estimated_time_minutes": float}`
If the upstream data pipeline changes and starts sending `distance` as a string, the registry can reject the request before it crashes the model.

## 3. The Deployment Handshake
The Model Registry acts as the bridge between Data Scientists and ML Engineers/DevOps.

1.  **Data Scientist:** Trains a model, sees good offline metrics, and pushes it to the Registry, tagging it as `Staging`.
2.  **CI/CD Pipeline:** A webhook detects a new model in `Staging`. It automatically pulls the model, spins up a Docker container, and runs a suite of latency and integration tests.
3.  **Approval:** If tests pass, an ML Engineer manually clicks "Approve" in the Registry UI.
4.  **Deployment:** The model transitions to `Production`. The Kubernetes cluster detects the state change, pulls the new weights, and performs a rolling update without dropping user requests.

## 4. Model Packaging (Standardization)
A major pain point is a Data Scientist building a model in PyTorch, but the backend engineering team uses Java.
*   **ONNX (Open Neural Network Exchange):** A universal, framework-agnostic format. You export a PyTorch or TensorFlow model to ONNX. The backend can load and run the ONNX file at extremely high speeds (using ONNX Runtime) usually without needing PyTorch installed.
*   **MLflow format:** Wraps the model weights, a `conda.yaml` defining the exact python dependencies, and a standard python inference script into a single zipped artifact.

## Interview Strategy
*   **Scenario:** "Our deployment process is messy. We just email pickle files."
*   **Your Answer:** "This is a massive operational risk. I would introduce a **Model Registry like MLflow**. This creates a single source of truth. We would implement a strict lifecycle where models are registered, tested in a Staging state, and only promoted to Production via automated CI/CD pipelines triggered by the Registry's state changes. Furthermore, I'd standardize packaging by exporting models to **ONNX** to decouple the data science frameworks from our high-performance backend serving infrastructure."