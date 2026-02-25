# Monitoring and Drift Detection in Production ML

Deploying an ML model is day one. Day two is watching it slowly degrade. Models degrade because the real world changes, but the model is frozen in time based on its training data. This is known as **Model Decay** or **Drift**.

---

## 1. Types of Drift

### A. Data Drift (Feature Drift / Covariate Shift)
The statistical distribution of the input features ($X$) changes over time, but the underlying relationship between features and the target ($y$) remains the same.
*   **Example (Logistics):** You train a model to predict delivery times. Over the summer, a new residential neighborhood is built. Suddenly, the model starts seeing a massive spike in GPS coordinates for that area—a geographic distribution it rarely saw in training.
*   **Action:** The model isn't necessarily broken, but it is operating in unfamiliar territory. Retraining on the new data is recommended.

### B. Concept Drift
The fundamental relationship between the input features ($X$) and the target variable ($y$) changes. What was true during training is no longer true today.
*   **Example (Logistics):** A model predicts that shipping to City A takes 2 days based on historical traffic. Then, a massive new highway is built, cutting the time to 1 day. The input features (distance, weather) are exactly the same, but the target (time) has fundamentally shifted.
*   **Example (E-commerce):** COVID-19 hits. Suddenly, the fact that someone bought hand sanitizer does *not* mean they are going camping (the old concept); it means there is a pandemic.
*   **Action:** The model is broken. Immediate retraining or rolling back to a different model is required.

### C. Prior Probability Shift (Label Drift)
The distribution of the target variable ($y$) changes.
*   **Example:** A spam classifier is trained when 10% of emails are spam. A massive botnet attack occurs, and suddenly 80% of incoming emails are spam.

## 2. How to Detect Drift (Metrics & Math)

To detect drift, you compare a **Reference Window** (usually the training data or data from last month) against a **Current Window** (data from today).

### For Continuous Numerical Features:
*   **Kolmogorov-Smirnov (K-S) Test:** A non-parametric statistical test that compares the cumulative distributions of the two windows. It outputs a p-value. If $p < 0.05$, the distributions are significantly different.
*   **Wasserstein Distance (Earth Mover's Distance):** Measures the "work" required to transform one distribution into the other. Highly robust.

### For Categorical Features / LLM Outputs:
*   **Population Stability Index (PSI):** An industry standard in finance. It calculates how much the population has shifted across different categories/bins.
*   **Kullback-Leibler (KL) Divergence:** Measures how one probability distribution differs from a second, reference probability distribution.

## 3. Monitoring System Architecture

A robust MLOps monitoring setup (using tools like Evidently AI, Arize, or Custom Grafana dashboards) requires three layers:

1.  **System/Operational Metrics:** Is the API up?
    *   Latency (p95, p99), Throughput (RPS), Error Rates (500s), CPU/GPU Utilization.
2.  **Data Quality Metrics:** Is the pipeline broken?
    *   Percentage of missing values (nulls).
    *   Schema changes (a float suddenly becomes a string).
3.  **ML Metrics (Drift & Performance):**
    *   *If Ground Truth is immediate (e.g., Click-Through Rate):* Monitor actual performance metrics (Accuracy, RMSE) directly. If RMSE spikes, trigger an alert.
    *   *If Ground Truth is delayed (e.g., Loan Default takes 6 months):* You *must* monitor Data Drift. If the input features drift significantly, you assume the model's performance will degrade, and you trigger an alert to investigate.

## 4. Retraining Strategies
*   **Scheduled Retraining:** Retrain the model every Sunday night. Simple, but wastes compute if the data hasn't drifted, and leaves you exposed if a sudden shock happens on Monday.
*   **Trigger-Based (Reactive) Retraining:** Monitor drift metrics continuously. Automatically trigger an Airflow/Kubeflow retraining pipeline only when Data Drift or Concept Drift exceeds a predefined threshold (e.g., K-S p-value drops below 0.05 on critical features).