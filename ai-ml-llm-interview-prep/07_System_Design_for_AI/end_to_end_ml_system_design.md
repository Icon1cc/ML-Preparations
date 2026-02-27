# Framework for End-to-End ML System Design Interviews

ML System Design interviews (like those at FAANG or major tech companies) are notoriously unstructured. You are given a vague prompt like "Design a recommendation system for Netflix" or "Design a route optimization engine."

To succeed, you must drive the conversation using a structured framework. Do not jump straight to the model. Use this 7-step framework.

---

## Step 1: Problem Formulation & Clarification (5 mins)
*Goal: Turn an ambiguous business problem into a concrete ML problem.*
*   **Business Objective:** What are we trying to achieve? (e.g., Increase user engagement, reduce delivery time, cut cloud costs).
*   **Target Variable:** What *exactly* are we predicting? (e.g., Probability a user clicks a movie [Binary Classification], predicted ETA in minutes [Regression]).
*   **Scale/Constraints:** How many users? How many items? What is the required latency (10ms vs 10 hours)? Batch or Real-time?

## Step 2: Metrics (5 mins)
*Goal: Define how you will measure success, both offline and online.*
*   **Offline Metrics:** How do you evaluate the model before deployment? (e.g., AUC-ROC for classification, RMSE for regression, NDCG for ranking).
*   **Online Metrics (Business):** What metric determines if the project is a success? (e.g., Click-Through Rate (CTR), reduction in customer support tickets, average delivery time).

## Step 3: Data Engineering & Feature Extraction (10 mins)
*Goal: Identify what data you need, where to get it, and how to process it.*
*   **Data Sources:** What tables/logs do we need? (User profiles, historical interactions, real-time GPS pings).
*   **Features:** Group them logically:
    *   *User Features:* Age, location, historical click rate.
    *   *Item Features:* Video category, document length, road speed limit.
    *   *Contextual Features:* Time of day, device type, weather.
*   **Handling Issues:** Mention handling missing values, scaling, and categorical encoding. Address **Data Leakage** explicitly (e.g., using time-based splits).

## Step 4: Model Selection & Architecture (10 mins)
*Goal: Start simple, then add complexity with justification.*
*   **Baseline Model:** Always start here. "Before building a neural net, I would build a heuristic baseline (e.g., predict the average historical ETA) or a simple Logistic Regression model."
*   **Primary Model:** Propose the actual model (e.g., XGBoost, Two-Tower Neural Network, Transformers).
*   **Justification:** *Why* this model? (e.g., "I chose XGBoost because it handles tabular, non-linear data well and is fast for inference. A deep learning approach isn't necessary yet because we lack massive unstructured data.")

## Step 5: Training Pipeline & Evaluation (5 mins)
*Goal: How do you train and validate this model at scale?*
*   **Data Splitting:** Emphasize *Time-Series split* or *Out-of-Time validation* (never random k-fold for time-sensitive data like logistics or stock prices).
*   **Imbalance Handling:** Mention SMOTE, class weighting, or down-sampling if dealing with rare events (e.g., fraud detection).
*   **Hyperparameter Tuning:** Brief mention of random search or Bayesian optimization.

## Step 6: Serving & Deployment Architecture (10 mins)
*Goal: How does this model live in production?*
*   **Inference Mode:** Batch (offline predictions saved to a DB) vs. Online (real-time API calls).
*   **Latency Solutions:** If latency is strict (<50ms), discuss decoupling heavy compute. "We can pre-compute user embeddings nightly via Airflow, store them in a fast Feature Store like Redis, and only do a lightweight dot-product calculation at real-time inference."
*   **A/B Testing:** How do you roll it out? Mention shadow deployment, canary releases, and proper A/B testing against the baseline.

## Step 7: MLOps, Monitoring & Continuous Learning (5 mins)
*Goal: Show you are a senior engineer who cares about Day 2 operations.*
*   **Monitoring Drift:**
    *   *Data Drift:* Input feature distributions change over time.
    *   *Concept Drift:* The relationship between features and target changes (e.g., COVID changes user shopping behavior).
*   **Feedback Loop:** How do we get ground truth to retrain the model? (e.g., waiting 24 hours to see if a recommended item was actually bought).
*   **Retraining Strategy:** Trigger-based retraining (when drift exceeds a threshold) vs. Scheduled retraining (every Sunday).