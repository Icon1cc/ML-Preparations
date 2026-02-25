# Reliability and Availability in ML Systems

If your ML system goes down during Black Friday, the company loses millions. Interviewers at FAANG and major logistics companies will grill you on how you handle failure.

---

## 1. Defining the Metrics
*   **Reliability:** Does the system produce the correct, expected output without crashing? (Measured by Error Rates, e.g., 500 status codes).
*   **Availability:** Is the system actually online and reachable? (Measured by "Nines", e.g., 99.99% uptime means ~52 minutes of allowed downtime per year).

## 2. Redundancy (No Single Point of Failure)

### Multi-AZ / Multi-Region Deployment
*   Do not put all your ML inference servers in one AWS Availability Zone (e.g., `us-east-1a`). If that data center loses power, your system dies.
*   Deploy identical clusters across multiple zones (`us-east-1b`, `us-east-1c`) or entirely different geographic regions, with a Global Load Balancer routing traffic to healthy regions.

### Model API Fallbacks (Graceful Degradation)
*   If you rely on the OpenAI API and it goes down, your chatbot shouldn't just print `HTTP 503`.
*   **The Chain:**
    1. Try GPT-4. (Timeout after 3 seconds).
    2. Fallback to Claude 3.5 via Anthropic API. (Timeout after 3 seconds).
    3. Fallback to a small, locally hosted open-source model (Llama 3 8B) running on your own Kubernetes cluster as a last resort.

## 3. Handling Traffic Spikes

### Auto-Scaling
*   Configure Kubernetes Horizontal Pod Autoscaler (HPA).
*   *Rule:* "If average GPU utilization exceeds 70%, automatically spin up 5 more inference pods."

### Asynchronous Queues (Buffering)
*   A synchronous API (REST) will drop requests if traffic exceeds capacity.
*   If predicting an ETA doesn't need to be sub-second, put an **Apache Kafka** queue in front of the model. If 100,000 requests arrive in one second, Kafka safely holds them on disk. The ML workers pull from the queue at their maximum safe capacity, preventing the system from being overwhelmed.

## 4. Feature Store Reliability
If your model requires historical data from a Feature Store (Redis) to make a prediction, what happens if Redis crashes?

*   **Default Values:** The system must catch the database connection error and substitute "safe" default values for the missing features (e.g., if we can't fetch `user_historical_spend`, default it to the global average of `$50.00`). The model's accuracy drops slightly, but it still functions, rather than crashing the entire app.

## 5. Model Degradation (The ML Specific Failure)
A system can have 100% API uptime but 0% Reliability if the model's math has drifted and it's making terrible predictions.
*   **Circuit Breakers:** If the monitoring system detects that the distribution of output predictions has wildly shifted (e.g., the model suddenly predicts every single transaction is Fraud), a circuit breaker trips. The system automatically routes all traffic to a simple, hardcoded heuristic baseline (or a previous, stable version of the model) until an engineer investigates.

## Interview Strategy
*   **The "What if it fails?" Game:** When drawing a system architecture on a whiteboard, proactively point to every box and state what happens when it crashes.
*   **Example:** "I've placed a Feature Store here to provide low-latency lookups. However, recognizing that distributed caches can fail, I've implemented a **Circuit Breaker pattern** in the Inference Service. If the Redis lookup times out after 10ms, the code catches the exception, imputes the missing features with pre-calculated global medians, and executes the inference, ensuring the user still gets a prediction rather than a catastrophic failure."