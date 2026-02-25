# Batch vs. Real-Time (Online) Inference

Choosing between Batch and Real-Time inference fundamentally changes the architecture, cost, and complexity of an ML system. This is a foundational system design choice.

---

## 1. Batch Inference (Offline)

Predictions are generated asynchronously on a schedule (e.g., nightly) for a large group of inputs, and the results are stored in a database to be read later.

*   **Example:** Netflix generating "Recommended for You" movie lists for all 200 million users every night at 2 AM.
*   **Workflow:**
    1. A job scheduler (Apache Airflow) triggers at 2 AM.
    2. A distributed compute framework (Apache Spark) pulls all user profiles from a Data Warehouse.
    3. The model processes the data in massive parallel batches.
    4. The predictions are written back to a fast Key-Value store (Redis or Cassandra).
    5. *Next day:* When a user opens the app, the backend simply does a fast `GET` request to Redis to fetch the pre-computed list.
*   **Pros:**
    *   **Cost & Efficiency:** Highly optimized. GPUs run at 100% utilization.
    *   **No Latency Pressure:** The user is not waiting for the model to run.
    *   **Simplicity:** If the ML pipeline crashes, you just restart it. The user never notices because the database still holds yesterday's predictions.
*   **Cons:**
    *   **Stale Data:** If a user suddenly watches 5 action movies at 10 AM, their recommendations won't update to reflect "Action" until the next day's batch run.

## 2. Real-Time Inference (Online / Synchronous)

Predictions are generated on-the-fly, synchronously, the exact moment the user requests them.

*   **Example:** Credit Card Fraud Detection. The model must analyze the transaction and return an Approve/Deny decision within 100 milliseconds before the card terminal times out.
*   **Workflow:**
    1. User swipes card. Backend sends a REST/gRPC API request to the ML Inference Server (e.g., Kubernetes + TorchServe).
    2. The server instantly fetches the user's historical features from a low-latency Feature Store (Redis).
    3. The model runs the forward pass.
    4. The decision is returned to the terminal.
*   **Pros:**
    *   Reacts instantly to changing context and real-time user behavior.
*   **Cons:**
    *   **Strict Latency Constraints:** Complex models (like massive Transformers) might be too slow.
    *   **Complex Infrastructure:** Requires auto-scaling clusters to handle sudden spikes in traffic. If the model server goes down, the entire business process stops.
    *   **Cost:** You must keep API servers running 24/7, even if traffic is low.

## 3. Streaming Inference (Near Real-Time / Asynchronous)

A middle ground. Predictions are triggered by real-time events, but processed asynchronously via message queues.

*   **Example:** DHL dynamic route ETA updates. As a truck drives, its GPS pings every 10 seconds.
*   **Workflow:**
    1. GPS pings are sent to a message broker (Apache Kafka).
    2. A stream processing engine (Apache Flink or Spark Streaming) consumes the ping.
    3. It updates rolling features and passes the data to the model.
    4. The updated ETA is pushed back to Kafka, which updates the driver's dashboard.
*   **Pros:** Very fast, but doesn't block the core application thread like a synchronous API call. Handles massive traffic spikes elegantly via queue buffering.

## Interview Strategy: The Hybrid Approach
If asked to design a recommendation system, always propose a **Hybrid Architecture**:
"Generating recommendations for millions of items in real-time is impossible. I would use **Batch Inference** nightly to pre-compute User Embeddings and store them in a Feature Store. At runtime (**Real-Time Inference**), when the user logs in, we do a fast nearest-neighbor search, and use a lightweight reranking model to adjust the top 50 items based on their immediate, real-time clicks from the last 5 minutes."