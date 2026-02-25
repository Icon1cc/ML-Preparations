# Streaming Pipelines and Real-Time ML

Traditional ML assumes a static dataset (Batch processing). Modern enterprise logistics, fraud detection, and recommendation systems require continuous analysis of unbounded data streams.

---

## 1. Batch vs. Streaming Processing
*   **Batch (e.g., Apache Spark):** Processes data at rest. "Run this job every night at 2 AM to analyze the 10 million rows that arrived yesterday."
*   **Streaming (e.g., Apache Flink, Kafka Streams):** Processes data in motion. "Evaluate every single GPS ping from our delivery trucks the millisecond it hits the server."

## 2. The Core Technology: Apache Kafka
Kafka is the nervous system of modern real-time architectures. It is a distributed event streaming platform.
*   **Publish/Subscribe:** Systems (like a delivery truck IoT sensor) *publish* events (messages) to a specific Kafka "Topic". ML microservices *subscribe* to that topic and consume the events as they arrive.
*   **Durability:** Kafka writes events to disk. If your ML inference server crashes, Kafka holds the messages. When the server restarts, it picks up exactly where it left off, ensuring zero data loss.
*   **Decoupling:** The truck sensor doesn't need to know if the ML model is online. It just drops the message in Kafka and moves on.

## 3. Real-Time Feature Engineering (The Hard Part)
You cannot execute a slow `SELECT SUM(cost) FROM database WHERE user_id = X` during a 50ms real-time inference request. Features must be calculated *as the data flows*.

### Stream Processing Engines (Apache Flink)
Flink sits between Kafka and the ML model. It is designed to calculate "stateful" aggregations on infinite streams.
*   **Rolling Windows:** A common ML feature is "Average speed of the truck over the last 5 minutes."
*   Flink holds a 5-minute sliding window in memory. Every time a new GPS ping arrives, Flink updates the average, and drops the ping from 5 minutes and 1 second ago.
*   It then instantly pushes this pre-calculated feature to a fast **Online Feature Store** (like Redis).

## 4. The Streaming ML Architecture (End-to-End)

1.  **Ingestion:** User swipes a credit card. API gateway publishes `TransactionEvent` to a Kafka Topic.
2.  **Feature Computation:** Apache Flink consumes the event. It calculates real-time features (e.g., "Transactions in last 10 mins = 4").
3.  **Feature Enrichment:** Flink queries the Online Feature Store (Redis) to get static historical features (e.g., "Account Age = 5 years").
4.  **Inference Trigger:** The fully assembled feature vector is published to a new Kafka Topic: `FeaturesReadyEvent`.
5.  **Model Scoring:** The ML Inference Service consumes the vector, runs the XGBoost model, and publishes the result (`FraudProbability = 0.92`) to a `ScoringResult` Topic.
6.  **Action:** The backend banking service consumes the score and blocks the transaction.

*All of this happens in less than 200 milliseconds.*

## 5. Challenges in Streaming ML

### A. Late or Out-of-Order Data
In logistics, a truck might drive through a tunnel and lose cell service. It caches 5 minutes of GPS pings and sends them all at once when it reconnects.
*   *Solution:* Stream processors handle **Event Time** (when the event actually happened) vs. **Processing Time** (when the server received it). Flink uses "Watermarks" to wait a specified amount of time for late-arriving data before finalizing window aggregations.

### B. Online-Offline Skew
If you calculate rolling averages using Flink in production, but you use Pandas to calculate the historical rolling averages for training data, the slight math differences will ruin the model. You must use frameworks that unify batch and stream processing logic.

## Interview Strategy
"For a real-time system like parcel ETA prediction, a traditional database architecture will collapse under the read/write load of continuous GPS pings. I would architect a fully event-driven system using **Apache Kafka** as the message broker, **Apache Flink** to compute rolling window features in real-time, and serve the model predictions asynchronously to avoid blocking upstream services."