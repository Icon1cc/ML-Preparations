# Data Architecture Patterns for ML

Machine learning models are only as good as the data pipelines feeding them. In system design interviews, you must demonstrate how you store, transform, and access data at scale.

---

## 1. The Data Warehouse (The Traditional Approach)
*   **Examples:** Snowflake, Amazon Redshift, Google BigQuery.
*   **Concept:** Structured data only. Data is extracted from source systems, transformed, and loaded (ETL) into highly organized SQL tables.
*   **Pros:** Excellent for BI dashboards and clean tabular ML training (e.g., XGBoost models on sales data).
*   **Cons:** Cannot store unstructured data (images, audio, raw text logs). Extremely expensive to scale storage.

## 2. The Data Lake (The Big Data Approach)
*   **Examples:** AWS S3, Google Cloud Storage, Azure Data Lake.
*   **Concept:** Store absolutely everything in its raw, native format (JSON, CSV, JPG, MP4). No schema is required upon insertion.
*   **Pros:** Very cheap storage. Perfect for Deep Learning (Computer Vision, NLP).
*   **Cons:** Can easily turn into a "Data Swamp" where nobody knows what the files actually mean or how to parse them. Cannot run SQL queries directly without complex abstraction layers.

## 3. The Data Lakehouse (The Modern Standard)
The architecture you should propose in almost every interview.
*   **Examples:** Databricks (Delta Lake), Apache Iceberg, Apache Hudi.
*   **Concept:** Combines the cheap, scalable storage of a Data Lake with the structured SQL querying and ACID transaction capabilities of a Data Warehouse.
*   **How it works:** You dump raw data into cheap object storage (S3). Then, you use a framework like Delta Lake to build a metadata layer on top of those files. This allows you to run fast `SELECT` queries directly on your S3 buckets.
*   **ML Benefit:** Data Scientists can use PySpark to query 50 Terabytes of raw data using standard SQL syntax directly from the Lakehouse, without needing to copy the data into an expensive Redshift cluster first.

## 4. Lambda Architecture (Handling Speed vs. Scale)
How do you build a system that can train on 10 years of historical data, but also update predictions based on data from 5 seconds ago?

*   **The Batch Layer (Cold Path):** Calculates massive aggregations over the entire historical dataset (e.g., using Spark on a Lakehouse). It is comprehensive but slow (runs daily).
*   **The Speed Layer (Hot Path):** Processes only the most recent data arriving in real-time (e.g., using Flink reading from Kafka). It calculates fast, approximate metrics.
*   **The Serving Layer:** A database (like Cassandra or Redis) that merges the views from the Batch and Speed layers so the API can return a complete, up-to-date picture to the user.

*Note: The Kappa Architecture is a modern alternative that attempts to use the stream processing engine (Flink) for both the hot and cold paths to simplify the code base.*

## 5. Event-Driven Architecture (Pub/Sub)
*   **Concept:** Microservices do not communicate directly via synchronous REST APIs. Instead, they publish "Events" to a message broker (Kafka).
*   **ML Benefit:** When a user clicks an ad, the backend doesn't need to know that 5 different ML models need that data. It just publishes a `UserClickedAd` event to Kafka. The Recommendation Model, the Fraud Model, and the Analytics Model all independently subscribe to that topic and consume the data at their own pace.

## Interview Strategy
*   "To design the data architecture for this computer vision defect detection system, I would adopt a **Data Lakehouse** approach using **Delta Lake** on S3. This allows us to cheaply store the massive volume of raw high-resolution factory images, while simultaneously maintaining structured SQL tables containing the metadata and QA labels for those images, enabling seamless PyTorch training loops."