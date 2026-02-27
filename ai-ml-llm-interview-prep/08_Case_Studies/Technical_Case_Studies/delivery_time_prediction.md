# Case Study: Delivery Time Prediction

**Scenario:** "A logistics company is launching a premium 'Exact Hour' delivery service for medical supplies. We need a model that predicts the exact delivery time (ETA) for a specific package currently in transit in a delivery van. You have real-time GPS of the van and the manifest of packages. Design this ML system."

---

## 1. Problem Formulation
*   **Business Objective:** Ensure $>95\%$ SLA compliance for 1-hour delivery windows for premium medical shipments.
*   **Target Variable:** Continuous Regression. We are predicting the **Remaining Transit Time (in minutes)** from the van's current location to the final drop-off point.
*   **Constraints:** High reliability, real-time inference (needs to update on the driver's scanner and customer app every minute), latency $<200ms$.

## 2. Data & Feature Engineering

### Data Sources
1.  **Static Manifest:** Package details (weight, hazmat status), delivery address.
2.  **Dynamic Telemetry:** Van GPS pings (every 10 seconds), current speed, heading.
3.  **Contextual (External):** Real-time traffic APIs (TomTom/Google), Weather APIs.

### Feature Extraction
*   *Spatial Features:* Haversine distance to destination, remaining number of stops before this package.
*   *Temporal Features:* Day of week, hour of day (to capture rush hour patterns).
*   *Historical Features:* Driver's historical average speed in this specific zip code, historical average dwell time at this specific building type (apartment vs. hospital).
*   *Dynamic Features:* Current ETA provided by a basic routing API (used as a baseline feature).

### Handling Data Issues
*   **Missing GPS:** Vans enter tunnels. The system must gracefully handle missing real-time features by falling back to the last known location + average speed.

## 3. Model Architecture

### Baseline
*   Total distance to destination / average historical speed in that zone.

### ML Approach: Two-Stage Model
Predicting ETA is hard because it consists of two very different physical processes: driving on the road, and physically walking the package to the door.
1.  **Stage 1: Drive Time Model (XGBoost or Graph Neural Network)**
    *   Predicts the time it takes the van to physically arrive at the building.
    *   *Why XGBoost?* Excellent at tabular data, handles non-linear relationships (like traffic spikes at 5 PM), and has extremely fast CPU inference.
    *   *Advanced (GNN):* If we have the city road network, a Graph Neural Network can capture spatial spillover (traffic jam on edge A affects edge B).
2.  **Stage 2: Service Time Model (LightGBM)**
    *   Predicts "Dwell Time" (time spent finding parking, going up the elevator, getting a signature).
    *   Features: Building type, floor number, historical signature delays at this address.
*   **Final Prediction:** `Drive Time + Service Time = Total ETA`.

### Loss Function
*   Standard MSE penalizes all errors equally. Since delivering *late* is much worse for medical supplies than delivering *early*, we should use an **Asymmetric Loss Function** (e.g., Quantile Regression, predicting the 85th percentile of expected time) to intentionally bias the model to be slightly conservative.

## 4. Serving & Infrastructure
*   **Streaming Pipeline:** Kafka ingests the 10-second GPS pings. Apache Flink processes these streams, calculates rolling features (e.g., "driver's average speed over the last 5 minutes"), and pushes the updated state to a Redis Feature Store.
*   **Inference API:** A FastAPI service deployed on Kubernetes. When a customer opens the app, the API fetches the latest features from Redis and runs the XGBoost model via ONNX runtime (extremely fast, $<50ms$ latency).

## 5. MLOps & Monitoring
*   **Metric:** We monitor **MAPE** (Mean Absolute Percentage Error) to see how far off our predictions are relative to the journey length.
*   **Drift Detection:** We heavily monitor weather and traffic features for Data Drift. If a blizzard hits, the model will likely under-predict times.
*   **Feedback Loop:** The driver scanning the package at the door provides the absolute ground truth. This is logged immediately to a data warehouse (Snowflake) to be used in the nightly retraining pipeline if concept drift is detected.