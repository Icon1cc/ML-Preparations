# Case Study: Anomaly Detection in the Supply Chain

**Company:** Global Logistics (e.g., DHL)
**Scenario:** DHL moves millions of containers annually. Occasionally, things go wrong: a container is stolen, damaged, or diverted for smuggling. We have sensor data (GPS, Temperature, Humidity, Accelerometer) from IoT devices inside the containers. Design an ML system to detect these anomalies in real-time.

---

## 1. Problem Formulation
*   **Goal:** Identify high-risk containers for inspection without disrupting the flow of 99.9% of legitimate cargo.
*   **Task:** Unsupervised or Semi-supervised Anomaly Detection.
*   **The Challenge:** We have almost no "ground truth" labels for "theft" or "smuggling" (extreme class imbalance). We must find "deviations from normal behavior."

## 2. Feature Engineering

### Time-Series Features (Rolling Windows)
*   *Trajectory:* Current GPS distance from the planned route (corridor analysis).
*   *Dynamics:* Sudden spikes in Accelerometer data (indicating the container was dropped or forced open).
*   *Environmental:* Temperature deviations (critical for "Cold Chain" items like vaccines).
*   *Aggregates:* Rolling mean/variance of humidity over the last 4 hours.

### Contextual Features
*   *Port Profile:* Is the container currently at a port with high historical theft rates?
*   *Item Profile:* Is the cargo high-value (electronics) vs. low-value (gravel)?
*   *Carrier Profile:* Reliability score of the specific sub-contractor trucking company.

## 3. Modeling Strategy

### Stage 1: The Baseline (Rule-Based)
*   Simple thresholding: `IF distance_from_route > 10km OR temp > 8°C THEN Alert`.
*   *Cons:* High false-positive rate. A truck might take a 12km detour because of a road accident, which is not an anomaly.

### Stage 2: Unsupervised Learning (Isolation Forest)
*   **Isolation Forest:** A tree-based algorithm that works by "isolating" observations. Anomalies are easier to isolate (require fewer splits) than normal points.
*   **Pros:** Very fast, handles high-dimensional data, no labels required.

### Stage 3: Autoencoders (Deep Learning)
*   **Concept:** Train a neural network to reconstruct the input features.
    *   *Input:* Rolling window of sensor data.
    *   *Bottleneck:* Compress to low dimension.
    *   *Output:* Reconstruct the original sensor data.
*   **Detection:** The model is trained *only* on normal data. When it sees an anomaly, it will fail to reconstruct it accurately. The **Reconstruction Error** becomes our anomaly score.

## 4. Evaluation & Human-in-the-Loop
Since we lack labels, how do we know if it works?
1.  **Precision at K:** If we alert on 100 containers, how many were actually found to have an issue by the physical inspection team?
2.  **False Discovery Rate:** This is the most important business metric. If we flag too many false positives, the port operations will grind to a halt.
3.  **Active Learning:** Every time a human inspector checks a container, they log a "Reason Code" (e.g., `Mechanical Failure`, `Theft Attempt`, `False Alarm`). We use these labels to transition from Unsupervised to a **Supervised** XGBoost classifier over time.

## 5. Deployment Architecture
*   **Edge vs. Cloud:** The IoT device should perform simple thresholding (Edge) to save battery and data costs.
*   **Real-time Streaming:** Sensor pings are sent via MQTT/Kafka to a cloud-based inference engine.
*   **Alerting:** High-risk scores trigger a ticket in the Security Operations Center (SOC) dashboard.

## Interview Tip: "The Business Tradeoff"
"Anomaly detection isn't just about math; it's about the **Cost of False Positives**. In logistics, stopping a container for 24 hours costs money. I would design the system to have different 'Alert Tiers'. 
*   *Tier 1 (High):* Direct human intervention.
*   *Tier 2 (Medium):* Flag for secondary digital documentation check.
*   *Tier 3 (Low):* Log for future historical carrier auditing." 
This shows you understand the operational reality of DHL.