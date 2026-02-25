# Case Study: ML for Dynamic Route Optimization

**Company:** Logistics / Ride-Hailing (DHL, Uber, FedEx)
**Scenario:** A driver has 50 packages to deliver today. A standard GPS routing algorithm (like Dijkstra or A*) calculates the shortest distance. However, it assumes a static average speed limit. In reality, traffic, weather, and parking availability drastically alter transit times. Design an ML system to predict dynamic edge transit times to feed into the routing engine.

---

## 1. Problem Formulation
*   **Objective:** Minimize the total daily route time for the driver and maximize SLA compliance (delivering within promised time windows).
*   **The ML Task:** We are *not* solving the routing problem (TSP) with ML. We are solving the **Edge Weight Prediction** problem. We need to predict the exact transit time (in seconds) between Node A and Node B at a specific time of day.
*   **Target Variable:** Continuous Regression (`Actual_Transit_Time_Seconds`).

## 2. Data Engineering & Features

This is a spatio-temporal problem. The features must capture both *where* we are and *when* we are.

### A. Graph & Spatial Features
*   `Distance_Meters` (baseline).
*   `Road_Type` (Highway, Residential, Dirt Road).
*   `Number_of_Intersections` and `Number_of_Traffic_Lights` on the segment.
*   `Historical_Average_Speed_on_Edge` (calculated over the last 3 months for this specific road).

### B. Temporal & Dynamic Features
*   `Hour_of_Day` and `Day_of_Week` (captures rush hour seasonality).
*   `Current_Weather_Severity` (Rain, Snow).
*   `Rolling_Congestion_Index`: (The average speed of all DHL trucks in a 2-mile radius over the last 15 minutes. This is a crucial real-time feature to capture sudden accidents).

## 3. Model Architecture

### Baseline Model
*   A simple `Historical Average` grouped by (Edge_ID, Hour_of_Day, Day_of_Week). Very fast, but completely fails during abnormal events (like a snowstorm or an accident).

### Model 1: XGBoost (The Practical Choice)
*   Treat every historical trip across an edge as a single row of tabular data.
*   **Pros:** Incredibly fast to train and serve. Handles tabular data perfectly.
*   **Cons:** Ignores the spatial relationship of the graph. (If Edge A is jammed, XGBoost doesn't inherently know that Edge B, which connects to it, is likely to get jammed in 5 minutes).

### Model 2: Graph Neural Networks (GNN) (The SOTA Choice)
*   Represent the city as a Graph (Nodes = Intersections, Edges = Roads).
*   Pass the dynamic features (weather, current traffic) into a **Graph Convolutional Network (GCN) or GraphSAGE**.
*   **Mechanism:** The GNN performs "Message Passing." The state of Edge A shares information with Edge B. The model literally learns the spatial flow of traffic.
*   **Pros:** Highly accurate for complex, cascading traffic jams.
*   **Cons:** Extremely complex to implement and slow for real-time inference on a massive city graph.

## 4. System Design & Latency Constraints
The routing engine needs to evaluate thousands of possible routes in seconds. The ML model cannot take 1 second per edge prediction.

**The Decoupled Architecture:**
1.  **Batch Layer (Every 15 mins):** We run the heavy GNN/XGBoost model asynchronously every 15 minutes for *every major road segment in the city*.
2.  **Feature Store (Redis):** The predicted transit times are pushed to a Redis Key-Value store. (`Key: Edge_123_Time:10:15` $ightarrow$ `Value: 45 seconds`).
3.  **Routing Engine (Real-Time):** When the driver requests a route, the C++ routing engine (running A* search) simply does a sub-millisecond `GET` request to Redis to pull the pre-computed edge weights. It never actually calls the ML model directly.

## 5. Evaluation Metrics
*   **Offline:** MAPE (Mean Absolute Percentage Error) of the predicted edge time vs. actual GPS logs.
*   **Online:** "Route Completion Time Delta" - Did the drivers using the ML-weighted routes finish their shifts earlier than those using the baseline static-speed routes?

## Interview Strategy
"The key to this problem is understanding the latency requirements of routing algorithms. An ML model cannot be in the critical path of an A* search evaluating 10,000 edges. My architecture decouples the ML inference from the routing engine. We use a **Graph Neural Network** to pre-compute the dynamic edge weights asynchronously every 10 minutes, store them in **Redis**, and allow the deterministic routing algorithm to perform O(1) lookups."