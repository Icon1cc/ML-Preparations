# Cloud Deployment Patterns for ML

Once a model is trained and registered, how do you actually serve it to users? The deployment pattern chosen impacts latency, cost, and reliability.

---

## 1. Model as a Microservice (REST/gRPC API)
The most common approach. The model is wrapped in a web server framework and deployed as an independent service.

### Technologies
*   **FastAPI / Flask (Python):** Easy, but standard Python is notoriously slow for concurrent requests due to the GIL (Global Interpreter Lock). Good for low-traffic or prototyping.
*   **TorchServe / Triton Inference Server:** Enterprise-grade model servers built by PyTorch and Nvidia. They handle dynamic batching, model versioning, and highly optimized GPU memory management natively.

### Dynamic Batching (Crucial for GPU Efficiency)
*   *The Problem:* A GPU is wasted if you send it 1 image at a time. It wants 32 images at once.
*   *The Solution:* Triton Server intercepts incoming API requests. If 10 different users request a prediction within a 10ms window, Triton holds them, batches them together into a single matrix, sends the batch to the GPU, and demultiplexes the answers back to the correct users. This drastically increases throughput.

## 2. Serverless Deployment
You deploy the model code, and the cloud provider handles the infrastructure, scaling from 0 to 10,000 servers automatically.
*   **Tools:** AWS Lambda, Google Cloud Run, AWS SageMaker Serverless.
*   **Pros:** You only pay for exactly the compute you use. Zero maintenance.
*   **Cons:** 
    *   **Cold Starts:** If the service hasn't been used in 15 minutes, it scales to 0. The next request requires the cloud to spin up a server, download a 2GB model into RAM, and boot Python. This can cause a 10+ second delay for that unlucky user. 
    *   Usually not suitable for massive models requiring GPUs.

## 3. Deployment Topologies (Rolling out safely)

### A. Shadow Deployment
*   The new model (V2) is deployed alongside the old model (V1).
*   Live user traffic is sent to V1, and V1's prediction is returned to the user.
*   However, the exact same data is also asynchronously forwarded to V2. V2 makes a prediction, but it is only logged to a database, never shown to the user.
*   *Why?* The safest way to test if V2 crashes under real-world data or latency loads before trusting it with actual business impact.

### B. Canary Deployment
*   A small percentage (e.g., 5%) of live user traffic is routed to the new model (V2).
*   You monitor business metrics (e.g., conversion rate) and system metrics (error rates). If V2 performs well, you gradually increase the dial (10% $ightarrow$ 50% $ightarrow$ 100%) until V1 is completely replaced.
*   If V2 fails, the blast radius is limited to only 5% of users, and you instantly roll back.

### C. A/B Testing
*   Similar to Canary, but focused on measuring *business impact* rather than system stability. Traffic is split evenly (50/50), and statistically rigorous hypothesis testing is used to determine if the new model actually drives more revenue or better user engagement.

## 4. Edge Deployment (On-Device)
Deploying the model directly onto a mobile phone, IoT device, or a scanner gun in a a warehouse.
*   **Pros:** Zero network latency (works offline), zero cloud compute costs, maximum data privacy.
*   **Cons:** Devices have terrible CPUs, minimal RAM, and battery limits. 
*   **Techniques:** Requires heavy model compression: Quantization (INT8), Pruning, and Knowledge Distillation. Models must be exported to mobile-optimized formats like TFLite, CoreML (Apple), or ONNX.