# Cost vs. Latency Tradeoffs in AI Systems

In the real world, you cannot deploy a massive GPT-4 equivalent model for every minor task. The economics will destroy the business. System design interviews test your ability to balance intelligence, speed, and hardware costs.

---

## 1. The Core Tradeoff Triangle
You can rarely optimize for all three simultaneously:
1.  **Accuracy (Intelligence):** Requires massive parameter counts (70B+), causing slow latency and high cost.
2.  **Latency (Speed):** Requires small parameter counts or massive distributed GPU clusters.
3.  **Cost (Efficiency):** Requires small models, quantization, or maximizing batch sizes (which inherently adds latency).

## 2. Model Routing (Dynamic Dispatch)
Don't use a sledgehammer for a nail.
*   **The Strategy:** Implement a Router (a very small, fast classifier or regex engine).
*   **The Logic:**
    *   *Query:* "What is 2+2?" $ightarrow$ Route to cheap, fast Llama-3-8B. (Cost: \$0.0001, Latency: 200ms).
    *   *Query:* "Write a complex Python script to manage a Kubernetes cluster." $ightarrow$ Route to expensive, slow GPT-4o. (Cost: \$0.05, Latency: 5s).
*   *Business Impact:* You cut your API bill by 80% while maintaining high perceived intelligence.

## 3. Batching Tradeoffs (Throughput vs. Latency)
*   **Max Latency Focus:** Process every request the millisecond it arrives. 
    *   *Result:* Lightning fast for the user (50ms). But the GPU only processes 1 item at a time. Throughput is low. Server costs are astronomical because you need 100 GPUs to handle 100 concurrent users.
*   **Max Throughput Focus (Cost Savings):** Make requests wait in a queue for 500ms until you have a batch of 64 requests.
    *   *Result:* User waits slightly longer (550ms). But the GPU processes all 64 items simultaneously. You can handle 100 concurrent users with just 2 GPUs. 
*   **The Decision:** In fraud detection, optimize for latency. In nightly batch recommendations, optimize completely for throughput.

## 4. Hardware and Optimization Tradeoffs

### Quantization (INT8 / INT4)
*   **Tradeoff:** We compress the model from 16-bit to 4-bit.
*   **Gain:** Model fits on a cheap \$1,000 consumer GPU instead of a \$10,000 enterprise GPU. Inference is 3x faster.
*   **Loss:** The model loses about 2-3% of its benchmark accuracy and might hallucinate slightly more often on complex logic.

### Context Window Truncation (RAG)
*   **Tradeoff:** An LLM API charges *per token*. Passing 20 documents into the context window is highly accurate but costs \$0.10 per query and takes 10 seconds to read.
*   **Gain/Fix:** Use an open-source Cross-Encoder to rerank the documents locally (Free compute). Only pass the top 2 documents to the API. Cost drops to \$0.01 per query, latency drops to 2 seconds.

## 5. Caching Strategies (The Ultimate Hack)
*   **Exact Caching (Redis):** If the user asks the exact same tracking question, return the saved answer. (Latency: 1ms, Cost: \$0.00).
*   **Semantic Caching:** If the user asks "Where is my package?" and another asked "Where's my box?", calculate the vector distance. If $>95\%$ similar, return the cached answer. 

## Interview Strategy
"In designing this customer support chatbot, I recognize a severe cost constraint. Calling a frontier model for 50,000 daily queries would cost hundreds of thousands of dollars a year. Therefore, my architecture utilizes **Semantic Caching** to deflect 30% of queries instantly. For the remaining queries, I use an **LLM Router** to send basic FAQ questions to a locally hosted, quantized 8B model, reserving the expensive GPT-4 API strictly for complex, multi-turn technical support escalations."