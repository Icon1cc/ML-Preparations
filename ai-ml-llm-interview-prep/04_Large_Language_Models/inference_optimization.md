# LLM Inference Optimization

Inference (generating text) is the most expensive part of running an LLM in production. Unlike training, which is done once, inference happens millions of times a day. Optimizing it is critical for business viability.

---

## 1. The Bottleneck: Memory Bandwidth
For modern LLMs, generating a token is usually **Memory-Bound**, not Compute-Bound. The GPU spends more time moving the massive weight matrices from High Bandwidth Memory (HBM) into the compute cores (SRAM) than it does actually doing the math.

## 2. Key Optimization Techniques

### A. KV Cache (Key-Value Cache)
*   **The Problem:** LLMs generate text auto-regressively (one word at a time). To predict word #100, the model normally has to recalculate the attention scores for words 1-99 all over again. This is $O(N^2)$ and incredibly slow.
*   **The Solution:** The **KV Cache**. We save the computed Key and Value matrices for words 1-99 in GPU memory. When predicting word #100, we only compute the Q, K, V for word #100, and query it against the *cached* keys and values of the past.
*   **The New Bottleneck:** The KV cache grows linearly with sequence length. A 128k context window KV cache will easily consume 50GB+ of VRAM, severely limiting how many concurrent users a single GPU can handle.

### B. PagedAttention (vLLM)
The most important breakthrough in LLM serving.
*   **The Problem:** Standard KV caching pre-allocates a massive contiguous block of memory for the maximum possible sequence length. If a user only generates 10 tokens, 99% of that allocated memory is wasted (internal fragmentation).
*   **The Solution:** Inspired by Operating System virtual memory. PagedAttention breaks the KV cache into small fixed-size blocks (pages). It only allocates a new page when the model actually needs to generate more tokens.
*   **Impact:** Completely eliminates memory fragmentation, allowing you to serve 4x to 5x more concurrent users on the exact same GPU hardware. (This is the core technology behind the `vLLM` library).

### C. Continuous Batching (In-Flight Batching)
*   **The Problem:** Standard batching waits for all 10 requests in a batch to finish generating before returning. If one request is generating a 5-word sentence and another is generating a 500-word essay, the 5-word request is held hostage, wasting GPU cycles.
*   **The Solution:** As soon as the 5-word request finishes, the system immediately evicts it from the batch and inserts a brand new user request into that empty slot, while the 500-word essay continues generating uninterrupted.

### D. Speculative Decoding
*   **Concept:** Uses a tiny, fast "Draft" model (e.g., 1B parameters) to quickly guess the next 5 tokens. Then, the massive "Target" model (e.g., 70B parameters) checks all 5 tokens in parallel in a single forward pass.
*   **Why it works:** Checking tokens in parallel is much faster than generating them sequentially. If the Draft model guessed correctly, you just generated 5 tokens for the time cost of 1. If it guessed wrong, you throw away the bad tokens and keep going. Yields a 2x-3x speedup with absolutely zero loss in quality.

## 3. Serving Engines
Do not write custom PyTorch loops for production inference. Use optimized engines:
*   **vLLM:** The industry standard for open-source models. Incredible throughput via PagedAttention.
*   **TGI (Text Generation Inference):** HuggingFace's enterprise solution.
*   **TensorRT-LLM:** Nvidia's highly optimized (but difficult to configure) engine for absolute maximum performance on specific hardware.