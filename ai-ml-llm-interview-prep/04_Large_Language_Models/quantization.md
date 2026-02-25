# Model Quantization

Deploying massive LLMs in production is bottlenecked by VRAM (GPU memory). A 70B parameter model in standard 16-bit precision requires over 140GB of VRAM just to load the weights, requiring multiple expensive A100 GPUs. Quantization solves this.

---

## 1. The Core Concept
Quantization is the process of mapping continuous, high-precision numbers (like 32-bit or 16-bit floats) into a lower-precision format (like 8-bit or 4-bit integers). 
*   **The Benefit:** Drastically reduces VRAM footprint and memory bandwidth requirements (the main bottleneck for LLM generation speed), enabling models to run on consumer GPUs or edge devices.
*   **The Cost:** Slight loss of numerical precision, which can lead to a minor degradation in model accuracy/reasoning capability.

## 2. Post-Training Quantization (PTQ)
Quantizing a model *after* it has been fully trained in high precision.

### A. Naive Quantization (e.g., Min-Max)
You find the minimum and maximum values in a weight tensor and linearly map that range to integer values (e.g., -127 to +127 for INT8).
*   **The Outlier Problem:** If 99% of your weights are between -1 and 1, but one massive outlier weight is 100, the min-max scaling will compress all the important information into a tiny range, destroying the model's intelligence.

### B. Advanced PTQ (AWQ, GPTQ)
Modern techniques solve the outlier problem.
*   **GPTQ:** Uses second-order mathematical information (the Hessian matrix) during quantization to determine which weights are most "sensitive." It quantizes the less sensitive weights aggressively while preserving the critical outlier weights in higher precision.
*   **AWQ (Activation-Aware Weight Quantization):** Realizes that weights aren't equally important. It runs a small calibration dataset through the model and observes the *activations*. Weights that produce large activations are kept in higher precision, while the rest are aggressively quantized to 4-bit. AWQ is currently the standard for fast, high-quality production deployment (e.g., via vLLM).

## 3. Quantization-Aware Training (QAT)
If PTQ degrades performance too much, QAT is used.
*   **Mechanism:** The model simulates the effects of low-precision quantization *during the training process itself*. The weights are stored in high precision, but during the forward pass, they are "fake quantized" to 8-bit.
*   **Result:** The optimizer learns to adapt the weights to be robust against the quantization noise. Yields much better accuracy than PTQ, but requires running a training loop.

## 4. QLoRA (Quantized LoRA)
The standard for fine-tuning large models on a budget.
1.  Take a massive Base Model (e.g., 70B parameters).
2.  Quantize its frozen weights to **4-bit NormalFloat (NF4)**.
3.  Attach small, trainable LoRA adapters in **16-bit (bfloat16)**.
4.  During the forward pass, the 4-bit weights are briefly de-quantized to 16-bit to calculate activations.
5.  Gradients are calculated and applied *only* to the 16-bit LoRA adapters.
*   **Impact:** You can fine-tune a 70B model on a single 48GB GPU, which was previously impossible.

## 5. KV Cache Quantization
In production inference, storing the Key and Value matrices for every token in the context window (the KV Cache) consumes massive VRAM.
*   **FP8 KV Cache:** We can quantize not just the model weights, but the KV cache itself down to 8-bit. This doubles the maximum context length you can fit on your GPU, drastically increasing maximum concurrent users (throughput) with almost zero accuracy loss.

## Interview Summary
*   **For fast deployment:** Download an **AWQ** quantized 4-bit model from HuggingFace and serve it with vLLM.
*   **For cheap fine-tuning:** Use **QLoRA**.
*   **Data Types:** Be familiar with `FP16` (standard), `BF16` (better dynamic range, prevents overflow during training), `INT8` (fast integer math), and `NF4` (specialized 4-bit format for QLoRA).