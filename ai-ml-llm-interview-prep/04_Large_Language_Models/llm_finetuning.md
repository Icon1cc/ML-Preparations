# LLM Fine-Tuning: Parameter-Efficient Fine-Tuning (PEFT) and LoRA

While pre-training creates a base model, fine-tuning adapts it to specific tasks or domains. Full fine-tuning (updating all parameters) of massive LLMs is often computationally prohibitive. This leads to the dominance of Parameter-Efficient Fine-Tuning (PEFT) methods.

---

## 1. Full Fine-Tuning
Updating every single weight in the neural network during training.
*   **Pros:** Maximum theoretical performance; can fundamentally alter the model's knowledge or behavior.
*   **Cons:** Requires massive GPU memory (VRAM). For a 7B parameter model, you need memory for weights, gradients, optimizer states (e.g., Adam stores 2 extra states per parameter), and activations. This often requires multi-GPU clusters (DeepSpeed, FSDP).
*   **Risk:** Catastrophic Forgetting (forgetting general knowledge while over-fitting to the narrow fine-tuning dataset).

## 2. Parameter-Efficient Fine-Tuning (PEFT)
Techniques that freeze most of the original model weights and only train a small number of extra parameters.
*   **Benefits:** Drastically reduces VRAM requirements (can fine-tune 7B models on a single consumer GPU). Prevents catastrophic forgetting. Allows for hot-swapping different fine-tuned "adapters" on the same base model in production.

### LoRA (Low-Rank Adaptation)
The industry standard PEFT method.
*   **Concept:** Instead of updating the massive weight matrix $W$ directly (where $W_{new} = W + \Delta W$), LoRA freezes $W$ and approximates the update matrix $\Delta W$ using two smaller, low-rank matrices $A$ and $B$.
*   **Math:** $\Delta W = A 	imes B$. If $W$ is $1000 	imes 1000$ (1,000,000 parameters), and we choose a rank $r=8$, $A$ is $1000 	imes 8$ and $B$ is $8 	imes 1000$. Total new parameters = $8000 + 8000 = 16,000$. We just reduced trainable parameters by 98.4%!
*   **Inference:** During inference, the learned matrices $A$ and $B$ can be multiplied together and permanently added back into the original weights ($W_{new} = W + AB$). This means **zero latency overhead** during inference.

### QLoRA (Quantized LoRA)
An extension of LoRA that enables fine-tuning even larger models on even less hardware.
*   **How it works:**
    1.  Quantize the frozen base model weights down to 4-bit precision (using NormalFloat4).
    2.  Attach 16-bit (bfloat16) LoRA adapters.
    3.  During the forward pass, the 4-bit weights are briefly dequantized to 16-bit to compute activations. Gradients are only calculated for the 16-bit LoRA adapters.
*   **Impact:** A 65B parameter model (which normally requires >130GB VRAM just to load) can be fine-tuned on a single 48GB GPU using QLoRA.

## 3. When to Fine-Tune vs. Use RAG

This is a classic interview question.

| Feature | Fine-Tuning | RAG (Retrieval-Augmented Generation) |
| :--- | :--- | :--- |
| **Adding New Knowledge** | Poor (Prone to hallucination, hard to update) | **Excellent** (Sources are explicit, easy to update DB) |
| **Teaching a New Format** | **Excellent** (e.g., output strict JSON, specific tone) | Okay (can use few-shot, but consumes context) |
| **Reducing Hallucination** | Moderate | **High** (Grounded in retrieved context) |
| **Cost to Update Info** | High (Requires retraining) | Low (Just update the vector database) |
| **Latency** | Fast (Knowledge is baked in) | Slower (Requires search + longer context reading) |

**The Golden Rule:** Use RAG to give the model *context and facts*. Use Fine-Tuning to teach the model *form, style, and behavior*. They are often used together!