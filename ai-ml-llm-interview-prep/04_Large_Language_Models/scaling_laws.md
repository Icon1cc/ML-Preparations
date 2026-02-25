# LLM Scaling Laws

The defining characteristic of the modern AI boom is not a specific algorithmic breakthrough, but the empirical discovery of **Scaling Laws**. Understanding these laws explains why companies are spending billions on GPU clusters.

---

## 1. The Core Premise (Kaplan et al., OpenAI 2020)
Researchers discovered that the performance of language models (measured by cross-entropy loss on next-token prediction) scales as a highly predictable **Power-Law** relationship with three factors:

1.  **Compute (C):** Total number of floating-point operations used during training.
2.  **Dataset Size (D):** Number of tokens trained on.
3.  **Model Size (N):** Number of parameters.

### The Rule of Thumb
If you increase $N$, $D$, or $C$ exponentially, the model's loss decreases linearly. This means you can reliably predict the performance of a massive 100B parameter model by training smaller 10M, 100M, and 1B models and simply drawing a straight line on a log-log plot. 
*   **Takeaway:** There are no "diminishing returns" in sight. Bigger is universally better, provided you scale everything together.

## 2. The Chinchilla Scaling Laws (Hoffmann et al., DeepMind 2022)
The original OpenAI scaling laws stated that model size ($N$) was more important than data size ($D$). 
DeepMind's "Chinchilla" paper corrected this, proving that earlier models like GPT-3 (175B parameters trained on 300B tokens) were massively **under-trained**.

### The Chinchilla Ratio
To optimally utilize a given compute budget (to get the lowest possible loss), Model Size ($N$) and Dataset Size ($D$) must be scaled equally.
*   **The Rule:** You should train on roughly **20 tokens per parameter**.
*   **Example:** If you build a 70 Billion parameter model, you must train it on at least $1.4$ Trillion tokens to reach optimal compute efficiency.

*This explains why Llama 3 (8B) destroys older 70B models. Meta trained the 8B model on a massive 15 Trillion tokens—vastly beyond the Chinchilla optimal point—resulting in a small, fast model that is incredibly smart.*

## 3. "Over-training" (The Llama Approach)
Why train past the Chinchilla optimal point?
*   Chinchilla optimizes for **Training Compute** (the cost to build the model).
*   In production, we care about **Inference Compute** (the daily cost to serve the model to millions of users).
*   By purposefully "over-training" a smaller model (like 8B parameters) on astronomically large datasets (15T tokens), it becomes as smart as a much larger model, but costs 1/10th the price to serve in production.

## 4. Emergent Abilities
A fascinating phenomenon in scaling laws. As models scale linearly in terms of loss, they suddenly exhibit sharp, discontinuous leaps in capability on specific tasks.
*   **Example:** A 1B, 5B, and 10B parameter model might all score 0% on a 3-digit math test. But suddenly, at 50B parameters, the model's accuracy shoots up to 80%. The ability to do complex reasoning "emerges" only at specific scales.

## Interview Strategy
*   If an interviewer asks: "We want to improve our model. Should we double the parameters or double the data?"
*   **Answer:** "According to the Chinchilla scaling laws, compute is best spent scaling them proportionally. We need 20 tokens per parameter. However, if our priority is cheap deployment inference, we should hold parameters steady and massively increase the training data."