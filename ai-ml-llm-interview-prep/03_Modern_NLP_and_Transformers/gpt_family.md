# The GPT Family (Decoder-Only Models)

Generative Pre-trained Transformers (GPT) form the backbone of the generative AI revolution (ChatGPT, Llama, Claude). They are the archetype of the **Decoder-Only** Transformer architecture.

---

## 1. Core Architecture
GPT models discard the Encoder entirely. They consist solely of a stack of Transformer **Decoders**.
*   **Key Feature: Masked (Causal) Self-Attention.** When processing the word "Apple" in a sequence, the attention mechanism is physically masked so it can only look at words that came *before* "Apple". It is mathematically blinded to the future. This is strictly required for auto-regressive generation.

## 2. Pre-training: The Ultimate Simplification
Unlike BERT's complex masked tasks, GPT is trained on a single, brutally simple objective: **Next-Token Prediction**.
*   Given the context "The capital of France is", predict the next token ("Paris").
*   It does this across trillions of tokens of internet text. By simply trying to guess the next word, the model is forced to compress all of human knowledge, grammar, coding syntax, and logic into its weights.

## 3. The Generative Inference Loop (Auto-regressive)
How does ChatGPT write a paragraph?
1.  **Prompt:** You input: `Write a poem about dogs.`
2.  **Pass 1:** The model processes the prompt and predicts the highest probability next token: `The`.
3.  **Append:** We take `The` and append it to the prompt.
4.  **Pass 2:** The model processes `Write a poem about dogs. The` and predicts `furry`.
5.  **Loop:** This repeats over and over, generating one token at a time, until the model explicitly predicts a special `<EOS>` (End of Sequence) token.
*   *Note on Latency:* This sequential process is why LLMs are inherently slow to generate text. The model must run a full forward pass for every single token generated.

## 4. The Evolution of the Paradigm

### GPT-1 & GPT-2 (The Zero-Shot Realization)
Proved that language models trained simply to predict the next word could perform tasks (like translation or summarization) without being explicitly fine-tuned for them, just by changing the input prompt (Zero-Shot learning).

### GPT-3 (The Scaling Laws)
Proved that simply making the model massive (175 Billion parameters) and throwing more data at it caused "Emergent Abilities" to appear. It could do complex Few-Shot reasoning. *However, GPT-3 was a "Base Model." It just completed text. If you asked it a question, it might answer with another question.*

### InstructGPT & ChatGPT (The RLHF Revolution)
Base models are useless for products. OpenAI introduced **RLHF (Reinforcement Learning from Human Feedback)** to fine-tune the Base Model into an "Assistant" that follows instructions, formats answers nicely, and refuses harmful requests. (See LLM Training Pipeline).

## 5. Modern Open-Weight Equivalents
While "GPT" is an OpenAI trademark, the exact same Decoder-Only architecture is used by:
*   **Llama 3 (Meta):** The current king of open-source. Uses modern tricks like RoPE (Rotary Positional Embeddings), GQA (Grouped Query Attention) for faster inference, and SwiGLU activations.
*   **Mistral:** Introduced Sliding Window Attention and Mixture of Experts (MoE) to open-weights, allowing for high performance with very low active parameter counts during inference.

## Interview Summary
*   **Use GPT/Decoder models when:** You need to *generate* novel text, write code, act as a conversational agent, or perform complex multi-step reasoning (Chain-of-Thought).
*   **Do NOT use GPT when:** You only need to do simple, high-speed binary classification (e.g., spam detection) on edge devices. A 7B parameter LLM is massive overkill; use DistilBERT or XGBoost instead.