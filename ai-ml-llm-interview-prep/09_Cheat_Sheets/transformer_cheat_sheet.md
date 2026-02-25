# Transformer Architecture Cheat Sheet

A rapid-fire reference for the Transformer architecture (Vaswani et al., 2017).

---

## 1. High-Level Concept
Transformers replaced Recurrent Neural Networks (RNNs) by processing all tokens in a sequence *simultaneously* (parallelization) rather than sequentially, relying entirely on the "Attention" mechanism to build context.

## 2. Core Components

### A. Input Embeddings & Positional Encoding
*   **Tokenization:** Text is split into sub-words (tokens).
*   **Input Embedding:** Each token is mapped to a dense vector (e.g., dimension 512).
*   **Positional Encoding:** Because Transformers process everything at once, they have no inherent concept of word order. We add a positional vector (often using Sine/Cosine functions) to the input embedding so the model knows where the word is in the sentence.

### B. The Attention Mechanism (The Heart)
*   **Self-Attention:** Allows each word to look at every other word in the sentence to understand context (e.g., figuring out what "it" refers to).
*   **Q, K, V Matrices:**
    *   **Query (Q):** What I am looking for.
    *   **Key (K):** What I have to offer.
    *   **Value (V):** The actual information I contain.
*   **The Math:** `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`
    *   The dot product of $Q$ and $K$ creates an "attention score" (how much focus word A should put on word B).
    *   Divide by `sqrt(d_k)` to stabilize gradients.
    *   `Softmax` turns scores into probabilities (summing to 1).
    *   Multiply by $V$ to get the final context-aware vector.
*   **Multi-Head Attention:** Instead of one attention mechanism, we run $H$ (e.g., 8) mechanisms in parallel. This allows the model to simultaneously focus on different relationships (e.g., one head focuses on grammar, another on semantic meaning).

### C. Feed-Forward Neural Network (FFN)
*   After attention, the vector passes through a standard fully connected network (usually two linear layers with a ReLU/GELU activation in between). This applies non-linear transformations and "memorizes" facts.

### D. Add & Norm
*   **Residual Connections (Add):** The input to a sub-layer is added to its output (`x + Sublayer(x)`). This solves the vanishing gradient problem in deep networks.
*   **Layer Normalization (Norm):** Normalizes the outputs to stabilize training.

## 3. Architecture Variations

### Encoder-Only (e.g., BERT)
*   Uses bidirectional attention (looks at words before and after).
*   **Goal:** Deep understanding of the text.
*   **Use Cases:** Text classification, sentiment analysis, generating embeddings for RAG.

### Decoder-Only (e.g., GPT, Llama, Mistral)
*   Uses *Masked* Self-Attention. It can only look at words *before* it, not after it (to prevent cheating during training).
*   **Goal:** Text generation (predicting the next token).
*   **Use Cases:** Chatbots, code generation, most modern LLMs.

### Encoder-Decoder (e.g., T5, BART, Original Transformer)
*   Encoder reads the input, Decoder generates the output.
*   **Goal:** Sequence-to-Sequence mapping.
*   **Use Cases:** Machine translation, text summarization.