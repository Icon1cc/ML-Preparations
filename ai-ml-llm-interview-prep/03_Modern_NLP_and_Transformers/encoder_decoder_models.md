# Encoder-Decoder Models (Sequence-to-Sequence)

While BERT (Encoder-only) handles understanding and GPT (Decoder-only) handles generation, the original Transformer architecture was an **Encoder-Decoder**, designed to map one sequence into a completely different sequence.

---

## 1. Core Architecture

The model consists of two distinct halves that communicate with each other.

### The Encoder
*   Reads the entire input sequence (e.g., an English sentence).
*   Uses *Bidirectional* Self-Attention (can look forward and backward).
*   Output: Creates a rich, context-aware matrix of vectors representing the deep meaning of the input text.

### The Decoder
*   Generates the output sequence (e.g., a French sentence) auto-regressively, one token at a time.
*   Uses *Masked* Self-Attention (can only look at the French words it has already generated, not future ones).
*   **Crucial Addition: Cross-Attention.** In addition to looking at its own past generated words, every layer of the Decoder features a "Cross-Attention" block. Here, the Decoder's **Query** looks back at the **Keys and Values** provided by the *Encoder's final output*. This is how the Decoder knows *what* it is supposed to be translating.

## 2. Famous Encoder-Decoder Models

### T5 (Text-to-Text Transfer Transformer) by Google
*   **The Philosophy:** T5 treats *every* NLP problem as a text-to-text problem.
*   Instead of having a classification head for sentiment and a separate architecture for translation, T5 relies on prefixes in the prompt.
    *   *Input:* `translate English to German: That is good.` $
ightarrow$ *Output:* `Das ist gut.`
    *   *Input:* `cola sentence: The course is jumping well.` $
ightarrow$ *Output:* `unacceptable` (grammar checking).
    *   *Input:* `summarize: [Long text...]` $
ightarrow$ *Output:* `[Short text]`

### BART (Bidirectional and Auto-Regressive Transformers) by Facebook
*   **Pre-training:** It acts like a combination of BERT and GPT. It takes a corrupted document (words masked, deleted, or permuted) and forces the Encoder-Decoder to reconstruct the pristine, original document.
*   **Use Cases:** It is exceptional at abstractive summarization.

### Whisper (OpenAI)
*   Encoder-Decoder models aren't just for text. Whisper is an Audio-to-Text model.
*   *Encoder:* Reads the audio spectrogram and extracts acoustic features.
*   *Decoder:* Uses cross-attention to look at the audio features and generates the text transcript.

## 3. Why are LLMs moving away from this?
In the modern era (2022+), almost all massive LLMs (GPT-4, Llama 3) are Decoder-Only. Why did we abandon the Encoder?

*   **Simplicity and Scaling:** Decoder-only models are structurally simpler. It is much easier to optimize the training loop and scale them across thousands of GPUs without dealing with the cross-attention bottleneck between two different network stacks.
*   **Zero-Shot Generalization:** While an Encoder-Decoder (like T5) is highly efficient for specific fine-tuned tasks (like Translation), massive Decoder-only models proved to have better "few-shot" reasoning capabilities across unseen tasks. A sufficiently large Decoder model can effectively "encode" the context within its massive prompt window.

## Interview Summary
*   **When to use Encoder-Decoder:** When the task is strictly transforming one sequence into another, especially where the input text is relatively bounded, and you have training data to fine-tune it. **Summarization** and **Translation** are the gold standard use cases for models like T5 or BART, often performing better than similarly sized Decoder-only models.