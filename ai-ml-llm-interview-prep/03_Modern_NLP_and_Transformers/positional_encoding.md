# Positional Encoding in Transformers

The defining feature of a Transformer is that it processes all tokens in a sequence simultaneously (in parallel), rather than one by one like an RNN. While great for speed, it creates a fatal problem: **The model has no concept of word order.** To a bare Transformer, "The dog bit the man" and "The man bit the dog" are mathematically identical bags of words.

To fix this, we must explicitly inject positional information into the input embeddings.

---

## 1. Absolute Positional Encoding (The Original Paper)
Introduced in *Attention is All You Need* (2017).

### The Concept
We create a unique vector for every position in the sequence (Pos 1, Pos 2, Pos 3...) and simply **add** it to the word's embedding vector before passing it into the network.
`Final_Input = Word_Embedding + Positional_Vector`

### The Math (Sine and Cosine)
Why not just add the integer `[1, 2, 3]`? Because for long sequences, the values would grow huge and wash out the actual word embedding values. We need values bounded between -1 and 1.
The original paper uses alternating sine and cosine functions of varying frequencies.
*   Imagine the positional encoding as a binary counter (000, 001, 010, 011). The right-most bit flips constantly. The left-most bit flips rarely. 
*   The sine/cosine functions mimic this. The first few dimensions of the positional vector fluctuate rapidly (high frequency sine waves). The last few dimensions fluctuate very slowly.
*   **Why this works:** It allows the model to easily calculate relative distances. The math dictates that the encoding for position $t+k$ is a linear transformation of the encoding at position $t$.

## 2. Learned Positional Embeddings (GPT-Style)
Instead of hardcoding a sine/cosine wave, we treat positions just like words.
*   We create a separate embedding table (an array of weights) where `Row 0` represents "Position 0", `Row 1` represents "Position 1".
*   These vectors are randomly initialized and updated by backpropagation during training.
*   **Pros:** It learns exactly what it needs for the specific dataset.
*   **Cons:** It completely breaks if the model encounters a sequence longer than it was trained on (e.g., if trained on length 2048, it has no vector for Position 2049).

## 3. Relative Positional Encoding (RoPE / ALiBi)
The modern standard. Used in Llama, Mistral, and almost all state-of-the-art LLMs.

### The Intuition
In language, *absolute* position rarely matters. It doesn't matter if the word "King" is at position 5 or position 500. What matters is its *relative* position to the word "Queen" (e.g., are they 2 words apart?).

### RoPE (Rotary Positional Embeddings)
Instead of *adding* a vector to the input, RoPE modifies the Attention mechanism directly.
*   **The Mechanism:** It mathematically **rotates** the Query (Q) and Key (K) vectors in a 2D plane by an angle that is proportional to their absolute position.
*   **The Magic:** When the model calculates the Attention Score (the dot product $Q \cdot K$), the math works out so that the absolute positions cancel out, and the resulting score is *purely a function of the relative distance* between the two words.
*   **Why it's the standard:** 
    1. It captures relative distance perfectly.
    2. It allows for **Length Extrapolation** (training on 4k context windows but running inference on 32k context windows with techniques like YaRN or NTK-Aware scaling).