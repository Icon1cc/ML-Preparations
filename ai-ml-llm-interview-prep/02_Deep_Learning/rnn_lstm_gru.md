# Recurrent Neural Networks (RNNs), LSTMs, and GRUs

Before Transformers, RNNs were the primary architecture for processing sequential data (time-series, text, audio). While mostly replaced by Transformers in NLP, they are still heavily used in time-series forecasting (like logistics demand prediction) due to their lower computational cost and ability to handle long, continuous streams.

---

## 1. The Vanilla RNN
A standard neural network processes one input and gives one output, with no memory of the past. An RNN has a "hidden state" (memory) that it passes to itself as it steps through time.

*   **Mechanism:** At time step $t$, the RNN takes two inputs: the current data point $x_t$ and the hidden state from the previous step $h_{t-1}$.
*   **Equation:** $h_t = 	anh(W_x x_t + W_h h_{t-1} + b)$
*   **The Fatal Flaw (Vanishing Gradient):** Because the *exact same* weight matrix $W_h$ is multiplied repeatedly at every time step, gradients exponentially shrink (vanish) or explode during backpropagation through time (BPTT). A vanilla RNN practically forgets anything that happened more than 5-10 steps ago.

## 2. LSTM (Long Short-Term Memory)
Introduced in 1997, LSTMs solved the vanishing gradient problem of vanilla RNNs, allowing them to learn long-term dependencies.

### The Intuition: The Conveyor Belt
Instead of a single hidden state, LSTMs have two:
1.  **Hidden State ($h_t$):** The short-term memory (what it's actively thinking about right now).
2.  **Cell State ($C_t$):** The long-term memory. Think of this as a straight conveyor belt running through the entire network. Information can flow down it unchanged unless the network explicitly decides to alter it. This "superhighway" is why gradients don't vanish.

### The Three Gates
LSTMs control the Cell State using three mathematical "gates" (usually sigmoid functions that output a value between 0 and 1):
1.  **Forget Gate:** Looks at the new input ($x_t$) and the previous hidden state ($h_{t-1}$) and decides what information to throw away from the long-term Cell State. (e.g., "The sentence just ended, forget the current subject").
2.  **Input Gate:** Decides what *new* information from $x_t$ should be added to the long-term Cell State.
3.  **Output Gate:** Decides what part of the long-term Cell State should be exposed as the current short-term Hidden State ($h_t$) to make a prediction for this specific time step.

## 3. GRU (Gated Recurrent Unit)
A newer (2014) and simplified version of the LSTM.
*   **Mechanism:** It merges the Cell State and Hidden State into a single vector. It combines the Forget and Input gates into a single **Update Gate**.
*   **Pros:** Requires fewer parameters than an LSTM (faster to train, less memory).
*   **Cons:** In some highly complex tasks, LSTMs still slightly outperform GRUs, but in most practical applications, they perform identically. Always try a GRU first as a baseline before moving to an LSTM.

## 4. Sequence-to-Sequence (Seq2Seq)
How do you use an RNN to translate English (5 words) to French (7 words)?
*   **Encoder:** An RNN reads the English sentence one word at a time. It doesn't output anything. When it reaches the end, its final Hidden State is considered the "Context Vector" (a mathematical summary of the whole sentence).
*   **Decoder:** A second RNN takes that Context Vector as its initial state and starts generating French words, one by one, until it outputs an `<END>` token.

## Interview Summary
"While I would default to a Transformer for NLP tasks, if I were building a real-time anomaly detection system for streaming IoT sensors on delivery trucks, I would likely use an LSTM or GRU. They process data sequentially, meaning they require much less memory than a Transformer's attention matrix, which is crucial for edge devices with continuous data streams."