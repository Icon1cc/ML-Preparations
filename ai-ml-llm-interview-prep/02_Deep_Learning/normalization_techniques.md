# Normalization Techniques in Deep Learning

Normalization layers are critical components in modern neural networks. They stabilize the training process, allow for higher learning rates, and act as a form of mild regularization.

---

## 1. The Core Concept: Internal Covariate Shift
As data flows through a deep network, the weights of the first layer change. This changes the distribution of the inputs to the second layer. The second layer is constantly trying to chase a moving target. This phenomenon is called **Internal Covariate Shift**.
*   **The Solution:** At every layer, physically force the activations to have a mean of $0$ and a variance of $1$ before passing them to the activation function.

## 2. Batch Normalization (BatchNorm)
The standard normalization for **Convolutional Neural Networks (CNNs)**.
*   **How it works:** For a specific feature channel, it calculates the mean and variance across the *entire batch* of images currently being processed. It scales the data using this batch mean/variance.
*   **Learnable Parameters:** After forcing the data to Mean=0, Var=1, it multiplies by a learnable parameter ($\gamma$) and adds a learnable parameter ($\beta$). This allows the network to undo the normalization if it decides the original distribution was actually optimal.
*   **The Flaw:** It is highly dependent on Batch Size. If you have a small batch size (e.g., $N=2$ due to massive 3D medical images), the calculated mean/variance is extremely noisy, and BatchNorm fails completely. It also breaks down in RNNs/Transformers because sequence lengths vary.

## 3. Layer Normalization (LayerNorm)
The standard normalization for **Transformers (LLMs) and RNNs**.
*   **How it works:** Instead of calculating statistics across the batch for a single feature, it calculates the mean and variance across *all features for a single specific data point* (e.g., a single token in a sentence).
*   **Pros:** It is completely independent of the batch size. It works perfectly for sequential data of varying lengths.

## 4. Pre-Norm vs. Post-Norm (Transformer Architecture)
A classic interview question regarding Transformer design.
*   **Post-Norm (Original "Attention is All You Need"):** The normalization happens *after* the residual addition: `LayerNorm(x + Sublayer(x))`.
    *   *Issue:* Gradients can become unstable near the output layer. Requires a complex "Learning Rate Warmup" schedule to prevent the model from diverging early in training.
*   **Pre-Norm (Modern LLMs like GPT, Llama):** The normalization happens *before* the sublayer: `x + Sublayer(LayerNorm(x))`.
    *   *Benefit:* The main gradient superhighway (`x`) is completely untouched by normalization. This results in incredibly stable training, allowing for deeper networks and removing the strict necessity for learning rate warmup.

## 5. RMSNorm (Root Mean Square Normalization)
Used in the newest state-of-the-art LLMs (like Llama 3).
*   **Concept:** Researchers realized that the *mean-centering* part of LayerNorm wasn't actually doing much; the *variance-scaling* was the important part.
*   **Mechanism:** RMSNorm drops the mean calculation entirely. It only divides the inputs by their Root Mean Square.
*   **Benefit:** Computationally much cheaper than LayerNorm (saves roughly 10% of training time) with identical model performance.