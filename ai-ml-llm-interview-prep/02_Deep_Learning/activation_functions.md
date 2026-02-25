# Activation Functions

Activation functions are mathematical equations attached to each neuron in a network. They introduce **non-linearity**, allowing the network to learn complex patterns instead of just straight lines.

---

## 1. Sigmoid (Logistic)
Maps any input to a value between 0 and 1.
*   **Equation:** $f(x) = \frac{1}{1 + e^{-x}}$
*   **Primary Use:** The **Output Layer** of a binary classification model (representing a probability).
*   **Why it's rarely used in Hidden Layers anymore:**
    1.  **Vanishing Gradients:** The derivative of Sigmoid peaks at 0.25 and approaches 0 at the tails. During backpropagation, multiplying many small gradients causes the signal to vanish, preventing early layers from learning.
    2.  **Not Zero-Centered:** Outputs are always positive, making gradient updates less efficient.

## 2. Tanh (Hyperbolic Tangent)
Maps any input to a value between -1 and 1.
*   **Equation:** $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
*   **Primary Use:** Hidden layers in older architectures (like RNNs/LSTMs).
*   **Pros/Cons:** It is zero-centered (unlike Sigmoid), which makes optimization easier. However, it still suffers heavily from the vanishing gradient problem at the tails.

## 3. ReLU (Rectified Linear Unit)
The industry standard default for almost all hidden layers in Deep Learning (CNNs, MLPs).
*   **Equation:** $f(x) = \max(0, x)$ (If negative, output 0. If positive, output $x$).
*   **Pros:**
    1.  **Solves Vanishing Gradients:** For any positive input, the derivative is exactly 1. The gradient does not shrink as it passes through layers.
    2.  **Computational Efficiency:** Extremely cheap to compute (just a max operation) compared to exponentials in Sigmoid/Tanh.
    3.  **Sparsity:** Outputs exact zeros for negative inputs, resulting in sparse, efficient representations.
*   **Cons (The "Dying ReLU" Problem):** If a large gradient pushes a bias so low that the neuron always outputs a negative number, the ReLU outputs 0, and the gradient becomes 0. The neuron is permanently "dead" and will never recover during training.

## 4. Leaky ReLU
A variant designed to fix the Dying ReLU problem.
*   **Equation:** $f(x) = \max(\alpha x, x)$ where $\alpha$ is a small constant like 0.01.
*   **Pros:** Instead of a flat 0 for negative inputs, it has a slight slope. This allows a small gradient to flow backwards, allowing "dead" neurons a chance to recover.

## 5. Softmax
The standard for Multi-Class Classification.
*   **Equation:** $f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$
*   **Primary Use:** The **Output Layer** of a multi-class network (e.g., predicting 1 of 10 digits).
*   **How it works:** It takes a vector of raw, unbounded scores (logits) from the final hidden layer and turns them into a probability distribution. Every output is between 0 and 1, and the sum of all outputs is exactly 1.

## 6. GELU (Gaussian Error Linear Unit)
The standard activation function used in modern Transformers (BERT, GPT, Llama).
*   **Concept:** It weighs inputs by their percentile in a Gaussian distribution. It behaves like a smoother version of ReLU. It allows a small amount of negative values to pass through.
*   **Why it's used in LLMs:** Empirically, it performs slightly better than ReLU in highly complex, deep architectures like Transformers, smoothing out the loss landscape.

## Interview Summary
*   **Hidden Layers:** Always start with **ReLU**. If you have dead neuron issues, try Leaky ReLU. If building an LLM, use **GELU** (or SwiGLU).
*   **Output Layer (Binary):** Sigmoid.
*   **Output Layer (Multi-class):** Softmax.
*   **Output Layer (Regression):** Linear (No activation function).