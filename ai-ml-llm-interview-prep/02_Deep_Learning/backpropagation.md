# Backpropagation and Computational Graphs

Backpropagation is the algorithm that allows neural networks to learn. It is how we calculate the gradient of the loss function with respect to every single weight in the network, so the optimizer (like Adam or SGD) knows which way to adjust the weights to reduce the error.

---

## 1. The Core Concept: The Chain Rule
Backpropagation is simply the application of the Chain Rule of calculus from the end of the network back to the beginning.

If $y = f(u)$ and $u = g(x)$, the chain rule states:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

In a neural network:
1.  **Forward Pass:** We pass input data $X$ through the network to get a prediction $\hat{y}$. We calculate the Loss (e.g., Mean Squared Error between $\hat{y}$ and true $y$).
2.  **Backward Pass:** We want to know: "How does a tiny change in Weight $W_1$ (which is deep in the network) affect the final Loss?"
    $$\frac{\partial Loss}{\partial W_1} = \frac{\partial Loss}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Layer_2} \cdot \frac{\partial Layer_2}{\partial Layer_1} \cdot \frac{\partial Layer_1}{\partial W_1}$$
    We calculate these local gradients starting from the end (the Loss) and multiply them backwards through the network.

## 2. Computational Graphs
Deep learning frameworks (like PyTorch and TensorFlow) do not calculate gradients by hand. They build a directed acyclic graph (DAG) during the forward pass.
*   **Nodes:** Represent mathematical operations (addition, multiplication, ReLU).
*   **Edges:** Represent the flow of data (tensors).
*   **Autograd:** Because PyTorch knows the derivative of every basic operation (e.g., the derivative of $x^2$ is $2x$, the derivative of ReLU is 1 if $x>0$ else 0), it simply traverses the graph backwards, applying the chain rule automatically.

## 3. The Vanishing Gradient Problem
This is the historical reason why deep neural networks (many layers) were impossible to train before 2012.

*   **The Cause:** Look at the chain rule equation above. It involves multiplying many gradients together. If you use an activation function like Sigmoid or Tanh, the maximum derivative is relatively small (max 0.25 for Sigmoid).
*   **The Effect:** When you multiply many numbers smaller than 1 together, the result exponentially decays to zero. $\frac{\partial Loss}{\partial W_1}$ becomes $0.00000001$.
*   **The Result:** The layers closest to the input receive essentially zero gradient signal. They don't learn. The network "vanishes."

### Solutions to Vanishing Gradients:
1.  **ReLU Activation:** The derivative of ReLU is exactly 1 for all positive numbers. Multiplying by 1 does not shrink the gradient!
2.  **Residual Connections (ResNets / Transformers):** Adding a "skip connection" ($x + F(x)$). During backpropagation, the gradient flows directly through the '$x$' path completely unaltered, acting as a super-highway for gradients to reach the early layers.
3.  **Batch Normalization / Layer Normalization:** Normalizing activations keeps them out of the "saturated" (flat) regions of activation functions where gradients die.
4.  **Better Initialization (He / Xavier):** Initializing weights carefully so that the variance of the inputs isn't crushed or exploded as it passes through layers.

## 4. Exploding Gradients
The opposite problem. If the local gradients are $>1$, multiplying them together leads to infinity. The model weights update too drastically, resulting in `NaN` (Not a Number) errors.
*   **Solution:** Gradient Clipping (literally capping the maximum value of the gradient to a threshold like 1.0 before applying the optimizer update). Highly common in training LSTMs and large Transformers.