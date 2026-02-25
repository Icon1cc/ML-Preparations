# Weight Initialization

How you set the initial weights of a neural network before training begins is critically important. Bad initialization leads to exploding or vanishing gradients, causing the network to never learn.

---

## 1. The Naive Approaches (What NOT to do)

### A. Initializing all weights to Zero
*   **The Problem:** If all weights are zero, every neuron in a hidden layer performs the exact same calculation. During backpropagation, they all receive the exact same gradient update. The network fails to break **symmetry**. A 1000-neuron layer acts like a 1-neuron layer.

### B. Initializing with large random numbers
*   **The Problem:** If weights are too large, the outputs of the matrix multiplications explode. If using Sigmoid/Tanh activations, large numbers push the activation into the flat "tails," causing gradients to instantly vanish.

## 2. Xavier (Glorot) Initialization
Designed specifically for networks using **Sigmoid or Tanh** activation functions.
*   **The Goal:** We want the variance of the outputs of a layer to be equal to the variance of its inputs. This keeps the signal from exploding or vanishing as it passes through the network.
*   **The Math:** We initialize weights from a normal distribution with a mean of 0 and a variance of $\frac{1}{n_{in}}$, where $n_{in}$ is the number of input neurons to that layer (fan-in).
    *   *Normal Distribution:* $W \sim N(0, \frac{1}{n_{in}})$
    *   *Uniform Distribution:* $W \sim U(-\sqrt{\frac{3}{n_{in}}}, \sqrt{\frac{3}{n_{in}}})$

## 3. He (Kaiming) Initialization
Designed specifically for networks using **ReLU or Leaky ReLU** activation functions.
*   **The Problem with Xavier on ReLU:** ReLU zeros out half of the inputs (all negative numbers). This halves the variance of the signal. If you use Xavier with ReLU, the variance shrinks by half at every layer, eventually vanishing.
*   **The Math:** To compensate for ReLU zeroing out half the data, He initialization doubles the variance of Xavier.
    *   *Normal Distribution:* $W \sim N(0, \frac{2}{n_{in}})$

## 4. Modern LLM Initialization
Transformers have very specific initialization needs to maintain stability across 100+ layers.
*   **Standard approach:** A variation of Xavier initialization.
*   **Small Init for Residuals:** A crucial trick in GPT-style models is initializing the weights of the final projection layer *inside* a residual block to near-zero (e.g., scaled by $\frac{1}{\sqrt{2 \cdot 	ext{num\_layers}}}$). This ensures that at the very beginning of training, the residual blocks act almost like identity functions ($x + 0 = x$), making the deep network behave like a shallow one, which is much easier to start training.

## Summary Checklist
*   **Sigmoid / Tanh:** Use **Xavier/Glorot** Initialization.
*   **ReLU / Leaky ReLU:** Use **He/Kaiming** Initialization.
*   **Biases:** Usually initialized to exactly $0$ (except sometimes in LSTMs/ReLUs where a small positive bias like $0.01$ is used to prevent dead neurons early on).