# Optimization Basics

Optimization is the engine of machine learning. Training a model fundamentally means finding the set of parameters (weights and biases) that minimize a specific Loss Function.

---

## 1. Gradient Descent (The Foundation)
Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.

### The Algorithm
1.  Initialize weights randomly.
2.  Calculate the gradient (derivative) of the Loss Function with respect to the weights. The gradient points in the direction of the steepest *ascent*.
3.  Take a step in the *opposite* direction of the gradient to reduce the loss.
4.  Update weights: $W_{new} = W_{old} - (\alpha 	imes 	ext{Gradient})$, where $\alpha$ is the Learning Rate.
5.  Repeat until convergence.

### Learning Rate ($\alpha$)
The most critical hyperparameter in optimization.
*   **Too large:** The algorithm overshoots the minimum, bouncing back and forth across the valley, failing to converge (divergence).
*   **Too small:** The algorithm takes tiny steps. It takes forever to train and is highly likely to get stuck in shallow local minima.

## 2. Variations of Gradient Descent

### A. Batch Gradient Descent
Calculates the gradient using the *entire* training dataset before taking a single step.
*   **Pros:** Smooth, stable convergence trajectory.
*   **Cons:** Incredibly slow. If the dataset doesn't fit in memory, it's impossible to compute.

### B. Stochastic Gradient Descent (SGD)
Calculates the gradient using a *single* random data point per step.
*   **Pros:** Extremely fast steps. The noise (erratic path) can help it jump out of local minima.
*   **Cons:** Very noisy convergence. Never truly settles at the absolute minimum; it wanders around it.

### C. Mini-Batch Gradient Descent
The industry standard. Calculates the gradient using a small batch of data (e.g., 32, 64, or 256 samples).
*   **Pros:** Balances the speed of SGD with the stability of Batch GD. Highly optimized for matrix multiplication on GPUs.

## 3. Advanced Optimizers (Adaptive Learning Rates)
Standard SGD uses the exact same learning rate for all parameters. This is inefficient if some features are sparse and others are dense.

### A. Momentum
*   **Concept:** "A ball rolling down a hill." It adds a fraction of the previous update vector to the current update vector.
*   **Benefit:** Helps accelerate gradients in the right direction, dampens oscillations, and pushes the model past shallow local minima.

### B. RMSprop
*   **Concept:** Keeps a moving average of the squared gradients for each weight. It divides the learning rate by this average.
*   **Benefit:** If a weight has huge gradients, its learning rate shrinks. If a weight has tiny gradients, its learning rate grows. This automatically adapts the learning rate per parameter.

### C. Adam (Adaptive Moment Estimation)
The default optimizer for 95% of Deep Learning and LLM training.
*   **Concept:** It combines the best of both worlds: It uses **Momentum** (first moment, moving average of gradients) AND **RMSprop** (second moment, moving average of squared gradients).
*   **Benefit:** Fast convergence, requires very little hyperparameter tuning, handles sparse gradients exceptionally well.

## 4. Convex vs. Non-Convex Functions
*   **Convex:** Bowl-shaped. There is only one global minimum. (e.g., Linear Regression, Logistic Regression). Gradient descent is guaranteed to find the best possible solution.
*   **Non-Convex:** Mountains and valleys. Has many local minima and saddle points. (e.g., All Deep Neural Networks). We are never guaranteed to find the absolute best weights, only a "good enough" local minimum.

## Interview Tip: "When Adam fails"
While Adam is the default for deep learning, for traditional models (like Linear Regression) or when you need absolute maximum generalization on simple CV datasets (like CIFAR-10), a well-tuned **SGD with Momentum** often yields slightly better final accuracy on the test set than Adam, though it takes longer to tune the learning rate schedule.