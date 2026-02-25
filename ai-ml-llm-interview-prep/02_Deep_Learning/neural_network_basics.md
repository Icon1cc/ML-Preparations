# Neural Network Basics

A neural network is a mathematical function approximator inspired by the biological brain. It learns to map inputs to outputs by adjusting interconnected weights through exposure to data.

---

## 1. The Perceptron (The Single Neuron)
The fundamental building block. It performs a simple mathematical operation, identical to Logistic Regression.
1.  **Inputs ($X$):** The data features (e.g., $x_1, x_2, x_3$).
2.  **Weights ($W$) & Bias ($b$):** Each input is multiplied by a learnable weight. A bias term is added to shift the result.
    $$Z = (w_1x_1 + w_2x_2 + w_3x_3) + b$$
3.  **Activation Function:** The linear result $Z$ is passed through a non-linear function (like Sigmoid or ReLU) to determine the neuron's final output (activation).

## 2. Multi-Layer Perceptron (MLP) / Feedforward Network
A single neuron can only draw a straight line. To solve complex, non-linear problems, we stack neurons into layers.
*   **Input Layer:** Passes the raw data into the network. (Not considered a "computational" layer).
*   **Hidden Layers:** Layers between input and output. A network with more than one hidden layer is considered "Deep Learning." Each layer learns increasingly abstract representations of the data.
*   **Output Layer:** Produces the final prediction (e.g., 1 node for regression/binary classification, $N$ nodes for multi-class classification).

## 3. Why Non-Linear Activation Functions?
This is a critical interview question.
*   **Question:** What happens if you build a 100-layer neural network but use a linear activation function (or no activation function) for every neuron?
*   **Answer:** The entire 100-layer network collapses mathematically into a single linear regression model. A linear function of a linear function is just another linear function. Without non-linear activations (like ReLU, Tanh), deep learning is impossible; the network could never learn curves, circles, or complex patterns.

## 4. The Learning Process (High-Level)
1.  **Initialization:** Weights and biases are set to small random numbers.
2.  **Forward Pass:** Data flows from input to output. The network makes a prediction ($\hat{y}$).
3.  **Loss Calculation:** The prediction is compared to the true label ($y$) using a Loss Function (e.g., Mean Squared Error or Cross-Entropy). This yields a scalar error value.
4.  **Backward Pass (Backpropagation):** The calculus phase. The network calculates the gradient (derivative) of the Loss with respect to every single weight in the network using the Chain Rule. It asks: "If I increase this specific weight slightly, does the error go up or down?"
5.  **Optimization:** The Optimizer (e.g., SGD, Adam) takes the gradients and updates all the weights simultaneously to reduce the error.
6.  **Epochs:** Steps 2-5 are repeated for the entire dataset many times (epochs) until the loss converges.

## 5. Network Capacity and Overfitting
*   **Capacity:** The number of neurons and layers dictates the network's capacity. A network with massive capacity can memorize the entire training dataset, leading to severe overfitting.
*   **Regularization:** Because MLPs are so prone to overfitting, techniques like **Dropout** (randomly disabling neurons during training), **L2 Weight Decay**, and **Early Stopping** are mandatory in practice.

## Interview Tip: "Universal Approximation Theorem"
This theorem states that a feed-forward network with a linear output layer and at least one hidden layer with any "squashing" activation function can approximate any continuous function to any desired degree of accuracy, provided the network has enough hidden neurons. It proves mathematically that Neural Networks can solve almost any pattern recognition problem given enough data and compute.