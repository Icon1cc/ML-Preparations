# 02 Deep Learning

## Overview

Deep Learning powers modern AI - from computer vision to natural language processing. This section covers **neural network fundamentals** that underlie everything from CNNs to Transformers.

## 📚 Contents

### Fundamentals
1. [Neural Network Basics](./neural_network_basics.md)
2. [Backpropagation](./backpropagation.md) ⭐ **Critical**
3. [Activation Functions](./activation_functions.md)
4. [Loss Functions](./loss_functions.md)
5. [Optimization Algorithms](./optimization_algorithms.md)

### Architectures
6. [CNN Architectures](./cnn_architectures.md)
7. [RNN, LSTM, GRU](./rnn_lstm_gru.md)
8. [Attention Mechanism](./attention_mechanism.md) ⭐ **Critical**

### Training Techniques
9. [Weight Initialization](./weight_initialization.md)
10. [Normalization Techniques](./normalization_techniques.md)
11. [Regularization in Deep Learning](./regularization_deep_learning.md)

### Practical
12. [PyTorch Fundamentals](./pytorch_fundamentals.md) ⭐ **Critical**
13. [Debugging Neural Networks](./debugging_neural_networks.md)

## 🎯 Learning Objectives

- Understand how neural networks learn (backpropagation)
- Implement networks in PyTorch
- Choose appropriate architectures for different tasks
- Debug and optimize training

## ⏱️ Time Estimate

**Total: 15-20 hours**

**High Priority:**
- Backpropagation: 2 hours
- PyTorch Fundamentals: 3 hours
- Neural Network Basics: 2 hours
- Activation/Loss Functions: 2 hours

**Medium Priority:**
- CNNs: 2 hours
- RNNs/LSTMs: 2 hours
- Attention: 2 hours
- Optimization: 1.5 hours

## 🔥 Interview Focus

### Most Asked

1. **"Explain backpropagation"** - See [Backpropagation](./backpropagation.md)
2. **"Why use ReLU over Sigmoid?"** - See [Activation Functions](./activation_functions.md)
3. **"How do you prevent overfitting in neural networks?"** - Regularization
4. **"Implement a simple neural network in PyTorch"** - See [PyTorch Fundamentals](./pytorch_fundamentals.md)

### Quick Answers

**"Neural network in one sentence?"**
"A neural network is a function approximator composed of stacked linear transformations followed by non-linear activations, trained via backpropagation to minimize a loss function."

**"Deep learning vs traditional ML?"**
- Traditional ML: Manual feature engineering + simple model
- Deep learning: Automatic feature learning + complex model
- DL better for: images, audio, text (unstructured data)
- Traditional better for: tabular data, small datasets, interpretability

## Key Concepts

### 1. The Basic Unit: Neuron

```
Input: x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias: b

Output: y = activation(w·x + b)
```

### 2. Forward Pass

```python
# Layer 1
h1 = activation(W1 @ x + b1)

# Layer 2
h2 = activation(W2 @ h1 + b2)

# Output
output = W3 @ h2 + b3
```

### 3. Loss Function

Measures how wrong predictions are:
- **Classification:** Cross-entropy
- **Regression:** MSE
- **Custom:** Task-specific

### 4. Backward Pass (Backpropagation)

Compute gradients via chain rule:
```
∂Loss/∂W = ∂Loss/∂output × ∂output/∂W
```

### 5. Optimization

Update weights:
```
W_new = W_old - learning_rate × gradient
```

## When to Use Deep Learning

### ✅ Use Deep Learning When:

1. **Large dataset** (>10K samples typically)
2. **Unstructured data** (images, audio, text)
3. **Complex patterns** (non-linear relationships)
4. **End-to-end learning** beneficial
5. **Compute available**

### ❌ Don't Use Deep Learning When:

1. **Small dataset** (<1K samples) → Use traditional ML
2. **Tabular data** → XGBoost often better
3. **Need interpretability** → Use linear models, trees
4. **Limited compute** → Use simpler models
5. **Real-time with strict latency** (<1ms) → Consider simpler models

## Architecture Selection

| Data Type | Best Architecture | Examples |
|-----------|------------------|----------|
| **Images** | CNN | ResNet, EfficientNet, Vision Transformer |
| **Sequences (short)** | RNN/LSTM | Time series, short text |
| **Sequences (long)** | Transformer | BERT, GPT, T5 |
| **Graphs** | GNN | Social networks, molecules |
| **Tabular** | MLP or XGBoost | Structured data |
| **Multi-modal** | Hybrid | CLIP (image+text) |

## Common Deep Learning Workflow

```python
import torch
import torch.nn as nn

# 1. Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Initialize
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(100):
    for batch_x, batch_y in dataloader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Key Techniques

### Regularization
- **Dropout:** Randomly zero neurons during training
- **L2 Weight Decay:** Penalize large weights
- **Data Augmentation:** Artificially expand dataset
- **Early Stopping:** Stop when validation plateaus

### Normalization
- **Batch Normalization:** Normalize activations
- **Layer Normalization:** Normalize across features (used in transformers)
- **Weight Normalization:** Normalize weight vectors

### Optimization Tricks
- **Learning Rate Scheduling:** Decay LR over time
- **Gradient Clipping:** Prevent exploding gradients
- **Warmup:** Gradually increase LR at start

## Common Problems & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Vanishing Gradients** | Early layers don't learn | ReLU, Batch Norm, Skip connections |
| **Exploding Gradients** | Loss becomes NaN | Gradient clipping, lower LR |
| **Overfitting** | Train good, val bad | Dropout, regularization, more data |
| **Underfitting** | Both train and val bad | Deeper network, train longer |
| **Slow Training** | Takes forever | Larger batch size, better optimizer (Adam) |

## Deep Learning vs Classical ML

| Aspect | Classical ML | Deep Learning |
|--------|-------------|---------------|
| **Data needed** | 100s-10Ks | 10Ks-millions |
| **Feature engineering** | Manual, critical | Automatic |
| **Training time** | Minutes-hours | Hours-days |
| **Interpretability** | High | Low |
| **Performance (tabular)** | Often better | Often worse |
| **Performance (images/text)** | Worse | Much better |
| **Hardware** | CPU fine | GPU preferred |

## Resources

- **PyTorch tutorials:** pytorch.org/tutorials
- **fast.ai course:** course.fast.ai (practical, top-down)
- **DeepLearning.AI:** deeplearning.ai (theory, bottom-up)
- **Papers with Code:** paperswithcode.com (SOTA models)

---

**Start with:** [Backpropagation](./backpropagation.md) (most important concept) or [PyTorch Fundamentals](./pytorch_fundamentals.md) (most practical)

**Next Section:** [03 Modern NLP and Transformers](../03_Modern_NLP_and_Transformers/README.md)
