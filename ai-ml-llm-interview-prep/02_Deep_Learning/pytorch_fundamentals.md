# PyTorch Fundamentals for AI Engineering

PyTorch is the dominant framework for research and production AI. Understanding its core mechanics is essential for any deep learning role.

---

## 1. Tensors: The Building Blocks
A Tensor is a multi-dimensional array, similar to a NumPy array, but with two superpowers:
1.  **GPU Acceleration:** Tensors can be moved to GPUs for massively parallel computation.
2.  **Automatic Differentiation:** PyTorch tracks every operation performed on a tensor to calculate gradients automatically.

```python
import torch

# Create a tensor and tell PyTorch to track its gradients
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2 + 5
z = y.mean()

z.backward() # Backpropagation!
print(x.grad) # Prints the derivative dz/dx
```

## 2. The `nn.Module` Pattern
Every neural network in PyTorch is a class that inherits from `torch.nn.Module`.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Define the forward pass logic
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

## 3. The Standard Training Loop
You must be able to describe (or write) this sequence by heart:

1.  **Forward Pass:** Pass data through model to get predictions.
2.  **Calculate Loss:** Compare predictions to true labels using a loss function (e.g., `nn.CrossEntropyLoss()`).
3.  **Zero Gradients:** Clear old gradients from the previous step (`optimizer.zero_grad()`).
4.  **Backward Pass:** Calculate new gradients via backpropagation (`loss.backward()`).
5.  **Optimizer Step:** Update model weights based on gradients (`optimizer.step()`).

## 4. Data Loading: `Dataset` and `DataLoader`
Production ML requires efficient data streaming so the GPU isn't sitting idle.
*   **`Dataset`:** A class that defines *how* to load a single item (e.g., read an image from disk, apply transforms).
*   **`DataLoader`:** A wrapper that handles batching, shuffling, and multi-process data loading.

## 5. PyTorch Ecosystem Components
*   **`torchvision` / `torchaudio`:** Specialized libraries for CV and Audio.
*   **`torch.optim`:** Optimizers like SGD, Adam, AdamW.
*   **`torch.device`:** Managing CPU vs GPU (`"cuda"` or `"mps"` for Mac).

## 6. Interview Tip: `model.train()` vs `model.eval()`
This is a very common "gotcha" question.
*   **`model.train()`:** Activates layers like **Dropout** and **Batch Normalization** which behave differently during training (e.g., Dropout randomly zeros out neurons).
*   **`model.eval()`:** Deactivates these behaviors for inference. *Always* call this before running validation or testing, or your results will be inconsistent!
*   **`with torch.no_grad():`:** Use this during evaluation to disable gradient tracking, which saves massive amounts of memory and increases speed.