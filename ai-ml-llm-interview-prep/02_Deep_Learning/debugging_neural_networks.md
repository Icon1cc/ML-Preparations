# Debugging Neural Networks

When a traditional software program fails, it throws an error. When a neural network fails, it simply outputs bad predictions silently. Debugging ML is fundamentally different. This is a highly tested area for Senior AI Engineers.

---

## 1. The Model Doesn't Train (Loss is stuck or NaN)

### A. Loss becomes NaN (Not a Number)
*   **Cause 1: Exploding Gradients.** Learning rate is far too high, or you lack Gradient Clipping.
*   **Cause 2: Division by Zero / Log(0).** Often happens in custom loss functions (e.g., passing exactly 0 or 1 to a Cross-Entropy log function). Add a tiny epsilon (`1e-8`) to denominators/logs.
*   **Cause 3: Dirty Data.** A single `NaN` or `Infinity` value in your input features or labels will instantly poison the entire network.

### B. Loss stays completely flat from Epoch 1
*   **Cause 1: Learning Rate is microscopic.** The steps are so small they don't register.
*   **Cause 2: Dead Neurons (ReLU).** If using a high learning rate initially, a massive gradient update might push all biases negative, permanently killing all ReLU neurons. The gradient becomes 0. Switch to Leaky ReLU or lower LR.
*   **Cause 3: Initialization Failure.** All weights initialized to 0.

## 2. The Model Trains, but Performance is Terrible

### A. The "Overfit a Single Batch" Test (The Golden Rule)
Before training on 1 million rows, take **10 rows** of data, turn off regularization (no dropout, no weight decay), and train for 500 epochs.
*   **If the loss DOES NOT hit 0.00:** Your code is fundamentally broken. Your architecture is flawed, your data loading is scrambled (e.g., labels don't match images), or your loss function is implemented wrong.
*   **If the loss DOES hit 0.00:** Your core logic is sound. The issue lies in generalization, data scale, or hyperparameters.

### B. High Training Loss, High Validation Loss (Underfitting)
*   **Model Capacity:** The network is too small (needs more layers/neurons).
*   **Regularization:** You have too much Dropout or Weight Decay. Turn them off.
*   **Data Scaling:** You forgot to scale your inputs (StandardScaler), causing the optimizer to struggle with oblong loss surfaces.

### C. Low Training Loss, High Validation Loss (Overfitting)
*   **Data Leakage:** Are you accidentally training on your test data?
*   **Need Regularization:** Add Dropout, Early Stopping, or L2 Weight Decay.
*   **More Data:** You need data augmentation or simply a larger dataset.

## 3. Silent Failures (The hardest bugs)

*   **Forgetting `model.train()` and `model.eval()`:** Running validation while Dropout is active will cause validation scores to look artificially terrible.
*   **Forgetting `optimizer.zero_grad()`:** PyTorch *accumulates* gradients by default. If you don't zero them out at the start of a training loop, step 2's gradients will be added to step 1's, causing chaotic, explosive updates.
*   **Softmax vs LogSoftmax:** PyTorch's `nn.CrossEntropyLoss` *automatically* applies Softmax internally. If you manually apply Softmax to the final layer of your network and then pass it to `CrossEntropyLoss`, you are applying it twice, ruining the math.

## Interview Strategy
If asked "How do you debug a model that isn't learning?":
1.  "First, I check the data for NaNs and ensure normalization was applied correctly."
2.  "Second, I attempt to **overfit a single batch**. If it can't memorize 10 examples, I know my architecture or loss function has a code bug."
3.  "Third, I check gradient flow. I print the mean and variance of the gradients at the first and last layers to check for vanishing or exploding gradients."