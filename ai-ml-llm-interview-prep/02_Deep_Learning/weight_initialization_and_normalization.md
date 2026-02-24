# Weight Initialization and Normalization

## Why initialization matters
Poor initialization causes vanishing/exploding activations and gradients, slowing or preventing training.

## Symmetry breaking
Zero initialization makes neurons identical; gradients remain identical and network cannot learn diverse features.

## Initialization methods

### Random normal/uniform
Basic approach; scale must match fan-in/fan-out.

### Xavier/Glorot
For tanh/sigmoid-like activations.
- Normal: `Var(W) = 2 / (fan_in + fan_out)`

### He/Kaiming
For ReLU-family activations.
- Normal: `Var(W) = 2 / fan_in`

### Orthogonal initialization
Preserves variance through deep linear transforms; useful for recurrent layers.

## Normalization methods

### BatchNorm
Normalizes per mini-batch:
`x_hat = (x - mu_batch) / sqrt(var_batch + eps)`
Then learn scale/shift (`gamma`, `beta`).

Pros:
- stabilizes optimization
- allows larger learning rates

Cons:
- unstable with very small batch sizes
- train/inference behavior differs

### LayerNorm
Normalizes across features per token/sample.
Great for NLP/Transformers (batch-size independent).

### GroupNorm
Normalizes by channel groups; robust for small-batch vision.

### InstanceNorm
Per-instance per-channel normalization; common in style transfer.

### RMSNorm
Normalizes by RMS only (no mean-centering); efficient and common in modern LLMs.

## Comparison table

| Method | Normalization axis | Best for | Limitation |
|---|---|---|---|
| BatchNorm | batch dimension | CNN with moderate/large batch | batch dependence |
| LayerNorm | feature dimension | Transformers/RNNs | less conv-specific inductive bias |
| GroupNorm | channel groups | small-batch vision | group hyperparameter |
| InstanceNorm | per sample/channel | style transfer | may remove useful global contrast |
| RMSNorm | feature RMS | LLMs | less centered representation |

## Train vs eval for BatchNorm
- `model.train()`: uses batch statistics and updates running averages.
- `model.eval()`: uses running averages only.

## Interview questions
1. Why does zero initialization fail in deep nets?
2. BatchNorm vs LayerNorm in transformers?
3. Why is AdamW paired with LayerNorm-heavy LLMs?

## PyTorch examples
```python
import torch
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, d_in=128, d_h=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.ln1 = nn.LayerNorm(d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        return self.fc2(x)
```

### BatchNorm from scratch (educational)
```python
def batch_norm(x, gamma, beta, eps=1e-5):
    # x: [batch, features]
    mu = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta
```

## Practical heuristics
- ReLU networks: He init.
- Tanh networks: Xavier.
- Transformers: LayerNorm/RMSNorm, careful init + warmup.
- Small vision batches: GroupNorm.
