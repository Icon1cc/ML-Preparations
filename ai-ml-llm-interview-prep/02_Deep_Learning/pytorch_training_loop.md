# Practical PyTorch Training Loop

## Core pipeline
1. Dataset + DataLoader
2. Model (`nn.Module`)
3. Loss function + optimizer
4. Training loop
5. Validation loop
6. Checkpointing + early stopping

## Why `train()` vs `eval()` matters
- `train()`: enables dropout, BatchNorm updates.
- `eval()`: deterministic inference behavior.

## Complete reference loop
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# toy data
X_train = torch.randn(2000, 20)
y_train = (torch.rand(2000) > 0.5).long()
X_val = torch.randn(400, 20)
y_val = (torch.rand(400) > 0.5).long()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_val = float('inf')
patience, wait = 5, 0

for epoch in range(30):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    scheduler.step()

    print(f'Epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.3f}')

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save({'model_state_dict': model.state_dict()}, 'best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping')
            break
```

## Mixed precision and gradient accumulation
```python
scaler = torch.cuda.amp.GradScaler()
accum_steps = 4

for step, (xb, yb) in enumerate(train_loader):
    with torch.cuda.amp.autocast():
        loss = criterion(model(xb), yb) / accum_steps
    scaler.scale(loss).backward()

    if (step + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Multi-GPU notes
- `DataParallel`: easy but suboptimal.
- `DistributedDataParallel`: preferred for performance.

## Common bugs
- Forgot `optimizer.zero_grad()`.
- Validation still in `train()` mode.
- Loss reduction mismatch (`mean` vs `sum`).
- Data leakage between train and validation.

## Interview questions
1. What happens if you forget `optimizer.zero_grad()`?
2. Why use `model.eval()` at validation?
3. Why gradient clipping in sequence models?
