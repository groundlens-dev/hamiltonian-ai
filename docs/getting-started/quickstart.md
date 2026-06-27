# Quickstart

!!! warning "Scope first"
    The optimizer trades peak accuracy for temporal stability / robustness to
    distribution shift. It is **not** a general improvement over Adam or SGD.
    Use it only when you have a temporal distribution-shift problem.

## Minimal training loop

```python
import torch.nn as nn
from hamiltonian_ai.optimizer import HamiltonianOptimizer

model = nn.Sequential(
    nn.Linear(7, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2),
)

optimizer = HamiltonianOptimizer(
    model.parameters(),
    lr=0.01,        # reference value from the credit-scoring study
    beta=0.9,       # momentum coefficient
    epsilon=1e-8,
    lambda_reg=0.01,  # temporal-stability regularization
)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
```

## Runnable example

A self-contained smoke test (synthetic data; not a benchmark) ships in the
repository:

```bash
python examples/optimizer_minimal.py
```

Next: [Optimizer usage](../optimizer/usage.md) and the
[honest benchmark](../optimizer/benchmark.md).
