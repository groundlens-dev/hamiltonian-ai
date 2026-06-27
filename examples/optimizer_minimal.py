"""Minimal runnable example for the Hamiltonian optimizer.

This trains a tiny MLP on a synthetic binary classification task. It is a
smoke test of the API and the training loop, NOT a benchmark and NOT evidence
that the optimizer beats Adam/SGD. For the bounded, peer-reviewed result, see
``applications/credit-scoring/`` and the IEEE DSAA 2025 study.

Run:
    python examples/optimizer_minimal.py
"""

import torch
import torch.nn as nn

from hamiltonian_ai.optimizer import HamiltonianOptimizer


def main() -> None:
    torch.manual_seed(0)

    # Synthetic linearly-separable-ish data.
    n, d = 512, 8
    X = torch.randn(n, d)
    w_true = torch.randn(d, 1)
    y = (X @ w_true + 0.1 * torch.randn(n, 1) > 0).long().squeeze(1)

    model = nn.Sequential(
        nn.Linear(d, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    optimizer = HamiltonianOptimizer(
        model.parameters(),
        lr=0.01,
        beta=0.9,
        epsilon=1e-8,
        lambda_reg=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  acc {acc:.3f}")

    print("Done. (Smoke test only — not a benchmark.)")


if __name__ == "__main__":
    main()
