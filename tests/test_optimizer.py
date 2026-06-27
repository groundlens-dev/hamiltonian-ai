"""Minimal test suite for the Hamiltonian optimizer.

These are smoke + sanity tests, not benchmarks. They verify the API, parameter
validation, that a step actually reduces a convex loss, and state round-tripping.
They require torch (a runtime dependency) and run in CI.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from hamiltonian_ai.optimizer import (  # noqa: E402
    HamiltonianOptimizer,
    HamiltonianOptimizerAdvanced,
)


def _toy_problem(seed: int = 0):
    """A small, well-conditioned logistic-regression problem."""
    torch.manual_seed(seed)
    n, d = 256, 8
    X = torch.randn(n, d)
    w = torch.randn(d, 1)
    y = (X @ w + 0.1 * torch.randn(n, 1) > 0).long().squeeze(1)
    model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
    return model, X, y


def _train(optimizer, model, X, y, steps: int = 60) -> tuple[float, float]:
    crit = nn.CrossEntropyLoss()
    first = last = None
    for i in range(steps):
        optimizer.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        optimizer.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    return first, last


def test_imports_and_construction():
    model, _, _ = _toy_problem()
    assert HamiltonianOptimizer(model.parameters())
    assert HamiltonianOptimizerAdvanced(model.parameters(), symplectic_order=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"beta": 1.0},
        {"beta": -0.1},
        {"epsilon": -1e-8},
        {"lambda_reg": -1.0},
        {"weight_decay": -0.1},
    ],
)
def test_parameter_validation_raises(kwargs):
    model, _, _ = _toy_problem()
    with pytest.raises(ValueError):
        HamiltonianOptimizer(model.parameters(), **kwargs)


def test_step_reduces_convex_loss():
    model, X, y = _toy_problem()
    opt = HamiltonianOptimizer(model.parameters(), lr=0.05, beta=0.9, lambda_reg=0.0)
    first, last = _train(opt, model, X, y)
    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"


def test_advanced_optimizer_runs_and_reduces_loss():
    model, X, y = _toy_problem()
    opt = HamiltonianOptimizerAdvanced(model.parameters(), lr=0.05, symplectic_order=4)
    first, last = _train(opt, model, X, y, steps=40)
    assert last < first


def test_state_dict_roundtrip():
    model, X, y = _toy_problem()
    opt = HamiltonianOptimizer(model.parameters(), lr=0.05)
    _train(opt, model, X, y, steps=5)
    state = opt.state_dict()
    opt2 = HamiltonianOptimizer(model.parameters(), lr=0.05)
    opt2.load_state_dict(state)
    assert opt2.state_dict()["param_groups"][0]["lr"] == 0.05
