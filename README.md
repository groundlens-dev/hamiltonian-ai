<div align="center">

<!-- logo: drop org logo here -->

# hamiltonian-ai
## Energy-geometry methods for neural systems

A symplectic optimizer for temporal stability, plus reproducible studies on the geometry of reasoning — and where geometric structure stops helping.

[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue?style=flat-square)](https://github.com/groundlens-dev/hamiltonian-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-docs.groundlens.dev%2Fhamiltonian--ai-blue?style=flat-square)](https://docs.groundlens.dev/hamiltonian-ai)
[![Papers](https://img.shields.io/badge/papers-4-purple?style=flat-square)](papers/)

[Documentation](https://docs.groundlens.dev/hamiltonian-ai) · [Optimizer](src/hamiltonian_ai/optimizer/) · [Studies](studies/) · [Papers](papers/) · [Contributing](CONTRIBUTING.md)

</div>

> **A Groundlens research line.** Part of the [Groundlens](https://github.com/groundlens-dev) family — alongside [groundlens](https://github.com/groundlens-dev/groundlens) and [otwin](https://github.com/groundlens-dev/otwin).

---

## What this is

**hamiltonian-ai** consolidates a single line of work: applying energy and
phase-space (Hamiltonian) geometry to neural systems. It is deliberately
small and deliberately bounded. It contains:

- a **library** — one packaged, reusable piece: a symplectic optimizer with a
  Hamiltonian energy view of the parameter dynamics;
- **studies** — reproducible notebooks and write-ups: a phase-space lens on
  LLM multi-hop reasoning, and a systematic negative-results study on the
  limits of geometric optimization;
- **applications** — the credit-scoring work the optimizer was validated on;
- **papers** — the underlying preprints and peer-reviewed work.

The honest one-line summary of the whole line: *geometric structure is a
useful diagnostic and a niche stability tool, not a general performance win.*
This repository is organized to make both halves of that claim visible.

## Optimizer

> **Scope first: the optimizer trades peak accuracy for temporal stability /
> robustness to distribution shift. It is not a general improvement over Adam
> or SGD, and on standard in-distribution accuracy it generally loses to them.**

The optimizer (`hamiltonian_ai.optimizer.HamiltonianOptimizer`) treats the
parameter update as a Hamiltonian system — kinetic energy from momentum,
potential energy from the parameters — and scales each step inversely with the
total energy, damping destabilizing updates. The intent is *temporal*
generalization: keeping a model's risk ranking stable as the data distribution
drifts over months.

```python
import torch.nn as nn
from hamiltonian_ai.optimizer import HamiltonianOptimizer

model = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 2))
optimizer = HamiltonianOptimizer(model.parameters(), lr=0.01, beta=0.9, lambda_reg=0.01)
# ...standard PyTorch training loop...
```

**Honest results framing (credit scoring, Freddie Mac loan-level data).** In
the IEEE DSAA 2025 study, the method's out-of-time **AUC (ranking) goes up**
while its **standard classification accuracy goes down** relative to strong
baselines. The win is in forward-looking *ranking stability*, not accuracy.

| Dimension | Adam / SGD / XGBoost | Hamiltonian optimizer |
|---|---|---|
| Standard in-distribution accuracy | **win** | loses |
| Out-of-time ranking (AUC) stability under drift | loses | **win** (this domain) |
| Training overhead | baseline | +15–20% |
| Evidence base | broad | single domain (credit scoring) |

This is a trade-off, not a free lunch. If you do not have a temporal
distribution-shift problem, use Adam or SGD.

## Reasoning diagnostics

The reasoning-geometry study maps LLM multi-hop reasoning chains into a
phase space and assigns each a Hamiltonian "energy" balancing reasoning
progression (kinetic) against question relevance (potential). The empirical
finding: **valid reasoning chains show lower and more stable Hamiltonian
energy** than invalid ones — a usable *diagnostic* signal.

**Caveat, stated as the paper states it:** the framework's claimed ability to
*steer* or *improve* reasoning is metaphorical and not empirically established.
The solid contribution is the diagnostic patterns, not a causal mechanism.

See [`studies/reasoning-geometry/`](studies/reasoning-geometry/) ·
arXiv:[2410.04415](https://arxiv.org/abs/2410.04415).

## Where it helps — and where it doesn't

This is the credibility anchor of the line, presented as a first-class result.

A separate, systematic study tested whether *generic* isotropic geometric
structure helps optimization at all — a Geometric Preconditioned Method (GPM)
across **seven** tasks: image classification, language modeling,
rotation-equivariant learning, PDE solving, continual learning, temporal
prediction, and physical operator learning. The result is negative and
statistically significant across every task: **as a general optimizer,
geometric structure consistently underperforms Adam and SGD-with-momentum.**

The central lesson is precise and worth keeping: *data geometry and loss-
landscape geometry are distinct, and the latter dominates optimization
dynamics.* If geometric structure matters for a problem, encode it in the
**architecture** (equivariant layers, spectral normalization, symplectic
networks) — not in a generic optimizer.

This is why the optimizer above is framed as a narrow, domain-specific
stability tool rather than a default. Mapping the boundary *is* the result.

See [`studies/geometry-limits/`](studies/geometry-limits/).

## Papers

| Work | Topic | Status |
|---|---|---|
| Geometric Analysis of Reasoning Trajectories | Phase-space diagnostics for multi-hop QA reasoning | arXiv:[2410.04415](https://arxiv.org/abs/2410.04415) (2024) |
| Hamiltonian NN for Out-of-Time Credit Scoring | Symplectic optimizer + Hamiltonian loss; out-of-time ranking | **Accepted (peer-reviewed), IEEE DSAA 2025** — note: not in the proceedings; preprint |
| When Geometric Structure Doesn't Help | Systematic negative-results study on geometric optimization | Working paper / preprint |

PDFs and the foundational deck are in [`papers/`](papers/).

## Repository layout

```
hamiltonian-ai/
├── src/hamiltonian_ai/optimizer/   the packaged optimizer (the reusable part)
├── studies/
│   ├── reasoning-geometry/         phase-space reasoning diagnostics (arXiv:2410.04415)
│   └── geometry-limits/            the negative-results boundary study
├── applications/credit-scoring/    IEEE DSAA 2025 notebooks (Freddie Mac OOT)
├── papers/                         preprints, peer-reviewed work, deck
├── examples/                       minimal runnable optimizer example
└── docs/                           mkdocs-material site
```

## Installation

```bash
pip install -e .
python examples/optimizer_minimal.py   # smoke test, not a benchmark
```

The reasoning-diagnostics thread informs [Groundlens](https://github.com/groundlens-dev/groundlens).

## License

MIT License — see [LICENSE](LICENSE). Author: Javier Marin.

---

An open-source research line by **Groundlens** · groundlens.dev
