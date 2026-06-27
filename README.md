<div align="center">

<!-- logo: drop org logo here -->

# hamiltonian-ai
## Energy-geometry methods for neural systems

A symplectic optimizer for temporal stability, plus reproducible studies on the geometry of reasoning — and a systematic map of where geometric structure stops helping.

[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue?style=flat-square)](https://github.com/groundlens-dev/hamiltonian-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-docs.groundlens.dev%2Fhamiltonian--ai-blue?style=flat-square)](https://docs.groundlens.dev/hamiltonian-ai)
[![arXiv](https://img.shields.io/badge/arXiv-2410.04415-b31b1b?style=flat-square)](https://arxiv.org/abs/2410.04415)
[![Status](https://img.shields.io/badge/status-research%20line-orange?style=flat-square)](#scope)

[Overview](#overview) · [Install](#installation) · [Quickstart](#quickstart) · [Optimizer](#the-optimizer) · [Studies](#studies) · [Scope](#scope) · [Papers](#papers) · [Docs](https://docs.groundlens.dev/hamiltonian-ai)

</div>

> **A Groundlens research line.** Part of the [Groundlens](https://github.com/groundlens-dev) family — alongside [groundlens](https://github.com/groundlens-dev/groundlens) (geometric LLM verification) and [otwin](https://github.com/groundlens-dev/otwin) (physics-informed digital twins).

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The optimizer](#the-optimizer)
  - [How it works](#how-it-works)
  - [API](#api)
  - [Results, in scope](#results-in-scope)
- [Studies](#studies)
  - [Reasoning geometry](#reasoning-geometry)
  - [Geometry limits (negative result)](#geometry-limits-negative-result)
- [Applications: credit scoring](#applications-credit-scoring)
- [Scope: where it helps — and where it doesn't](#scope)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Papers](#papers)
- [Citation](#citation)
- [Contributing, conduct, security](#contributing-conduct-security)
- [License](#license)
- [About](#about)

---

## Overview

**hamiltonian-ai** consolidates one line of work: applying energy and phase-space
(Hamiltonian) geometry to neural systems. It is deliberately small and deliberately
bounded. It contains four things:

| Part | What it is | Where |
|---|---|---|
| **Library** | One packaged, reusable piece — a symplectic optimizer with a Hamiltonian energy view of the parameter dynamics | [`src/hamiltonian_ai/optimizer/`](src/hamiltonian_ai/optimizer/) |
| **Studies** | Reproducible notebooks and write-ups: a phase-space lens on LLM reasoning, and a negative-results study on the limits of geometric optimization | [`studies/`](studies/) |
| **Applications** | The credit-scoring work the optimizer was validated on | [`applications/credit-scoring/`](applications/credit-scoring/) |
| **Papers** | The underlying preprints and peer-reviewed work | [`papers/`](papers/) |

The one-line summary of the whole line: *geometric structure is a useful **diagnostic**
and a niche **stability** tool — not a general performance win.* The repository is
organized so that **both halves of that claim are visible**: the niche where it helps,
and the systematic study of where it does not.

## Installation

```bash
pip install -e .
python examples/optimizer_minimal.py   # smoke test of the API — not a benchmark
```

Requires Python ≥ 3.9, `torch ≥ 1.10`, `numpy ≥ 1.19`. The optimizer is the only
installed component; the studies and applications run as notebooks in the repo.

## Quickstart

```python
import torch.nn as nn
from hamiltonian_ai.optimizer import HamiltonianOptimizer

model = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 2))

optimizer = HamiltonianOptimizer(
    model.parameters(),
    lr=0.01,          # step size (time step of the integrator)
    beta=0.9,         # momentum coefficient
    lambda_reg=0.01,  # potential-energy regularization (Hamiltonian loss term)
)
# ...standard PyTorch training loop: loss.backward(); optimizer.step()...
```

Higher-order (Forest–Ruth, 4th-order symplectic) variant:

```python
from hamiltonian_ai.optimizer import HamiltonianOptimizerAdvanced
optimizer = HamiltonianOptimizerAdvanced(model.parameters(), lr=0.01, symplectic_order=4)
```

## The optimizer

> **Scope first: the optimizer trades peak accuracy for temporal stability /
> robustness to distribution shift. It is not a general improvement over Adam or
> SGD, and on standard in-distribution accuracy it generally loses to them.**

### How it works

The update is treated as a **Hamiltonian system**: kinetic energy from momentum,
potential energy from the parameters. Each step is scaled inversely with the total
energy, which damps destabilizing updates, and is integrated with a **symplectic**
scheme (symplectic Euler in the base class; Forest–Ruth 4th-order in
`HamiltonianOptimizerAdvanced`). Paired with it is a **Hamiltonian loss** —
`base_loss + lambda_reg · potential` — where the potential term penalizes large
parameter energy, analogous to damping oscillations in a mechanical system. The
design goal is *temporal* generalization: keeping a model's risk ranking stable as
the data distribution drifts.

### API

| Class | Key arguments | Notes |
|---|---|---|
| `HamiltonianOptimizer` | `params, lr=1e-2, beta=0.9, epsilon=1e-8, lambda_reg=1e-2, weight_decay=0.0` | Symplectic-Euler base optimizer |
| `HamiltonianOptimizerAdvanced` | `…, symplectic_order=4` | Forest–Ruth 4th-order symplectic integration |

Both subclass `torch.optim.Optimizer` and work in any standard PyTorch loop.

### Results, in scope

On the IEEE DSAA 2025 credit-scoring study (Freddie Mac Single-Family Loan-Level
Dataset), the method's **out-of-time ranking improves while its standard accuracy
does not**:

- Out-of-time discriminative power: **AUC ≈ 0.80 vs ≈ 0.61** for conventional models.
- Ablation: Hamiltonian normalization + momentum together account for a **+11.51%**
  improvement over a standard neural network, with statistical-significance testing.
- **Temporal stability**: minimal degradation across 12 / 36 / 60-month horizons,
  where standard techniques decline more steeply.
- **Cost**: standard accuracy, precision and recall are *lower* (gradient-boosted
  trees win those); the win is forward-looking *ranking stability*, not accuracy.

| Dimension | Adam / SGD / XGBoost | Hamiltonian optimizer |
|---|---|---|
| Standard in-distribution accuracy | **win** | loses |
| Out-of-time ranking (AUC) under drift | loses | **win** (this domain) |
| Evidence base | broad, general-purpose | single domain (credit scoring) |

This is a trade-off, not a free lunch. **If you do not have a temporal
distribution-shift problem, use Adam or SGD.**

## Studies

### Reasoning geometry

The reasoning-geometry study maps LLM multi-hop reasoning chains into a phase space
and assigns each a Hamiltonian "energy" balancing reasoning progression (kinetic)
against question relevance (potential). The empirical finding: **valid reasoning
chains show lower and more stable Hamiltonian energy** than invalid ones — a usable
**diagnostic** signal, plus smoother, lower-curvature embedding trajectories for
valid chains.

**Caveat, as the paper itself states it:** the claimed ability to *steer* or
*improve* reasoning is metaphorical and not empirically established. The solid
contribution is the diagnostic geometry, not a causal mechanism. This diagnostic
thread is the conceptual ancestor of [Groundlens](https://github.com/groundlens-dev/groundlens).

→ [`studies/reasoning-geometry/`](studies/reasoning-geometry/) · arXiv:[2410.04415](https://arxiv.org/abs/2410.04415)

### Geometry limits (negative result)

A separate, systematic study tested whether *generic* isotropic geometric structure
helps optimization at all — a Geometric Preconditioned Method (GPM) across **seven**
tasks: image classification, language modeling, rotation-equivariant learning, PDE
solving, continual learning, temporal prediction, and physical operator learning.
The result is negative and statistically significant on every task:

- **Geometric structure consistently underperforms Adam and SGD-with-momentum**,
  by roughly 10–300% depending on the task.
- The sole measured advantage is **~50% lower optimizer memory**.
- Central lesson: *data geometry and loss-landscape geometry are distinct, and the
  latter dominates optimization dynamics.* If geometry matters for a problem, encode
  it in the **architecture** (equivariant layers, spectral normalization, symplectic
  networks), not in a generic optimizer.

This study is treated as a first-class result: mapping the boundary is the
contribution, and it is why the optimizer above is scoped as a narrow stability tool
rather than a default.

→ [`studies/geometry-limits/`](studies/geometry-limits/)

## Applications: credit scoring

The optimizer was validated on out-of-time credit scoring with the Freddie Mac
loan-level dataset across 12/36/60-month horizons — the source of the "Results, in
scope" numbers above. The notebooks reproduce the out-of-time evaluation protocol
(adjacent, non-overlapping time intervals) used in the IEEE DSAA 2025 study.

→ [`applications/credit-scoring/`](applications/credit-scoring/)

## Scope

A compact statement of what this line claims and does not claim:

| Claim | Status |
|---|---|
| Geometric/phase-space energy is a useful **diagnostic** of valid reasoning | Supported (arXiv:2410.04415) |
| A symplectic optimizer can improve **out-of-time ranking stability** in a regulated, drifting domain | Supported, single domain (IEEE DSAA 2025) |
| Geometric structure improves optimization **in general** | **Refuted** — loses to Adam/SGD across 7 tasks |
| The reasoning framework **causally steers** reasoning | Not established (metaphor) |

## Repository layout

```
hamiltonian-ai/
├── src/hamiltonian_ai/optimizer/   the packaged optimizer (the reusable part)
├── studies/
│   ├── reasoning-geometry/         phase-space reasoning diagnostics (arXiv:2410.04415)
│   └── geometry-limits/            the negative-results boundary study
├── applications/credit-scoring/    IEEE DSAA 2025 notebooks (Freddie Mac OOT)
├── papers/                         preprints, peer-reviewed work, foundational deck
├── examples/                       minimal runnable optimizer example
└── docs/                           mkdocs-material site
```

## Documentation

Full docs (mkdocs-material, same design as the rest of Groundlens) cover
installation, the optimizer API, the honest benchmark, the reasoning diagnostics,
and the limits study: **[docs.groundlens.dev/hamiltonian-ai](https://docs.groundlens.dev/hamiltonian-ai)**.
Build locally with `pip install -e ".[docs]" && mkdocs serve`.

## Papers

| Work | Topic | Status |
|---|---|---|
| Geometric Analysis of Reasoning Trajectories | Phase-space diagnostics for multi-hop QA reasoning | arXiv:[2410.04415](https://arxiv.org/abs/2410.04415) (2024) |
| Hamiltonian NN for Out-of-Time Credit Scoring | Symplectic optimizer + Hamiltonian loss; out-of-time ranking | **Accepted (peer-reviewed), IEEE DSAA 2025** — not in the proceedings; cite as peer-reviewed preprint |
| When Geometric Structure Doesn't Help | Systematic negative-results study on geometric optimization | Working paper / preprint |

PDFs and the foundational deck are in [`papers/`](papers/).

## Citation

```bibtex
@article{marin2024reasoning,
  title   = {Geometric Analysis of Reasoning Trajectories: A Phase Space Approach
             to Understanding Valid and Invalid Multi-Hop Reasoning in LLMs},
  author  = {Mar\'in, Javier},
  journal = {arXiv preprint arXiv:2410.04415},
  year    = {2024}
}

@misc{marin2025hamiltonian,
  title  = {Hamiltonian Neural Networks for Robust Out-of-Time Credit Scoring:
            Empirical Validation and Temporal Stability Analysis},
  author = {Mar\'in, Javier},
  year   = {2025},
  note   = {Accepted (peer-reviewed), IEEE DSAA 2025; preprint, not in proceedings}
}
```

## Contributing, conduct, security

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

MIT License — see [LICENSE](LICENSE). Author: Javier Marin.

## About

`hamiltonian-ai` is a research line of **Groundlens**, an open-source practice for
trustworthy modeling. Its sibling projects are
[groundlens](https://github.com/groundlens-dev/groundlens) (geometric verification of
LLM outputs) and [otwin](https://github.com/groundlens-dev/otwin) (physics-informed
digital twins). The shared thesis: read the geometry, state the limits.

---

<div align="center">
An open-source research line by <b>Groundlens</b> · <a href="https://groundlens.dev">groundlens.dev</a>
</div>
