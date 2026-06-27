<div align="center">
<img src="https://raw.githubusercontent.com/groundlens-dev/hamiltonian-ai/main/docs/assets/hamiltonian_equivalence.png" width="760" alt="Hamiltonian mechanics mapped onto a neural-network loss / phase space" />

</div>

# Hamiltonian-AI: Symmetry, Invariance, and Structure in Neural Optimization

Symplectic optimization, phase-space diagnostics of reasoning, and a candid map of where geometric structure helps — and where it does not.

[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue?style=flat-square)](https://github.com/groundlens-dev/hamiltonian-ai)
[![Tests](https://img.shields.io/github/actions/workflow/status/groundlens-dev/hamiltonian-ai/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/groundlens-dev/hamiltonian-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-docs.groundlens.dev%2Fhamiltonian--ai-blue?style=flat-square)](https://docs.groundlens.dev/hamiltonian-ai)
[![arXiv](https://img.shields.io/badge/arXiv-2410.04415-b31b1b?style=flat-square)](https://arxiv.org/abs/2410.04415)

[Overview](#overview) · [Install](#installation) · [Quickstart](#quickstart) · [Optimizer](#the-optimizer) · [Reasoning](#reasoning-geometry) · [Scope](#scope) · [Papers](#papers) · [Docs](https://docs.groundlens.dev/hamiltonian-ai)

</div>

*A Groundlens research line. Part of the [Groundlens](https://github.com/groundlens-dev) family — alongside [groundlens](https://github.com/groundlens-dev/groundlens) (geometric LLM verification) and [otwin](https://github.com/groundlens-dev/otwin) (physics-informed digital twins).*

---

> *"Symmetry, as wide or narrow as you may define its meaning, is one idea by which man through the ages has tried to comprehend and create order, beauty, and perfection."* 
>
> — Hermann Weyl, *Symmetry* (1952)

## The Journey

This journey starts with understanding that physics earns its predictive power from 
**symmetry**: when a system is left unchanged by a transformation, something is conserved. 
Invariance under time translation gives conservation of energy; the bookkeeping device for that
conservation is the **Hamiltonian**, and its natural habitat is **phase space**,
a geometry that a symplectic structure keeps intact as the system evolves
(Noether; Weyl 1952) and the synthesis of symmetry, invariance and structure.

This repository asks a single question across two problems: *what happens when
we treat a neural system as a Hamiltonian one* — its parameters as positions,
its gradients as momenta, its loss as an energy landscape? We have found two possible answers,
and this repository is organized to show **both**:

- a **diagnostic** that works: the phase-space energy of an LLM's reasoning
  trajectory separates valid from invalid reasoning;
- a **boundary** that is just as important: as a *general* optimizer, geometric
  structure does not beat Adam or SGD.

For our journey, *geometric structure is a useful diagnostic and a niche
stability tool — not a general performance win.*

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The optimizer](#the-optimizer)
  - [How it works](#how-it-works)
  - [API](#api)
  - [Results, in scope](#results-in-scope)
- [Hamiltonian in LLMs: Reasoning Geometry](#hamiltonian-in-llms-reasoning-geometry)
- [The limits of geometric optimization](#the-limits-of-geometric-optimization)
- [Scope](#scope)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Papers](#papers)
- [Citation & references](#citation--references)
- [Contributing, conduct, security](#contributing-conduct-security)
- [License](#license) · [About](#about)

---

## Overview

**hamiltonian-ai** consolidates one line of work into four parts:

| Part | What | Where |
|---|---|---|
| **Library** | A symplectic optimizer with a Hamiltonian energy view of the parameter dynamics | [`src/hamiltonian_ai/optimizer/`](src/hamiltonian_ai/optimizer/) |
| **Studies** | A phase-space lens on LLM reasoning, and a negative-results study on the limits of geometric optimization | [`studies/`](studies/) |
| **Applications** | The out-of-time credit-scoring evaluation the optimizer was validated on | [`applications/credit-scoring/`](applications/credit-scoring/) |
| **Papers** | The underlying preprints and peer-reviewed work | [`papers/`](papers/) |


## Installation

```bash
pip install -e .
pytest -q                                # minimal test suite
python examples/optimizer_minimal.py     # smoke test — not a benchmark
```

Requires Python ≥ 3.9, `torch ≥ 1.10`, `numpy ≥ 1.19`. The optimizer is the only
installed component; the studies and applications run as notebooks.

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
# ...standard PyTorch loop: loss.backward(); optimizer.step()...
```

Higher-order (Forest–Ruth, 4th-order symplectic) variant:

```python
from hamiltonian_ai.optimizer import HamiltonianOptimizerAdvanced
optimizer = HamiltonianOptimizerAdvanced(model.parameters(), lr=0.01, symplectic_order=4)
```

## The optimizer

This optimizer is built for out-of-time *ranking* under class
imbalance and distribution drift. It is not a general improvement over Adam or
SGD, and on standard in-distribution accuracy it generally loses to them.**

### How it works

The update is treated as a **Hamiltonian system**: kinetic energy from momentum,
potential energy from the parameters. Each step is scaled by the total energy
(damping destabilizing updates) and integrated with a **symplectic** scheme —
symplectic Euler in the base class, Forest–Ruth 4th-order in
`HamiltonianOptimizerAdvanced`. Paired with it is a **Hamiltonian loss**,
`base_loss + lambda_reg · potential`, where the potential term penalizes large
parameter energy (damping oscillations in parameter space).


Each choice answers a specific weakness of plain gradient descent:

- **Why a Hamiltonian (momentum) system.** Gradient descent is the discretization of
  a *first-order* flow, $\dot{\theta} = -\nabla L$; it has no inertia, so it stalls in
  flat regions and zig-zags across ravines. Momentum methods are the discretization of
  a *second-order* (Newtonian/Hamiltonian) system with friction,
  $m\,\ddot{\theta} + \gamma\,\dot{\theta} = -\nabla L$ (Polyak; Su–Boyd–Candès 2016;
  Maddison et al. 2018). Writing it explicitly: being position $q = \theta$ and momentum $p$:

<div align="center">

  *Kinetic energy* $T(p) = \tfrac{1}{2}\lVert p \rVert^{2}$

  *Potential energy* $V(q) = L$

</div>

  This lets inertia carry the trajectory through small barriers and along narrow valleys.

- **Why scale the step by total energy.** A *conservative* Hamiltonian system never
  settles: the energy $H = T + V$ is conserved, so it orbits the minimum forever. To
  turn a simulator into an *optimizer* you must remove energy. Scaling the step by $H$
  is state-dependent damping — strong when the system is far and oscillating, gentle
  near a minimum — which converts the perpetual orbit into a convergent spiral while
  keeping the exploratory inertia.

- **Why a symplectic integrator (symplectic Euler / Forest–Ruth).** A generic
  integrator (e.g. forward Euler) does not just blur a Hamiltonian trajectory; it
  *systematically injects or drains energy*, so the simulated dynamics drift away from
  the system you designed. Symplectic integrators preserve the phase-space structure
  (the symplectic form $\omega = dq \wedge dp$) and keep $H$ bounded over long horizons
  (Sanz-Serna; Hairer–Lubich–Wanner). That keeps the energy bookkeeping faithful — so
  the energy-based damping above means what it is supposed to mean. Forest–Ruth is its
  4th-order version, more accurate per step.

- **Why the Hamiltonian loss.** The training objective adds a harmonic regularizer on
  parameter "energy", $\mathcal{L} = \mathcal{L}_{\text{base}} + \lambda\,V(\theta)$
  with $V(\theta) \propto \lVert \theta \rVert^{2}$ (≈ weight decay): in the mechanical
  picture, high potential energy means large oscillations, so penalizing it damps them
  and biases training toward flatter, lower-norm solutions — the regime usually
  associated with better generalization under shift (Keskar et al. 2017). It also keeps
  the potential well-defined for the energy-scaling step.

Taken together, these choices trade *speed* for *stability*: they favor flat, robust
basins over sharp minima, which is why the method earns its keep on out-of-time ranking
rather than on raw convergence. The boundary study
([limits](#the-limits-of-geometric-optimization)) is the candid counterweight — this
structure is not a general improvement over Adam or SGD.

<div align="center">
<img src="https://raw.githubusercontent.com/groundlens-dev/hamiltonian-ai/main/docs/assets/HamiltoniainOptimization.png" width="680" alt="Gradient descent vs Hamiltonian momentum-based trajectories on a loss surface" />
<br><sub>Step-by-step gradient descent (red) vs the Hamiltonian, momentum-based trajectory (orange) on the same loss surface.</sub>
<br><br>
<img src="https://raw.githubusercontent.com/groundlens-dev/hamiltonian-ai/main/docs/assets/Loan_risk-02.png" width="720" alt="Network, Hamiltonian loss and symplectic optimizer block diagram" />
<br><sub>The training loop: the network feeds a Hamiltonian loss, which the symplectic optimizer integrates over the Hamiltonian loss space.</sub>
</div>

### API

| Class | Key arguments | Notes |
|---|---|---|
| `HamiltonianOptimizer` | `params, lr=1e-2, beta=0.9, epsilon=1e-8, lambda_reg=1e-2, weight_decay=0.0` | Symplectic-Euler base optimizer |
| `HamiltonianOptimizerAdvanced` | `…, symplectic_order=4` | Forest–Ruth 4th-order symplectic integration |

Both subclass `torch.optim.Optimizer` and drop into any standard PyTorch loop.

### Results, in scope

Out-of-time credit scoring on the Freddie Mac Single-Family Loan-Level Dataset,
across 12 / 36 / 60-month horizons (`applications/credit-scoring/`). The pattern
is a clean trade-off — and the point is *which metric matters under class
imbalance*:

| Horizon | Symplectic AUC | XGBoost AUC | Symplectic Acc. | XGBoost Acc. |
|---|---|---|---|---|
| 12 months | **0.803** | 0.607 | 0.805 | 0.987 |
| 36 months | **0.764** | 0.622 | 0.764 | 0.966 |
| 60 months | **0.697** | 0.667 | 0.698 | 0.932 |

Read it carefully:

- **XGBoost wins accuracy by a wide margin (≈0.99) — but its AUC is close to
  chance (0.61–0.67).** On heavily imbalanced default data, a model can score
  ~99% accuracy by predicting "no default" for almost everyone while ranking
  risk barely better than a coin. Accuracy is the wrong target here.
- **The symplectic model wins AUC at every horizon (0.80 / 0.76 / 0.70)** — it
  actually *ranks* risk. That ranking degrades with the horizon (0.80→0.70), so
  this is "better discrimination," not "no degradation."
- **The claim, precisely:** for out-of-time, class-imbalanced risk ranking, the
  symplectic optimizer delivers real discriminative power where a strong
  gradient-boosted baseline collapses to near-chance ranking. *If you do not
  have that problem, use Adam, SGD, or XGBoost.*

## Hamiltonian in LLMs: Reasoning Geometry

The reasoning-geometry work (arXiv:[2410.04415](https://arxiv.org/abs/2410.04415))
maps an LLM's multi-hop reasoning chain into a phase space and assigns it a
Hamiltonian "energy" balancing reasoning progression (kinetic) against question
relevance (potential), using BERT embeddings of each reasoning step on the
OpenBookQA dataset, with a BERT classifier for chain validity.

We found that valid reasoning chains carry **lower and more tightly concentrated Hamiltonian 
energy** than invalid ones; the difference is statistically significant
(**t = −6.53, p = 0.0001**). Valid trajectories form a narrow, peaked 
energy distribution; invalid ones spread broadly toward higher energy. 
Curvature analysis (Frenet–Serret) shows smoother, lower-curvature
trajectories for valid chains.

<div align="center">
<img src="https://raw.githubusercontent.com/groundlens-dev/hamiltonian-ai/main/docs/assets/Symplectic.png" width="640" alt="Pendulum motion and its closed orbit in phase space" />
<br><sub>Phase space, the natural habitat of a Hamiltonian: a pendulum's motion traces a closed orbit in (angle, velocity). A reasoning chain traces an analogous trajectory in embedding space.</sub>
</div>
**Caveat, as the paper states it:** the framework's claimed ability to *steer*
or *improve* reasoning is metaphorical and not empirically established. The solid
contribution is the **diagnostic geometry**, not a causal mechanism. This
diagnostic thread is the conceptual ancestor of
[Groundlens](https://github.com/groundlens-dev/groundlens).

→ [`studies/reasoning-geometry/`](studies/reasoning-geometry/)

## The limits of geometric optimization

A separate, systematic study tested whether *generic* isotropic geometric
structure helps optimization at all — a Geometric Preconditioned Method across
**seven** tasks (image classification, language modeling, rotation-equivariant
learning, PDE solving, continual learning, temporal prediction, physical
operator learning). The result is negative and statistically significant on
every task: **as a general optimizer, geometric structure consistently
underperforms Adam and SGD-with-momentum** (by ~10–300% depending on the task);
its sole measured advantage is ~50% lower optimizer memory.

The lesson: *data geometry and loss-landscape geometry are distinct,
and the latter dominates optimization dynamics.* If geometry matters for a
problem, encode it in the **architecture** (equivariant layers, spectral
normalization, symplectic networks), not in a generic optimizer. Mapping this
boundary is the result, and it is why the optimizer above is scoped as a narrow
tool rather than a default.

→ [`studies/geometry-limits/`](studies/geometry-limits/)

## Scope

| Claim | Status |
|---|---|
| Phase-space energy is a useful **diagnostic** of valid reasoning | Supported, statistically significant (arXiv:2410.04415) |
| A symplectic optimizer improves **out-of-time risk ranking (AUC)** under class imbalance | Supported, single domain (credit scoring) |
| Geometric structure improves optimization **in general** | **Refuted** — loses to Adam/SGD across 7 tasks |
| The reasoning framework **causally steers** reasoning | Not established (metaphor) |

## Repository layout

```
hamiltonian-ai/
├── src/hamiltonian_ai/optimizer/   the packaged optimizer (the reusable part)
├── studies/
│   ├── reasoning-geometry/         phase-space reasoning diagnostics (arXiv:2410.04415)
│   └── geometry-limits/            the negative-results boundary study
├── applications/credit-scoring/    out-of-time notebooks (Freddie Mac, 12/36/60 mo)
├── papers/                         preprints, peer-reviewed work, foundational deck
├── examples/                       minimal runnable optimizer example
├── tests/                          minimal optimizer test suite
└── docs/                           mkdocs-material site
```

## Documentation

Full docs (mkdocs-material, same design as the rest of Groundlens):
**[docs.groundlens.dev/hamiltonian-ai](https://docs.groundlens.dev/hamiltonian-ai)**.
Build locally with `pip install -e ".[docs]" && mkdocs serve`.

## Papers

| Work | Topic | Status |
|---|---|---|
| Optimizing AI Reasoning: A Hamiltonian Dynamics Approach to Multi-Hop QA | Phase-space diagnostics of reasoning trajectories | arXiv:[2410.04415](https://arxiv.org/abs/2410.04415) (2024) |
| Hamiltonian NN for Out-of-Time Credit Scoring | Symplectic optimizer + Hamiltonian loss; out-of-time ranking | **Accepted (peer-reviewed), IEEE DSAA 2025** — not in the proceedings; cite as peer-reviewed preprint |
| When Geometric Structure Doesn't Help | Systematic negative-results study on geometric optimization | Working paper / preprint |

PDFs and the foundational deck are in [`papers/`](papers/).

## Citation & references

```bibtex
@article{marin2024reasoning,
  title   = {Optimizing AI Reasoning: A Hamiltonian Dynamics Approach to
             Multi-Hop Question Answering},
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

**Conceptual references.** Weyl, H. *Symmetry* (Princeton University Press, 1952).
Glattfelder, J. B. *The Semantics of Symmetry, Invariance, and Structure*, ch. 3
in *Information—Consciousness—Reality* (The Frontiers Collection, Springer, 2019,
open access) — for the synthesis of symmetry, invariance and structure that
motivates the Hamiltonian framing used here.

## Contributing, conduct, security

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [SECURITY.md](SECURITY.md).

## License

MIT License — see [LICENSE](LICENSE). Author: Javier Marin.

## About

`hamiltonian-ai` is a research line of **Groundlens**, an open-source practice for
trustworthy modeling. Sibling projects:
[groundlens](https://github.com/groundlens-dev/groundlens) (geometric verification
of LLM outputs) and [otwin](https://github.com/groundlens-dev/otwin)
(physics-informed digital twins). The shared thesis: read the geometry, state the limits.

---

<div align="center">
An open-source research line by <b>Groundlens</b> · <a href="https://groundlens.dev">groundlens.dev</a>
</div>
