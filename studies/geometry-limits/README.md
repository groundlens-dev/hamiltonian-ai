# Geometry limits — when geometric structure doesn't help

A systematic negative-results study, and the credibility anchor of this
research line. It asks a clean question: does *generic* isotropic geometric
structure help neural-network optimization at all?

## The study

A Geometric Preconditioned Method (GPM) — isotropic optimization with spectral
constraints — was evaluated across **seven diverse tasks**:

1. Image classification
2. Language modeling
3. Rotation-equivariant learning
4. PDE solving
5. Continual learning
6. Temporal prediction
7. Physical operator learning

## The result (negative, and statistically significant)

Across **all seven** tasks, GPM **consistently underperforms** standard
adaptive (Adam) and first-order (SGD with momentum) methods, by significant
margins, with statistical significance testing throughout.

## Why it matters

The central lesson is precise:

> Data geometry and loss-landscape geometry are distinct, and the latter
> dominates optimization dynamics.

Practical consequences drawn by the paper:

- If geometric structure matters, **encode it in the architecture**
  (group-equivariant layers, spectral normalization, symplectic networks),
  **not in a generic optimizer**.
- Spectral control does not universally aid generalization; it often hinders
  expressiveness.
- For memory-constrained deployment, prefer SGD with momentum over geometric
  methods.

This study is *why* the optimizer in this repository is presented as a narrow,
domain-specific stability tool rather than a default. Mapping the boundary is
the contribution — negative results prevent wasted effort and clarify the gap
between geometric intuition and optimization practice.

## Paper

*When Geometric Structure Doesn't Help: A Systematic Study of Isotropic
Optimization Failures.* Working paper / preprint. PDF:
[`../../papers/When_Geometric_Structure_Doesn_t_Help.pdf`](../../papers/When_Geometric_Structure_Doesn_t_Help.pdf).
