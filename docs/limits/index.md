# Limits — where geometric structure doesn't help

This is the credibility anchor of the line. A systematic negative-results study
asks: does *generic* isotropic geometric structure help neural-network
optimization at all?

## The study

A Geometric Preconditioned Method (GPM) — isotropic optimization with spectral
constraints — was evaluated across **seven diverse tasks**: image
classification, language modeling, rotation-equivariant learning, PDE solving,
continual learning, temporal prediction, and physical operator learning.

## The result

Across **all seven** tasks, GPM **consistently underperforms** Adam and SGD
with momentum, by significant margins, with statistical significance testing
throughout. The result is negative — and scientifically valuable.

## The lesson

!!! quote
    Data geometry and loss-landscape geometry are distinct, and the latter
    dominates optimization dynamics.

Consequences:

- If geometric structure matters, **encode it in the architecture**
  (equivariant layers, spectral normalization, symplectic networks), not in a
  generic optimizer.
- Spectral control does not universally aid generalization; it often hinders
  expressiveness.
- For memory-constrained deployment, prefer SGD with momentum over geometric
  methods.

This study is *why* the [optimizer](../optimizer/usage.md) is presented as a
narrow, domain-specific stability tool. Mapping the boundary is the result.

## Paper

*When Geometric Structure Doesn't Help: A Systematic Study of Isotropic
Optimization Failures.* Working paper / preprint. PDF in the repository's
`papers/` directory.
