# Changelog

All notable changes to **hamiltonian-ai** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-27

### Added

- Initial consolidation of the Hamiltonian / energy-geometry research line
  into a single repository under the Groundlens org.
- **Library:** packaged `hamiltonian_ai.optimizer` with `HamiltonianOptimizer`
  and `HamiltonianOptimizerAdvanced` (symplectic / Forest-Ruth integration),
  with scope-first docstrings (temporal stability trade-off, not a general
  Adam/SGD replacement).
- **Studies:** `reasoning-geometry` (phase-space diagnostics for multi-hop
  reasoning, arXiv:2410.04415) and `geometry-limits` (the negative-results
  study on the limits of geometric optimization).
- **Applications:** `credit-scoring` notebooks (IEEE DSAA 2025, Freddie Mac
  out-of-time validation).
- **Papers:** preprints, the peer-reviewed credit-scoring work, and the
  foundational deck, with precise status framing.
- mkdocs-material documentation aligned to the Groundlens design.
- Minimal runnable optimizer example, packaging, and project hygiene files.

### Notes

- Consolidates and dedupes the prior `Hamiltonian-Neural-Network-Optimization`
  and `hamiltonian_ai` repositories (which were identical) into one canonical
  source.
