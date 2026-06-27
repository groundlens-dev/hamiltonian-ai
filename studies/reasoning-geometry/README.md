# Reasoning geometry — phase-space diagnostics for multi-hop reasoning

A phase-space (Hamiltonian) lens on LLM multi-hop reasoning. Reasoning chains
are mapped into an embedding phase space, and each chain is assigned a
Hamiltonian "energy" balancing reasoning *progression* (kinetic energy)
against *question relevance* (potential energy).

**Finding.** Valid reasoning chains exhibit **lower and more stable**
Hamiltonian energy profiles than invalid chains. Trajectory curvature and
conservation-like quantities discriminate valid from invalid reasoning. This
is a **diagnostic** tool.

**Caveat (as stated in the paper).** The claimed ability to *steer* or
*improve* reasoning is largely metaphorical and requires more rigorous
empirical validation; the connection between physical systems and reasoning is
an analogy. The solid contribution is the geometric diagnostic patterns.

## Contents

- `Hamiltonian_final_version.ipynb` — experiments, analysis, and visualizations
  from the paper.
- `AdvancedSymplecticOptimizer_v2.ipynb` — exploratory symplectic-integration
  notebook accompanying the analysis.

Dataset: OpenBookQA (Mihaylov et al., 2018).

## Paper

Marín, J. (2024). *Geometric Analysis of Reasoning Trajectories: A Phase Space
Approach to Understanding Valid and Invalid Multi-Hop Reasoning in LLMs.*
arXiv:[2410.04415](https://arxiv.org/abs/2410.04415)
(DOI:[10.48550/arXiv.2410.04415](https://doi.org/10.48550/arXiv.2410.04415)).
PDF in [`../../papers/`](../../papers/).
