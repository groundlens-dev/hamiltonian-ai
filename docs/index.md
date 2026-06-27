# hamiltonian-ai

**Energy-geometry methods for neural systems.**

A small library — a symplectic optimizer — plus reproducible studies on the
geometry of reasoning, and an explicit map of where geometric structure stops
helping.

!!! note "A Groundlens research line"
    Part of the [Groundlens](https://github.com/groundlens-dev) family —
    alongside [groundlens](https://github.com/groundlens-dev/groundlens) and
    [otwin](https://github.com/groundlens-dev/otwin).

## What this is

This project applies energy and phase-space (Hamiltonian) geometry to neural
systems. It is deliberately bounded:

- **Optimizer** — one packaged, reusable piece: a symplectic optimizer with a
  Hamiltonian energy view of parameter dynamics. It trades peak accuracy for
  temporal stability; it is **not** a general improvement over Adam or SGD.
- **Reasoning diagnostics** — a phase-space lens on LLM multi-hop reasoning;
  valid chains show lower, more stable Hamiltonian energy. A diagnostic, not a
  steering mechanism.
- **Limits** — a systematic negative-results study: as a general optimizer,
  geometric structure consistently underperforms Adam/SGD. This is the
  credibility anchor.

The honest summary: *geometric structure is a useful diagnostic and a niche
stability tool, not a general performance win.*

## Where to start

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Optimizer usage](optimizer/usage.md) and its [honest benchmark](optimizer/benchmark.md)
- [Reasoning diagnostics](reasoning/index.md)
- [Limits](limits/index.md) — read this to understand the scope
- [Papers](papers.md)
