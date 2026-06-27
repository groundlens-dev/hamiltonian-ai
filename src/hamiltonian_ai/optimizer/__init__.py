"""Symplectic / Hamiltonian optimizer for temporal stability.

This subpackage contains the reusable, packaged part of ``hamiltonian_ai``:
a physics-inspired optimizer that trades peak in-distribution accuracy for
stability of out-of-time ranking under distribution shift.

It is NOT a general-purpose replacement for Adam or SGD. See the package
README and the ``geometry-limits`` study for the negative-results boundary
that defines where geometric structure helps and where it does not.
"""

from .optimizer import HamiltonianOptimizer, HamiltonianOptimizerAdvanced

__all__ = ["HamiltonianOptimizer", "HamiltonianOptimizerAdvanced"]
