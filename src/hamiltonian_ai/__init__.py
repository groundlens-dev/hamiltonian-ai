"""hamiltonian_ai — energy-geometry methods for neural systems.

The packaged, importable part of this research line is the symplectic
:class:`~hamiltonian_ai.optimizer.HamiltonianOptimizer`. The reasoning-geometry
diagnostics and the negative-results boundary study live as reproducible
studies in the repository (``studies/``, ``applications/``), not as installed
modules.

Scope, in one sentence: the optimizer trades peak accuracy for temporal
stability / robustness to distribution shift, and is not a general improvement
over Adam or SGD.
"""

from .optimizer import HamiltonianOptimizer, HamiltonianOptimizerAdvanced

__version__ = "0.1.0"
__author__ = "Javier Marin"
__email__ = "javier@jmarin.info"

__all__ = ["HamiltonianOptimizer", "HamiltonianOptimizerAdvanced", "__version__"]
