# Installation

`hamiltonian-ai` requires Python 3.9+ and PyTorch.

## From source

```bash
git clone https://github.com/groundlens-dev/hamiltonian-ai.git
cd hamiltonian-ai
pip install -e .
```

## With docs / dev extras

```bash
pip install -e ".[dev,docs]"
```

## Verify

```python
from hamiltonian_ai.optimizer import HamiltonianOptimizer
print("ok")
```

The packaged part is the optimizer. The studies and applications are run as
notebooks against your own data — see [Reasoning diagnostics](../reasoning/index.md)
and the `applications/credit-scoring/` notebooks in the repository.
