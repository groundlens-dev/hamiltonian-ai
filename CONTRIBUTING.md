# Contributing

Thanks for your interest in **hamiltonian-ai**, a Groundlens research line.

## Ground rules

This repository values **bounded, evidence-based claims**. Two things matter
more than features:

1. **Do not overclaim.** The optimizer is a temporal-stability tool, not a
   general improvement over Adam/SGD. Negative results are first-class
   contributions here — the `geometry-limits` study is the point, not an
   embarrassment.
2. **Keep results reproducible.** Code and prose must match what the notebooks
   and papers actually show. No invented numbers.

## How to contribute

1. Open an issue describing the change or finding.
2. Fork, branch from `main`, and make focused commits.
3. For library changes: keep the packaged optimizer minimal; add tests; run
   `black` and `ruff`.
4. For studies: keep notebooks runnable and document the dataset and seeds.
5. Open a pull request referencing the issue.

## Development setup

```bash
pip install -e ".[dev,docs]"
pytest
mkdocs serve
```

## Code style

- `black` (line length 100) and `ruff`.
- Google-style docstrings (used by the docs).

## License

By contributing you agree your contributions are licensed under the MIT
License.
