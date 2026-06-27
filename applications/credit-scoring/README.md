# Credit scoring — out-of-time ranking stability (IEEE DSAA 2025)

The application the Hamiltonian optimizer was validated on: out-of-time (OOT)
credit scoring on the Freddie Mac Single-Family Loan-Level Dataset.

## The bounded result

The method's contribution is **temporal ranking stability**, not accuracy:

- It improves **out-of-time AUC** (discriminative ranking power) and degrades
  more slowly across increasing time horizons (12, 36, 60 months) than
  conventional baselines.
- It **does not** win on standard classification accuracy — strong baselines
  (XGBoost, conventional NNs) achieve higher conventional accuracy. The paper
  is explicit that the gain is in risk *ranking* under temporal drift, "even
  when conventional accuracy metrics favor traditional approaches."
- Ablations attribute the effect to both Hamiltonian normalization and
  momentum updates.

In short: use it when accurate forward-looking risk *ranking* and long-term
model stability matter more than peak classification accuracy. Otherwise use a
standard method.

## Notebooks

Real experimental notebooks migrated from the source work:

- `Credit_scoring_FM12_v1.ipynb` — 12-month OOT validation (primary)
- `Credit_scoring_FM12_v2.ipynb` — alternative implementation + ablations
- `Credit_scoring_FM36_v1.ipynb` — 36-month OOT validation
- `Credit_scoring_FM60_v1.ipynb` — 60-month OOT validation

Dataset (free, registration required): [Freddie Mac Single-Family Loan-Level
Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
Class imbalance handled with SMOTE.

## Paper

Marin, J. (2025). *Hamiltonian Neural Networks for Robust Out-of-Time Credit
Scoring: Empirical Validation and Temporal Stability Analysis.*
**Accepted (peer-reviewed), IEEE DSAA 2025** — note: not in the proceedings;
preprint. PDFs:
[`HamiltonianNN_IEEE_DSAA.pdf`](../../papers/HamiltonianNN_IEEE_DSAA.pdf),
[`Hamiltonian_NN.pdf`](../../papers/Hamiltonian_NN.pdf).
