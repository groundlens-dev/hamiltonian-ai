# Optimizer — honest benchmark framing

!!! danger "The headline you will not see here"
    This page does **not** claim the optimizer beats Adam, SGD, or XGBoost.
    On standard accuracy, those methods win. The optimizer's only demonstrated
    edge is **out-of-time ranking stability** in one domain.

## What the IEEE DSAA 2025 study actually shows

On the Freddie Mac Single-Family Loan-Level Dataset, evaluated out-of-time
(forward-looking, non-overlapping intervals at 12 / 36 / 60 months):

- The method achieves **higher out-of-time AUC** (discriminative *ranking*
  power) than conventional baselines, and degrades more slowly across
  increasing time horizons.
- The method achieves **lower standard classification accuracy** than those
  same baselines. Strong baselines (XGBoost, conventional NNs) win on accuracy.
- Statistical significance testing confirms the ranking-stability effect is not
  random variation.

## How to read the comparison

| Dimension | Adam / SGD / XGBoost | Hamiltonian optimizer |
|---|---|---|
| Standard in-distribution accuracy | **win** | loses |
| Out-of-time ranking (AUC) under drift | loses | **win** (credit scoring only) |
| Training overhead | baseline | +15–20% |
| Evidence base | broad | single domain |

This is a **trade-off**. The right summary is: *if you need forward-looking
risk ranking that stays stable as the distribution drifts, and you can pay for
some accuracy and compute, this method may help in that narrow setting.*

## Reproducing

The notebooks are in `applications/credit-scoring/`. The dataset is free
(registration required) from
[Freddie Mac](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
Class imbalance is handled with SMOTE.

## Status of the paper

**Accepted (peer-reviewed), IEEE DSAA 2025** — note: not in the proceedings;
cite as a peer-reviewed preprint, not as a published IEEE Xplore paper.
