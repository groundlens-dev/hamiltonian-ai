# Optimizer — usage

The optimizer treats the parameter update as a Hamiltonian system and scales
each step inversely with total energy.

## How it works

For a parameter group, with momentum \(m\) and parameters \(\theta\):

\[
m_t = \beta\, m_{t-1} + (1-\beta)\, g_t
\]
\[
K_t = \tfrac{1}{2}\lVert m_t \rVert^2,\qquad
U_t = \tfrac{1}{2}\lVert \theta_t \rVert^2,\qquad
H_t = K_t + U_t
\]
\[
\eta_t = \frac{\text{lr}}{\sqrt{H_t} + \epsilon}\cdot\frac{1}{1+\lambda_\text{reg}},\qquad
\theta_{t+1} = \theta_t - \eta_t\, m_t
\]

The adaptive step \(\eta_t\) shrinks when system energy is high, damping
destabilizing updates. `HamiltonianOptimizerAdvanced` applies a Forest-Ruth
4th-order symplectic scheme for better energy conservation at higher cost
(experimental).

## Parameters

| Arg | Default | Meaning |
|---|---|---|
| `lr` | `1e-2` | learning rate |
| `beta` | `0.9` | momentum coefficient (0.95–0.99 for more stability) |
| `epsilon` | `1e-8` | numerical stability term |
| `lambda_reg` | `1e-2` | temporal-stability regularization (higher = more stable, slower) |
| `weight_decay` | `0.0` | optional L2 penalty |

## When to use it

- You have a **temporal distribution-shift** problem and care about
  out-of-time *ranking* stability.
- You can absorb **+15–20%** training overhead.

## When not to use it

- Standard, in-distribution accuracy is the goal — use Adam or SGD.
- Compute-constrained settings.
- Domains where it has not been validated (vision, NLP). The only validated
  domain is credit scoring.

See the [honest benchmark](benchmark.md) and the [limits](../limits/index.md).
