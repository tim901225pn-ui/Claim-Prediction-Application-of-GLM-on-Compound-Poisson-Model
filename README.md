# Claim Prediction

## Model Assumptions

Our loss model is described by the **Compound Poisson Model**, which takes the form

```math
S \mid (\mathbf{x}, \nu) = \sum_{n=1}^{N} Y_n, \quad N \mid (\mathbf{x}, \nu) \sim \text{Poi}(\lambda(\mathbf{x}) \cdot \nu), \quad Y_n \mid \mathbf{x} \stackrel{\text{i.i.d.}}{\sim} d(\mathbf{x})
```

where
- `$\nu$` — the policy exposure
- `$\mathbf{x}$` — the risk factors of the insured
- `$S$` — the aggregate loss given `$\mathbf{x}$` and `$\nu$`
- `$N$` — the claim count
- `$Y_n$` — the amount of the $n$-th claim

with `$N$` assumed independent of `$\{Y_n\}_{n \ge 1}$` given `$(\mathbf{x}, \nu)$`.

## Modeling the Claim Count $N$ With GLM

We model $N \mid (\mathbf{x}, \nu)$ via a Poisson GLM with a log link. The expected claim count is assumed to scale proportionally with exposure $\nu$, i.e.

```math
\mathbb{E}[N \mid \mathbf{x}, \nu] = \lambda(\mathbf{x}) \cdot \nu, \qquad \lambda(\mathbf{x}) = \exp(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)
```

where $\lambda(\mathbf{x})$ is the claim frequency per unit exposure given risk factors $\mathbf{x}$. Taking logs yields the linear predictor actually fit by the GLM

```math
\log(\mathbb{E}[N \mid \mathbf{x}, \nu]) = \log(\nu) + \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
```

with $\log(\nu)$ entered as an **offset**.