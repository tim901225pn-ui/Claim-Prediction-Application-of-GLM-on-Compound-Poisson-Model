# **Claim Prediction**

## Model Assumptions

Our loss model is described by the **Compound Poisson Model**, which takes the form

$$
S \mid (\boldsymbol{x}, \nu) = \sum_{n=1}^{N} Y_n, \quad N \mid (\boldsymbol{x}, \nu) \sim \text{Poi}(\lambda(\boldsymbol{x}) \cdot \nu), \quad Y_n \mid \boldsymbol{x} \stackrel{\text{i.i.d.}}{\sim} d(\boldsymbol{x})
$$

where:

- $\nu$ — the policy exposure
- $\boldsymbol{x}$ — the risk factors of the insured
- $S$ — the aggregate loss given $\boldsymbol{x}$ and $\nu$
- $N$ — the claim count
- $Y_n$ — the amount of the $n$-th claim

with $N$ assumed independent of $\{Y_n\}_{n \ge 1}$ given $(\boldsymbol{x}, \nu)$.

## Modeling the Claim Count $N$ With GLM

We model $N \mid (\boldsymbol{x}, \nu)$ via a Poisson GLM with a log link. The expected claim count is assumed to scale proportionally with exposure $\nu$, i.e.,

$$
\mathbb{E}[N \mid \boldsymbol{x}, \nu] = \lambda(\boldsymbol{x}) \cdot \nu, \qquad \lambda(\boldsymbol{x}) = \exp(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)
$$

where $\lambda(\boldsymbol{x})$ is the claim frequency per unit exposure given risk factors $\boldsymbol{x}$. Taking logs yields the linear predictor actually fit by the GLM:

$$
\log(\mathbb{E}[N \mid \boldsymbol{x}, \nu]) = \log(\nu) + \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

with $\log(\nu)$ entered as an **offset**.

### Train/Test Split

Since our goal is to predict future risk from historical data, instead of splitting the dataset randomly, we split the dataset by year and train the model on years 7–8 and hold out year 9 for model testing.

To confirm this split is meaningful, we first test for a year effect via a **likelihood-ratio test** (LRT), comparing a model with `year` (as a factor) against one without. This yields statistically significant evidence of a trend in claim frequency over time ($p \approx 1.8 \times 10^{-77}$). Relative to year 7, frequency is estimated to be 3.9% and 8.5% higher in years 8 and 9 respectively.

For the training model itself, `year` is entered as a **numeric** covariate rather than a factor, imposing a linear-trend assumption between years 7 and 8. This is necessary for our purpose, since `year = 9` is unseen during training.