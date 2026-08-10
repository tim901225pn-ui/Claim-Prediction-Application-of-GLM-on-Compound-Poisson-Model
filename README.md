# **Claim Prediction**

## Model Assumptions

Our loss model is described by the **Compound Poisson Model**, which takes the form

$$
S \mid (\mathbf{x}, \nu) = \sum_{n=1}^{N} Y_n, \quad N \mid (\mathbf{x}, \nu) \sim \text{Poi}(\lambda(\mathbf{x}) \cdot \nu), \quad Y_n \mid \mathbf{x} \stackrel{\text{i.i.d.}}{\sim} d(\mathbf{x})
$$

where:

- $\nu$ — the policy exposure
- $\mathbf{x}$ — the risk factors of the insured
- $S$ — the aggregate loss given $\mathbf{x}$ and $\nu$
- $N$ — the claim count
- $Y_n$ — the amount of the $n$-th claim

with $N$ assumed independent of $\lbrace Y_n\rbrace_{n \ge 1}$ given $(\mathbf{x}, \nu)$.

## Modeling the Claim Count $N$ With GLM

We model $N \mid (\mathbf{x}, \nu)$ via a Poisson GLM with a log link. The expected claim count is assumed to scale proportionally with exposure $\nu$, i.e.,

$$
\mathbb{E}[N \mid \mathbf{x}, \nu] = \lambda(\mathbf{x}) \cdot \nu, \qquad \lambda(\mathbf{x}) = \exp(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)
$$

where $\lambda(\mathbf{x})$ is the claim frequency per unit exposure given risk factors $\mathbf{x}$. Taking logs yields the linear predictor actually fit by the GLM:

$$
\log(\mathbb{E}[N \mid \mathbf{x}, \nu]) = \log(\nu) + \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

with $\log(\nu)$ entered as an **offset**.

### Train/Test Split

Since our goal is to predict future risk from historical data, instead of splitting the dataset randomly, we split the dataset by year and train the model on years 7–8 and hold out year 9 for model testing.

To confirm this split is meaningful, we first test for a year effect via a **likelihood-ratio test** (LRT), comparing a model with `year` (as a factor) against one without. This yields statistically significant evidence of a trend in claim frequency over time ($p \approx 1.8 \times 10^{-77}$). Relative to year 7, frequency is estimated to be 3.9% and 8.5% higher in years 8 and 9 respectively.

For the training model itself, `year` is entered as a **numeric** covariate rather than a factor, imposing a linear-trend assumption between years 7 and 8. This is necessary for our purpose, since `year = 9` is unseen during training.

### Model Accuracy

After fitting the model, we would like to assess the model fit. 
#### **Likelihood-Ratio test**
This test answers the question, whether the set of risk factors has predictive power. The test is formulated as following
> - $H_0$: $\beta=\mathbf{0}$. (None of the risk factors has predictive power)
> - $H_1$: $\beta_i\neq0$ for some $i$. (At least one of the risk factors has predictive power)

Test statistic:
$$\Delta D=D_\text{Null}-D_\text{Res}\approx 8,626$$
Under $H_0$, $\Delta D\sim \chi^2_{\Delta \text{df}}$ (s. Wilks' theorem), where $\Delta \text{df}=1,578,298-1,578,282=16$. The $p$-value is approximately 0. Hence, we **reject** $H_0$. 

#### **Deviance Goodness-of-Fit Test**
This test tells us how well the Poisson GLM specifies our population, from which the data are collected.
>- $H_0$: The Poisson GLM correctly describes the population.
>- $H_1$: The Poisson GLM is misspecified.

Under $H_0$, $D_{\text{Res}} \sim \chi^2_{\text{df}_{\text{residual}}}$. We compute the upper-tail $p$-value:

$$p\text{-value} = P\left(\chi^2_{1,578,282} \ge 1,052,742\right) \approx 1.0$$

Since $p \ge 0.05$, we **fail to reject $H_0$**, indicating no evidence of structural underfit.

#### **Pearson's Chi-Squared Test**
We test whether the data exhibits overdispersion ($\text{Var}(Y) > \mu$), violating the standard Poisson variance assumption ($\phi = 1$):
> - **$H_0$:** $\phi = 1$ (Equidispersion holds; variance equals the mean).
> - **$H_1$:** $\phi > 1$ (The data is overdispersed).

We calculate the sum of squared Pearson residuals $X^2 = \sum_{i=1}^N e_i^2$, which, as a sum of squared standardized errors, follows a $\chi^2_{\text{df}_{\text{residual}}}$ distribution under $H_0$. Computing the upper-tail $p$-value yields $p\text{-value} \approx 0.0$. This is a strong statistical evidence of overdispersion.

#### Final Verdict
In conclusion, our set of risk factors provides predictive power and the Poisson GLM is well specified. However, **Pearson's Chi-Squared Test** shows the evidence of overdispersion. This may be resolved by other modeling approaches, e.g. **negative binomial GLM**. At this point, we don't fit a negative binomial GLM. Instead, we continue with Poisson GLM and move on to modeling the claim severity.