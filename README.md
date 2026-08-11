# ***Claim Prediction***

## ***The Data***

Before we start modeling, we discuss how we interpret several columns in this dataset. The documentation (see CASdatasets-manual.pdf) uses terms such as "Card Gestionario" and "Card Debitore" without further explanation. We recognize these as referring to Italy's **CARD system** (*Convenzione tra Assicuratori per il Risarcimento Diretto* — the Direct Compensation Convention among Insurers), even though the manual itself does not state this explicitly.

This system creates two distinct roles an insurer can hold on a given claim:

> - **Gestionario**: The insurer of the **not-at-fault** policyholder. The insurer pays its own policyholder directly, then is owed reimbursement from the at-fault driver's insurer.
> - **Debitore**: The insurer of the **at-fault** policyholder. The insurer owes reimbursement to the insurer who already compensated the victim under their Gestionario role.

The manual describes the column as follows:

> `cost_fcd` Total claim amount for Forfait Card Gestionario (FCD) claims.  
> `num_fcd` Number of Forfait Card Gestionario (FCD) claims.

This is inconsistent on two levels. First, "Gestionario" refers to the not-at-fault role, yet the abbreviation "FCD" mirrors `cost_cd` — **Card Debitore**, the at-fault role. Second, against the actual data, `euMTPL` does not contain `cost_fcd`/`num_fcd` columns at all. Instead, `cost_fcg` and `num_fcg` are provided. However, unlike `cost_cg`, which can take negative values, `cost_fcg` contains no negative entries. We therefore interpret `cost_fcg`/`num_fcg` in the data as **Forfait Card Debitore** claims.

> [!TIP]
> From now on, we refer to these columns as `cost_fcd` and `num_fcd` in this document, matching their semantic roles (Debitore) rather than their literal names in the data (`cost_fcg`/`num_fcg`).

Since we are interested in losses caused by our own policyholders, we only consider ``num_fcd``, ``cost_fcd``, ``num_cd`` and ``cost_cd`` in this project.

## ***Model Assumptions***

Our loss model is described by the **Compound Poisson Model**, which takes the form

$$
S \mid (\mathbf{x}, \nu) = \sum_{j=1}^{N} Y_j, \quad N \mid (\mathbf{x}, \nu) \sim \text{Poi}(\lambda(\mathbf{x}) \cdot \nu), \quad Y_j \mid \mathbf{x} \stackrel{\text{i.i.d.}}{\sim} \Gamma(\alpha, \theta(\mathbf{x}))
$$

where:

- $\nu$ — the policy exposure
- $\mathbf{x}$ — the risk factors of the insured
- $S$ — the aggregate loss given $\mathbf{x}$ and $\nu$
- $N$ — the claim count
- $Y_j$ — the amount of the $j$-th claim

with $N$ assumed independent of $\lbrace Y_j\rbrace_{j \ge 1}$ given $(\mathbf{x}, \nu)$.

## ***Modeling Claim Count*** $N$ ***With Poisson GLM***

We model $N \mid (\mathbf{x}, \nu)$ via a Poisson GLM with a log link. The expected claim count is assumed to scale proportionally with exposure $\nu$, i.e.,

$$
\mathbb{E}[N \mid \mathbf{x}, \nu] = \lambda(\mathbf{x}) \cdot \nu, \qquad \lambda(\mathbf{x}) = \exp(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)
$$

where $\lambda(\mathbf{x})$ is the claim frequency per unit exposure given risk factors $\mathbf{x}$. Taking logs yields the linear predictor actually fit by the GLM:

$$
\log(\mathbb{E}[N \mid \mathbf{x}, \nu]) = \log(\nu) + \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

with $\log(\nu)$ entered as an **offset**.

### ***Train/Test Split***

Since our goal is to predict future risk from historical data, instead of splitting the dataset randomly, we split the dataset by year and train the model on years 7–8 and hold out year 9 for model testing.

To confirm this split is meaningful, we first test for a year effect via a **likelihood-ratio test** (LRT), comparing a model with `year` (as a factor) against one without. This yields statistically significant evidence of a trend in claim frequency over time ($p \approx 9.830848\times 10^{-163}$). Relative to year 7, frequency is estimated to be 9% and 17% higher in years 8 and 9 respectively.

For the training model itself, `year` is entered as a **numeric** covariate rather than a factor, imposing a linear-trend assumption between years 7 and 8. This is necessary for our purpose, since `year = 9` is unseen during training.

### ***Model Accuracy***

After fitting the model, we would like to assess the model fit. 
#### ***Overall Model Significance (Likelihood-Ratio test)***
This test answers the question, whether the set of risk factors has predictive power. The test is formulated as following
> - $H_0$: $\beta=\mathbf{0}$. (None of the risk factors has predictive power)
> - $H_1$: $\beta_i\neq0$ for some $i$. (At least one of the risk factors has predictive power)

Test statistic:
$$\Delta D=D_\text{Null}-D_\text{Res}\approx 4149$$
Under $H_0$, $\Delta D\sim \chi^2_{\Delta \text{df}}$ (s. Wilks' theorem), where $\Delta \text{df}=1,578,298-1,578,282=16$. The $p$-value is approximately 0. Hence, we **reject** $H_0$. 

#### ***Model Specification (Deviance Goodness-of-Fit Test)***
This test tells us how well the Poisson GLM specifies our population, from which the data are collected.
>- $H_0$: The Poisson GLM correctly describes the population.
>- $H_1$: The Poisson GLM is misspecified.

Under $H_0$, $D_{\text{Res}} \sim \chi^2_{\text{df}_{\text{residual}}}$. We compute the upper-tail $p$-value:

$$p\text{-value} = P\left(\chi^2_{1,578,282} \ge 610732\right) \approx 1.0$$

Since $p \ge 0.05$, we **fail to reject $H_0$**, indicating no evidence of structural underfit.

#### ***Overdispersion (Pearson's Chi-Squared Test)***
We test whether the data exhibits overdispersion ($\text{Var}(Y) > \mu$), violating the standard Poisson variance assumption ($\phi = 1$):
> - **$H_0$:** $\phi = 1$ (Equidispersion holds; variance equals the mean).
> - **$H_1$:** $\phi > 1$ (The data is overdispersed).

We calculate the sum of squared Pearson residuals $X^2=\sum_{i=1}^n\frac{(y\_i-\hat{\mu}\_i)^2}{\hat{\mu}\_i}$, which, as a sum of squared standardized errors, follows a $\chi^2_{\text{df}_{\text{residual}}}$ distribution under $H_0$. Computing the upper-tail $p$-value yields $p\text{-value} \approx 0.0$. This is a strong statistical evidence of overdispersion. In particular, our estimate of dispersion parameter $\phi$ is $\hat{\phi}=\frac{X^2}{N_{\text{train}}-p}\approx 1.27$.

Note this apparently conflicts with the Deviance Goodness-of-Fit result above. Both the deviance and Pearson statistics are only asymptotically $\chi^2$-distributed as 
fitted means $\hat\mu_i$ grow large (not merely as $n$ grows large). We therefore do not treat the GoF test's $p \approx 1$ as reliable evidence of correct specification. The dispersion estimate $\hat{\phi} \approx 1.27$, by contrast, is an estimator that does not rely on this approximation, so we treat it as the more trustworthy diagnostic and conclude that the overdispersion is real.

#### ***Final Verdict***

In conclusion, our set of risk factors provides genuine predictive power. The Deviance Goodness-of-Fit test's conclusion of correct specification is not considered reliable here, given the sparsity of the data; the dispersion estimate instead indicates meaningful overdispersion ($\hat\phi \approx 1.27$), meaning Poisson-reported standard errors likely understate the truth by a factor of roughly $\sqrt{\hat\phi}\approx 1.127$. This may be resolved by other modeling approaches, e.g. **negative binomial GLM** or **quasi-Poisson**. At this point, we don't refit either — we note this as a limitation and continue with the Poisson GLM, moving on to modeling claim severity.

## ***Modeling Claim Severity*** $Y_j$

### ***From Individual Claims to Grouped Averages***
> [!TIP]
> In this section $i$ indexes the rows (policies) and $j$ indexes the individual claims.

Since our data does not include every single claim amount, our gamma GLM is not straightforward anymore. However, given the ***closure of Gamma distribution under summation***, we can adjust the weight of each single row (policy) and fit a gamma GLM. Mathematically, given a risk profile $\mathbf{x}\_i$, we observe its corresponding average severity 

$$
\bar{Y}\_i\mid (N\_i=n\_i, \mathbf{x}\_i) = \frac{1}{n\_i}\sum_{j=1}^{n\_i}Y\_{ij}\sim \Gamma(n\_i\alpha, \frac{\theta(\mathbf{x}\_i)}{n\_i}).
$$

Furthermore, Gamma distribution is a member of exponential family, thus the variance takes the form $$\text{Var}(\bar{Y}_i)=\frac{\phi \cdot (\alpha \theta(\mathbf{x}_i))^2}{\omega},$$ where $\phi=1/\alpha$ and $\omega=n_i$ given $N_i=n_i$.