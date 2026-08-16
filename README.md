# Claim Prediction — European MTPL Portfolio

A statistical learning project modeling **claim frequency and severity** for at-fault motor liability claims, using the `euMTPL` dataset from [CASdatasets](http://cas.uqam.ca/) (`fremotorclaim`). Built as an applied companion to SRM exam prep — Compound Poisson-Gamma loss modeling via GLMs, with a full diagnostic and validation.

## What this project does

We model the **aggregate loss** an insurer incurs specifically from claims where **our own policyholder was at fault** (as opposed to claims where the policyholder was the victim), using a Compound Poisson–Gamma framework:

- **Claim frequency** — a Poisson GLM (log link, exposure as offset)
- **Claim severity** — a Gamma GLM (log link, weighted by claim count),
  fit on grouped/averaged claim amounts since individual claim-level data isn't available

Both models are trained on years 7–8 and validated against held-out year 9, with a full battery of diagnostics: likelihood-ratio tests, goodness-of-fit, overdispersion, collinearity, leverage/outliers, and residual-based checks of distributional adequacy.

## Key findings

- At-fault claim **frequency** rose meaningfully over the observed years (~9% and ~17% higher in years 8 and 9 vs. year 7).
- At-fault claim **severity**, by contrast, *declined* by ~12% year-over-year — frequency and severity moved in opposite directions.
- The Poisson frequency model shows real but modest **overdispersion** ($\hat\phi\approx1.27$).
- The Gamma severity model's tail is **too light**: two independent residual diagnostics (standardized deviance vs. randomized quantile residuals) confirm this is a genuine specification issue, not a diagnostic artifact. Heavier-tailed alternatives (inverse Gaussian, lognormal, Tweedie) are natural next steps.

## Project structure

```
├── README.md                          you are here
├── docs/
│   ├── CASdatasets-manual.pdf         original data documentation
│   └── methodology.md                 model assumptions & statistical reasoning
└── notebooks/
    ├── euMTPL.rda                     data file 
    ├── fit.modle.reduced.rds          saved model fit 
    ├── fitted.model.full.rds          saved model fit 
    └── RESULT.ipynb                   full R analysis, code + output

```

**New to this project? Start here:**
- Want the math behind the models? → [`docs/methodology.md`](docs/methodology.md)
- Want to see how well the models actually fit? → [`docs/RESULT.md`](docs/RESULT.md)

## Data

[`euMTPL`](http://cas.uqam.ca/) — three years of experience from a European (Italian) Motor Third-Party Liability portfolio, ~2.37M policy-year rows. We restrict to claims where the policyholder was at fault (Card Debitore / Forfait Card Debitore); see [`docs/methodology.md`](docs/methodology.md) for why.

## Requirements

- R (≥ 4.6), packages: `car`, `statmod`

## Author's note

Built while studying for SRM — feedback and corrections welcome.