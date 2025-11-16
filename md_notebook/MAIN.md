# Bayesian SVAR analysis for macroeconomic and IRRBB forecasting with Python

## Introduction

This notebook demonstrates a full pipeline that links Bayesian SVAR modeling to Basel III interest rate risk in the banking book (IRRBB) metrics. The goal is to translate yield-curve and macroeconomic scenarios into consistent forecasts of bank-level Net Interest Income (NII) and Economic Value of Equity (EVE) responses.

The analysis proceeds in three layers:

Bayesian VAR estimation – building a reduced-form model for key macro and yield-curve factors.

Conditional forecasts – imposing Basel III and structural macro scenarios as exogenous paths and generating consistent projections.

IRRBB mapping – translating yield-curve shifts into simplified but realistic NII and EVE sensitivities calibrated to supervisory magnitudes.


```python
import requests, io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
```

## 1) ETL

This section documents how the macro–financial panel used in the BVAR is built. The objective is to end up with a single, quarterly, balanced dataset containing the variables that will later enter the VAR and the conditional forecasts.

All of the data is imported from the European Central Bank ECB Statistical Data Warehouse

We work with two broad blocks of variables:

### Yield-curve block

Spot rates - Government bond, nominal, all issuers whose rating is triple A - Euro area (changing composition), Euro area

We have 3 maturities: 1y, 5y and 10y. From those, we apply simple formulas to estimate Nelson-Siegel-like factors: level (the average across the 3 maturities), slope (10y - 1y) and curvature (2 * 5y - 1y - 10y) 

### Macroeconomic block

gdp_q_ann – real GDP growth at market prices, Euro area 20, quarterly frequency, annualized.

infl_q_ann – Euro area HICP inflation at quarterly frequency, annualized.

gg_deficit_pct_gdp - Government deficit(-) or surplus(+) (as % of GDP), Euro area 20, Quarterly 

gg_debt_pct_gdp - Government debt (consolidated) (as % of GDP), Euro area 20, Quarterly 

policy_rate - Level of rate of main refinancing operations - fixed rate tenders (fixed rate) (date of changes), Euro area

We winsorize the GDP series at 15/-15 during the COVID shock

These are the variables referenced later in the notebook in var_order, so we keep the names consistent between ETL and modeling.


```python
from src import etl
```

    OK  https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.000000.4.INX?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/MNA/Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/GFS/Q.N.I9.W0.S13.S1._Z.B.B9._Z._Z._Z.XDC_R_B1GQ_CY._Z.S.V.CY._T?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/GFS/Q.N.I9.W0.S13.S1.C.L.LE.GD.T._Z.XDC_R_B1GQ_CY._T.F.V.N._T?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y?startPeriod=2000-01&format=csvdata
    OK  https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?startPeriod=2000-01&format=csvdata
    Saved data_raw/hicp_index_monthly.csv
    Saved data_raw/gdp_clv_q_sa_levels.csv
    Saved data_raw/policy_rate_mro.csv
    Saved data_raw/gov_deficit_pct_gdp_q.csv
    Saved data_raw/gov_debt_pct_gdp_q.csv
    Saved data_raw/yc_spot_1y_daily.csv
    Saved data_raw/yc_spot_5y_daily.csv
    Saved data_raw/yc_spot_10y_daily.csv
    Saved data_raw/hicp_quarterly_inflation_from_index.csv
    Saved data_raw/yc_spot_1y_quarterly_mean.csv
    Saved data_raw/yc_spot_5y_quarterly_mean.csv
    Saved data_raw/yc_spot_10y_quarterly_mean.csv
    Saved data_raw/policy_rate_mro.csv
    Saved data_raw/gdp_growth_from_levels_q.csv
    inflation : 2000-03-31 → 2025-12-31 (n= 104 )
    gdp : 2000-03-31 → 2025-09-30 (n= 103 )
    policy_rate : 2000-03-31 → 2025-06-30 (n= 102 )
    deficit : 2002-03-31 → 2025-06-30 (n= 94 )
    debt : 2000-03-31 → 2025-06-30 (n= 102 )
    yc1y : 2004-09-30 → 2025-12-31 (n= 86 )
    yc5y : 2004-09-30 → 2025-12-31 (n= 86 )
    yc10y : 2004-09-30 → 2025-12-31 (n= 86 )
    

Our sample period starts at 2004 Q3 and ends at 2025 Q2


```python
panel = pd.read_csv("data/quarterly_panel_modelvars.csv", parse_dates=["date"], index_col="date")
panel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>infl_q_ann</th>
      <th>gdp_q_ann</th>
      <th>policy_rate</th>
      <th>gg_deficit_pct_gdp</th>
      <th>gg_debt_pct_gdp</th>
      <th>yc_spot_1y</th>
      <th>yc_spot_5y</th>
      <th>yc_spot_10y</th>
      <th>level</th>
      <th>slope_10y_1y</th>
      <th>curvature_ns_like</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-09-30</th>
      <td>0.369011</td>
      <td>0.901423</td>
      <td>4.25</td>
      <td>-2.997</td>
      <td>71.161</td>
      <td>2.291667</td>
      <td>3.372940</td>
      <td>4.104690</td>
      <td>3.256432</td>
      <td>1.813023</td>
      <td>0.349523</td>
    </tr>
    <tr>
      <th>2004-12-31</th>
      <td>2.462038</td>
      <td>1.774910</td>
      <td>4.25</td>
      <td>-2.897</td>
      <td>69.721</td>
      <td>2.205564</td>
      <td>3.102433</td>
      <td>3.849834</td>
      <td>3.052610</td>
      <td>1.644269</td>
      <td>0.149469</td>
    </tr>
    <tr>
      <th>2005-03-31</th>
      <td>0.350486</td>
      <td>1.065693</td>
      <td>4.25</td>
      <td>-2.885</td>
      <td>71.059</td>
      <td>2.213501</td>
      <td>3.024397</td>
      <td>3.635017</td>
      <td>2.957638</td>
      <td>1.421516</td>
      <td>0.200277</td>
    </tr>
    <tr>
      <th>2005-06-30</th>
      <td>4.843364</td>
      <td>2.474232</td>
      <td>4.25</td>
      <td>-2.917</td>
      <td>71.692</td>
      <td>2.067228</td>
      <td>2.742555</td>
      <td>3.392596</td>
      <td>2.734126</td>
      <td>1.325368</td>
      <td>0.025285</td>
    </tr>
    <tr>
      <th>2005-09-30</th>
      <td>1.460457</td>
      <td>2.972809</td>
      <td>4.25</td>
      <td>-2.711</td>
      <td>71.186</td>
      <td>2.084327</td>
      <td>2.645390</td>
      <td>3.219840</td>
      <td>2.649852</td>
      <td>1.135513</td>
      <td>-0.013388</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-06-30</th>
      <td>6.009073</td>
      <td>0.900181</td>
      <td>4.25</td>
      <td>-3.436</td>
      <td>87.685</td>
      <td>3.258067</td>
      <td>2.487573</td>
      <td>2.557599</td>
      <td>2.767746</td>
      <td>-0.700468</td>
      <td>-0.840519</td>
    </tr>
    <tr>
      <th>2024-09-30</th>
      <td>0.980509</td>
      <td>1.618255</td>
      <td>3.65</td>
      <td>-3.247</td>
      <td>87.709</td>
      <td>2.834863</td>
      <td>2.180941</td>
      <td>2.377072</td>
      <td>2.464292</td>
      <td>-0.457790</td>
      <td>-0.850053</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>0.904573</td>
      <td>1.708808</td>
      <td>3.15</td>
      <td>-3.064</td>
      <td>87.080</td>
      <td>2.326712</td>
      <td>2.058374</td>
      <td>2.354717</td>
      <td>2.246601</td>
      <td>0.028006</td>
      <td>-0.564681</td>
    </tr>
    <tr>
      <th>2025-03-31</th>
      <td>1.363507</td>
      <td>2.249232</td>
      <td>2.65</td>
      <td>-2.965</td>
      <td>87.723</td>
      <td>2.188781</td>
      <td>2.282415</td>
      <td>2.650659</td>
      <td>2.373952</td>
      <td>0.461878</td>
      <td>-0.274610</td>
    </tr>
    <tr>
      <th>2025-06-30</th>
      <td>4.736000</td>
      <td>0.508239</td>
      <td>2.15</td>
      <td>-2.847</td>
      <td>88.173</td>
      <td>1.822381</td>
      <td>2.111617</td>
      <td>2.646672</td>
      <td>2.193557</td>
      <td>0.824291</td>
      <td>-0.245819</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 11 columns</p>
</div>



A glance at what our time series look like


```python
panel.plot(subplots=True, layout=(4,3), figsize=(12, 8), legend=True)
```




    array([[<Axes: xlabel='date'>, <Axes: xlabel='date'>,
            <Axes: xlabel='date'>],
           [<Axes: xlabel='date'>, <Axes: xlabel='date'>,
            <Axes: xlabel='date'>],
           [<Axes: xlabel='date'>, <Axes: xlabel='date'>,
            <Axes: xlabel='date'>],
           [<Axes: xlabel='date'>, <Axes: xlabel='date'>,
            <Axes: xlabel='date'>]], dtype=object)




    
![png](output_7_1.png)
    


## 2) Bayesian reduced-form VAR

In this section we set up the reduced-form Bayesian VAR that will serve as the engine for all conditional forecasts. The idea is to have a flexible multivariate time-series model that captures the joint dynamics of the macro block (GDP, inflation) and the financial block (yield-curve factors).

Vector Autoregressions (VARs) are a core tool in empirical macroeconomics because they model how key economic and financial variables move together over time without requiring strong theoretical assumptions. By capturing the persistence, co-movement, and shock propagation between variables such as interest rates, inflation, GDP, and yield-curve factors, VARs provide a data-driven representation of macroeconomic dynamics. This makes them ideal for forecasting and policy analysis.

A Bayesian treatment is useful in our case because:

- the sample is not huge (quarterly data),

- we want to retain parameter uncertainty for scenario analysis,

- and we later need the full posterior to run conditional (Waggoner–Zha style) forecasts.

### Model

We model a VAR($p$) with exogenous dummies for COVID quarters:

- Let $y_t \in \mathbb{R}^K$ be the vector of endogenous variables at time $t$.
- Let $z_t \in \mathbb{R}^q$ be the vector of exogenous dummies (in this case the COVID quarters).
- Define the lagged regressor vector

$$
x_t =
\begin{bmatrix}
y_{t-1}^\prime & y_{t-2}^\prime & \dots & y_{t-p}^\prime
\end{bmatrix}^\prime
\in \mathbb{R}^{Kp}.
$$

The observation equation for $t = p+1,\dots,T$ is

$$
y_t = c + B x_t + \Gamma z_t + u_t, \qquad
u_t \sim \mathcal{N}(0,\Sigma),
$$

where:

- $c \in \mathbb{R}^K$ is the intercept,
- $B \in \mathbb{R}^{K \times Kp}$ stacks the VAR coefficients  
  $$
  B = \begin{bmatrix} A_1 & A_2 & \dots & A_p \end{bmatrix},
  $$
  with $A_\ell$ the $K\times K$ coefficient matrix on lag $\ell$,
- $\Gamma \in \mathbb{R}^{K \times q}$ collects the coefficients on the dummy regressors,
- $\Sigma \in \mathbb{R}^{K\times K}$ is the residual covariance matrix.

In our model we set $p=2$

---

### Priors

**Residual covariance (LKJ prior)**

Over the last decade, the LKJ prior has become the standard way to model correlation matrices in Bayesian multivariate analysis because it separates correlations from variances and avoids the strong, unintended constraints imposed by inverse-Wishart conjugate priors. Its simple shape parameter and compatibility with Cholesky factorization make it far more stable and interpretable, which is why probabilistic programming frameworks like PyMC, Stan, and TFP now use LKJ by default.

We parameterize $\Sigma$ via its Cholesky factor:

$$
L \sim \text{LKJCholeskyCov}(\eta = 2,\ \sigma),
\qquad
\Sigma = L L^\prime,
$$

with marginal scales

$$
\sigma_i \sim \text{HalfNormal}(1.0), \quad i = 1,\dots,K.
$$

---

### VAR coefficients with Minnesota-style scales

#### Overview of Minnesota priors

A key feature of Bayesian VARs in macroeconomics is the use of **Minnesota priors**, which impose structured shrinkage on the autoregressive coefficients.  
This prevents overfitting when the number of parameters is large relative to the time span of quarterly data, while still allowing the model to capture important dynamics. 
There are 3 shrinkage parameters: $\lambda_1$ which controls autoregressive behavior of the variable itself, $\lambda_2$ which controls cross-variable effects and $\lambda_3$ is responsible for the decay of longer lags.

The prior assumes that:

- each variable in the system follows its own autoregressive process (so own lags matter most),
- cross-variable lags are likely to have smaller coefficients,
- and coefficients on longer lags decay roughly as $1 / L^{\lambda_3}$.

Mathematically, the prior for each coefficient $\beta_{ijL}$ on variable $j$ at lag $L$ in equation $i$ is:

$$
\beta_{ijL} \sim \mathcal{N}(0, S_{ijL}^2),
$$

with scale

$$
S_{ijL} = \lambda_1 \frac{\sigma_i}{\sigma_j} \frac{\lambda_2^{\mathbf{1}(i \ne j)}}{L^{\lambda_3}},
$$

where:

- $\lambda_1$: **overall tightness** (smaller → stronger shrinkage; 0.2 is moderate),
- $\lambda_2$: **cross-variable penalty** (down-weights other variables’ lags),
- $\lambda_3$: **lag decay** (ensures higher lags are more heavily shrunk),
- $\sigma_i$: residual scale of variable \( i \), typically its standard deviation.

The prior shrinks coefficients on longer lags and on “unrelated” equations, which stabilizes forecasts and improves acceptance of sign/level constraints later.

Our MCMC setup (PyMC + NUTS) would allow us to set priors on the $\lambda$s themselves, but we leave them as fixed values for simplicity.

#### Minnesota priors in our model

For each element of $B$:

$$
B_{ij} \sim \mathcal{N}\bigl(0,\ S_{\beta,ij}^2\bigr),
$$

where the prior standard deviations $S_{\beta,ij}$ come from the Minnesota prior using the variable standard deviations and hyperparameters $\lambda_1, \lambda_2, \lambda_3$.

Compactly:

$$
B \sim \mathcal{N}(0,\ S_\beta \odot S_\beta),
$$

with $S_\beta \in \mathbb{R}^{K \times Kp}$ the matrix of prior scales.

---

### Intercept

$$
c \sim \mathcal{N}(0,\ 5^2 I_K).
$$

---

### Exogenous dummy coefficients

We use a global shrinkage prior:

$$
\tau_D \sim \text{HalfNormal}(0.5),
$$

$$
\Gamma_{ik} \sim \mathcal{N}(0,\ \tau_D^2), \qquad i = 1,\dots,K,\ k = 1,\dots,q.
$$

---

### Likelihood (stacked form)

Define the stacked matrices:

$$
Y = \begin{bmatrix} y_{p+1}^\prime \\ \vdots \\ y_T^\prime \end{bmatrix},
\qquad
X = \begin{bmatrix} x_{p+1}^\prime \\ \vdots \\ x_T^\prime \end{bmatrix},
\qquad
Z = \begin{bmatrix} z_{p+1}^\prime \\ \vdots \\ z_T^\prime \end{bmatrix}.
$$

The likelihood is:

$$
y_t \mid B,c,\Gamma,\Sigma \sim \mathcal{N}(\mu_t,\Sigma),
\qquad
\mu_t = c + B x_t + \Gamma z_t.
$$

### Posterior objects

The estimation returns an `arviz.InferenceData` object (`idata`) with, for each posterior draw:

- the stacked coefficient matrix $B=[A_1, \ldots, A_p]$

- the intercept $c$

- the covariance matrix $\Sigma$

These are exactly the ingredients we need to:

- build the companion form and unconditional forecasts,

- impose linear constraints for Basel yield paths,

- re-simulate conditional distributions of future $y_{t+h}$

### Code

For Bayesian estimation, this project uses the PyMC probabilistic programming framework and the ArviZ diagnostics library.

PyMC provides a Python-native interface for defining Bayesian models directly in terms of probability distributions.
It relies on advanced Markov Chain Monte Carlo (MCMC) and variational inference backends, allowing scalable sampling even for high-dimensional models like Bayesian VARs.
Its integration with NumPy and Theano/PyTensor makes it easy to vectorize likelihoods and run efficient, gradient-based samplers such as NUTS (No-U-Turn Sampler).

ArviZ complements PyMC by offering powerful posterior analysis and visualization tools: trace plots, posterior predictive checks, and convergence diagnostics such as $\hat{R}$ and effective sample size.
It standardizes outputs through the `InferenceData` object, which stores chains, priors, and observed data in an easily accessible format.

Together, PyMC and ArviZ make it possible to:

- Estimate fully Bayesian macroeconomic models with proper uncertainty quantification,

- Reuse posterior draws for scenario simulations (conditional forecasts),

- Integrate seamlessly into a Python-based data and visualization pipeline.

With this, the BVAR section of our project does the following:

1. estimate the BVAR model

2. load the saved BVAR posterior (`idata`),

3. extract $B$, $c$, and $\Sigma$ for all posterior draws,

4. read the lag order $p$,

5. and define the `var_order` vector that will be used consistently in the rest of the notebook.

We model a VAR($p$) with exogenous dummies:

- Let $y_t \in \mathbb{R}^K$ be the vector of endogenous variables at time $t$.
- Let $z_t \in \mathbb{R}^q$ be the vector of exogenous dummies (in this case the COVID quarters).
- Define the lagged regressor vector

$$
x_t =
\begin{bmatrix}
y_{t-1}^\prime & y_{t-2}^\prime & \dots & y_{t-p}^\prime
\end{bmatrix}^\prime
\in \mathbb{R}^{Kp}.
$$

The observation equation for $t = p+1,\dots,T$ is

$$
y_t = c + B x_t + \Gamma z_t + u_t, \qquad
u_t \sim \mathcal{N}(0,\Sigma),
$$

where:

- $c \in \mathbb{R}^K$ is the intercept,
- $B \in \mathbb{R}^{K \times Kp}$ stacks the VAR coefficients  
  $$
  B = \begin{bmatrix} A_1 & A_2 & \dots & A_p \end{bmatrix},
  $$
  with $A_\ell$ the $K\times K$ coefficient matrix on lag $\ell$,
- $\Gamma \in \mathbb{R}^{K \times q}$ collects the coefficients on the dummy regressors,
- $\Sigma \in \mathbb{R}^{K\times K}$ is the residual covariance matrix.

---

### Priors

**Residual covariance (LKJ prior)**

We parameterize $\Sigma$ via its Cholesky factor:

$$
L \sim \text{LKJCholeskyCov}(\eta = 2,\ \sigma),
\qquad
\Sigma = L L^\prime,
$$

with marginal scales

$$
\sigma_i \sim \text{HalfNormal}(1.0), \quad i = 1,\dots,K.
$$

---

### VAR coefficients with Minnesota-style scales

For each element of $B$:

$$
B_{ij} \sim \mathcal{N}\bigl(0,\ S_{\beta,ij}^2\bigr),
$$

where the prior standard deviations $S_{\beta,ij}$ come from the Minnesota prior using the variable standard deviations and hyperparameters $\lambda_1, \lambda_2, \lambda_3$.

Compactly:

$$
B \sim \mathcal{N}(0,\ S_\beta \odot S_\beta),
$$

with $S_\beta \in \mathbb{R}^{K \times Kp}$ the matrix of prior scales.

---

### Intercept

$$
c \sim \mathcal{N}(0,\ 5^2 I_K).
$$

---

### Exogenous dummy coefficients

We use a global shrinkage prior:

$$
\tau_D \sim \text{HalfNormal}(0.5),
$$

$$
\Gamma_{ik} \sim \mathcal{N}(0,\ \tau_D^2), \qquad i = 1,\dots,K,\ k = 1,\dots,q.
$$

---

### Likelihood (stacked form)

Define the stacked matrices:

$$
Y = \begin{bmatrix} y_{p+1}^\prime \\ \vdots \\ y_T^\prime \end{bmatrix},
\qquad
X = \begin{bmatrix} x_{p+1}^\prime \\ \vdots \\ x_T^\prime \end{bmatrix},
\qquad
Z = \begin{bmatrix} z_{p+1}^\prime \\ \vdots \\ z_T^\prime \end{bmatrix}.
$$

The likelihood is:

$$
y_t \mid B,c,\Gamma,\Sigma \sim \mathcal{N}(\mu_t,\Sigma),
\qquad
\mu_t = c + B x_t + \Gamma z_t.
$$


## 2) Bayesian reduced-form VAR

In this section we set up the reduced-form Bayesian VAR that will serve as the engine for all conditional forecasts. The idea is to have a flexible multivariate time-series model that captures the joint dynamics of the macro block (GDP, inflation) and the financial block (yield-curve factors).

### Model

We use a VAR(p) of the form

$$
y_t = c + A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma),
$$

where $y_t$ stacks the variables from the panel.

A Bayesian treatment is useful here because:

- the sample is not huge (quarterly data),

- we want to retain parameter uncertainty for scenario analysis,

- and we later need the full posterior to run conditional (Waggoner–Zha style) forecasts.

### Priors and lags

A key feature of Bayesian VARs in macroeconomics is the use of **Minnesota priors**, which impose structured shrinkage on the autoregressive coefficients.  
This prevents overfitting when the number of parameters is large relative to the time span of quarterly data, while still allowing the model to capture important dynamics. 
There are 3 shrinkage parameters: $\lambda_1$ which controls autoregressive behavior of the variable itself, $\lambda_2$ which controls cross-variable effects and $\lambda_3$ is responsible for the decay of longer lags.

The prior assumes that:

- each variable in the system follows its own autoregressive process (so own lags matter most),
- cross-variable lags are likely to have smaller coefficients,
- and coefficients on longer lags decay roughly as $1 / L^{\lambda_3}$.

Mathematically, the prior for each coefficient $\beta_{ijL}$ on variable $j$ at lag $L$ in equation $i$ is:

$$
\beta_{ijL} \sim \mathcal{N}(0, s_{ijL}^2),
$$

with scale

$$
s_{ijL} = \lambda_1 \frac{\sigma_i}{\sigma_j} \frac{\lambda_2^{\mathbf{1}(i \ne j)}}{L^{\lambda_3}},
$$

where:

- $\lambda_1$: **overall tightness** (smaller → stronger shrinkage; 0.2 is moderate),
- $\lambda_2$: **cross-variable penalty** (down-weights other variables’ lags),
- $\lambda_3$: **lag decay** (ensures higher lags are more heavily shrunk),
- $\sigma_i$: residual scale of variable \( i \), typically its standard deviation.


The prior shrinks coefficients on longer lags and on “unrelated” equations, which stabilizes forecasts and improves acceptance of sign/level constraints later.

Our MCMC setup would allow us to set priors on the $\lambda$s themselves, but we leave them as fixed values for simplicity.

### Posterior objects

The estimation returns an `arviz.InferenceData` object (`idata`) with, for each posterior draw:

- the stacked coefficient matrix $B=[A_1, \ldots, A_p]$

- the intercept $c$

- the covariance matrix $\Sigma$

These are exactly the ingredients we need to:

- build the companion form and unconditional forecasts,

- impose linear constraints for Basel yield paths,

- re-simulate conditional distributions of future $y_{t+h}$

### Code

For Bayesian estimation, this project uses the PyMC probabilistic programming framework and the ArviZ diagnostics library.

PyMC provides a Python-native interface for defining Bayesian models directly in terms of probability distributions.
It relies on advanced Markov Chain Monte Carlo (MCMC) and variational inference backends, allowing scalable sampling even for high-dimensional models like Bayesian VARs.
Its integration with NumPy and Theano/PyTensor makes it easy to vectorize likelihoods and run efficient, gradient-based samplers such as NUTS (No-U-Turn Sampler).

ArviZ complements PyMC by offering powerful posterior analysis and visualization tools: trace plots, posterior predictive checks, and convergence diagnostics such as $\hat{R}$ and effective sample size.
It standardizes outputs through the `InferenceData` object, which stores chains, priors, and observed data in an easily accessible format.

Together, PyMC and ArviZ make it possible to:

- Estimate fully Bayesian macroeconomic models with proper uncertainty quantification,

- Reuse posterior draws for scenario simulations (conditional forecasts),

- Integrate seamlessly into a Python-based data and visualization pipeline.

With this, the BVAR section of our project does the following:

1. estimate the BVAR model

2. load the saved BVAR posterior (`idata`),

3. extract $B$, $c$, and $\Sigma$ for all posterior draws,

4. read the lag order $p$,

5. and define the `var_order` vector that will be used consistently in the rest of the notebook.


```python
from src import bvar
```

    C:\Users\thoma\.conda\envs\pymc_env\Lib\site-packages\arviz\stats\diagnostics.py:596: RuntimeWarning: invalid value encountered in scalar divide
      (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    

Specification of our model:

`pm.model_to_graphviz` allows us to have a simple graph of our Bayesian model


```python
# 1) Load data (you already created this file)
df = panel

vars_used = [
    "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    "level","slope_10y_1y","curvature_ns_like"
]
Ydf = df[vars_used].dropna()
Y = Ydf.to_numpy()
T, K = Y.shape

# 2) Build lag matrices

p = 2
X, Yt = bvar.build_var_design(Y, p)        # X=(T_eff, K*p), Yt=(T_eff, K)
T_eff = Yt.shape[0]

# 3) Minnesota-style prior scales (simple, effective defaults)

S_beta = bvar.minnesota_scales(K, p, sigma=Ydf.std().values, lam1=0.20, lam2=0.5, lam3=1.0)

# 4) PyMC model (LKJ Cholesky prior on Σu; gallery pattern)

with pm.Model() as bvar_model:
    # Residual covariance Σu with LKJ prior
    sd_dist = pm.HalfNormal.dist(1.0, shape=K)
    chol, corr, sds = pm.LKJCholeskyCov(
        "chol", n=K, eta=2.0, sd_dist=sd_dist, compute_corr=True, store_in_trace=True
    )
    Sigma = pm.Deterministic("Sigma", chol @ chol.T)

    # Coefficients: B is K x (K*p); intercept c is K
    B = pm.Normal("B", mu=0.0, sigma=S_beta, shape=(K, K * p))
    c = pm.Normal("c", mu=0.0, sigma=5.0, shape=K)
    
    # COVID dummy
    Z = np.zeros((T-p, 1), dtype=float)
    Z[[63-p, 64-p], 0] = 1.0
    q = Z.shape[1]                 # number of dummy columns (1 or 2)
    Z_shared = pm.Data("Z", Z)     # register as data for PyMC

    # Coefficients for exogenous block: Γ has shape (K x q), one coefficient per VAR equation per dummy
    tau_D = pm.HalfNormal("tau_D", sigma=0.5)   # global shrinkage
    Gamma = pm.Normal("Gamma", mu=0.0, sigma=tau_D, shape=(K, q))

    # Mean and likelihood (vectorized over T_eff rows)
    mu = c + pt.dot(X, B.T) + pt.dot(Z_shared, Gamma.T) # (T_eff, K)
    pm.MvNormal("y", mu=mu, chol=chol, observed=Yt)     # use chol for stability

pm.model_to_graphviz(bvar_model)
```




    
![svg](output_13_0.svg)
    



Modern Bayesian inference relies heavily on gradient-based Monte Carlo algorithms, which offer far better efficiency than classical random-walk MCMC. PyMC uses Hamiltonian Monte Carlo (HMC) and its adaptive variant, the No-U-Turn Sampler (NUTS), to explore high-dimensional posterior distributions quickly and with minimal tuning. HMC treats sampling as a physics problem—following simulated “trajectories” informed by the posterior’s gradients—allowing it to take large, informed steps and avoid the slow diffusion of traditional MCMC. NUTS improves on HMC by automatically adjusting trajectory lengths, removing the need for manual tuning. Under the hood, PyMC can now run these algorithms through BlackJAX, a JAX-based engine that provides fast, differentiable, and hardware-accelerated (CPU/GPU) implementations of HMC/NUTS. This combination delivers stable, scalable Bayesian inference even for complex models such as high-dimensional VARs.

For our sampling we use BlackJAX, but configure a logical check to fall back to the standard PyMC sampler if issues arise


```python
# Sampler (BlackJAX NUTS if installed; fallback to pm.sample)
from pymc.sampling.jax import sample_blackjax_nuts
import jax
jax.config.update("jax_enable_x64", True)  # recommended for numerical stability
with bvar_model:
    try:
        idata = sample_blackjax_nuts(draws=5000, tune=1500, chains=4, target_accept=0.9, random_seed=123, idata_kwargs={"log_likelihood": True})
    except Exception:
        idata = pm.sample(2000, nuts_sampler="numpyro", init="jitter+adapt_diag", tune=1000, chains=4, target_accept=0.9, adapt_step_size=True, random_seed=123, idata_kwargs={"log_likelihood": True})
az.to_netcdf(idata, "results/bvar_results.nc")
```

    Running window adaptation
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:00&lt;?]
</div>





<div>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:00&lt;?]
</div>





<div>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:00&lt;?]
</div>





<div>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:00&lt;?]
</div>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='5000' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [5000/5000 00:00&lt;?]
</div>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='5000' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [5000/5000 00:00&lt;?]
</div>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='5000' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [5000/5000 00:00&lt;?]
</div>





<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='5000' class='' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [5000/5000 00:00&lt;?]
</div>



    There were 6 divergences after tuning. Increase `target_accept` or reparameterize.
    




    'results/bvar_results.nc'




```python
idata = az.from_netcdf("results/bvar_results.nc")
```


```python
# Quick numeric summary
az.summary(idata, var_names=["B", "Sigma", "tau_D"], round_to=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B[0, 0]</th>
      <td>-0.05</td>
      <td>0.09</td>
      <td>-0.22</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>34508.91</td>
      <td>15595.07</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>B[0, 1]</th>
      <td>0.03</td>
      <td>0.06</td>
      <td>-0.08</td>
      <td>0.15</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>35988.23</td>
      <td>15182.45</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>B[0, 2]</th>
      <td>0.04</td>
      <td>0.15</td>
      <td>-0.25</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36379.85</td>
      <td>15787.18</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>B[0, 3]</th>
      <td>-0.13</td>
      <td>0.13</td>
      <td>-0.37</td>
      <td>0.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>37043.68</td>
      <td>15445.59</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>B[0, 4]</th>
      <td>0.01</td>
      <td>0.03</td>
      <td>-0.04</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>31890.49</td>
      <td>15737.98</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Sigma[7, 4]</th>
      <td>0.01</td>
      <td>0.02</td>
      <td>-0.03</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>19580.68</td>
      <td>15138.91</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Sigma[7, 5]</th>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20463.01</td>
      <td>15936.13</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Sigma[7, 6]</th>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20297.54</td>
      <td>15365.98</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>Sigma[7, 7]</th>
      <td>0.03</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20681.92</td>
      <td>14776.48</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>tau_D</th>
      <td>0.64</td>
      <td>0.37</td>
      <td>0.01</td>
      <td>1.23</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>1663.23</td>
      <td>890.92</td>
      <td>1.01</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 9 columns</p>
</div>




```python
coords = {"B_dim_0": [0, 1], "B_dim_1": [0, 1, 2]} # adjust to dims

az.plot_trace(
    idata,
    var_names=["tau_D", "B"],
    coords=coords,
)
```




    array([[<Axes: title={'center': 'tau_D'}>,
            <Axes: title={'center': 'tau_D'}>],
           [<Axes: title={'center': 'B'}>, <Axes: title={'center': 'B'}>]],
          dtype=object)




    
![png](output_18_1.png)
    



```python
idata.attrs["vars_used"] = vars_used
idata.attrs["lags"] = p

var_order = [
    "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    "level","slope_10y_1y","curvature_ns_like"
]
```

## 3) Structural VAR (SVAR) and sign restrictions

The reduced-form BVAR gives us correlations and dynamics, but not **economic shocks**. To tell monetary, fiscal, or financial shocks apart we need to impose identifying restrictions, i.e., map reduced-form errors $u_t$ into structural shocks $\varepsilon_t$ with a causal, economic meaning.

In a reduced-form VAR we have
$$
u_t \sim \mathcal{N}(0, \Sigma_u),
$$
but we want a representation
$$
u_t = C \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I),
$$
where the columns of $C$ describe the instantaneous effect of each structural shock on the variables. Since many different $C$ matrices can produce the same $\Sigma_u = C C'$, we need extra **identifying restrictions**.

Structural VARs (SVARs) aim to recover economically meaningful shocks, such as monetary tightening or fiscal expansion, from the reduced-form errors of a standard VAR. The key challenge is that the reduced-form residuals only give us a covariance matrix $\Sigma_u$, and infinitely many structural decompositions are consistent with it. Identification requires additional assumptions to select a unique (or at least interpretable) mapping from reduced-form shocks to structural shocks. Classical approaches imposed zero restrictions (e.g., Cholesky ordering) or long-run restrictions, but these often rely on strong and potentially arbitrary modeling choices. Over the last two decades, sign restrictions have become a widely used alternative: instead of specifying exact zero patterns, we require the impulse responses of certain variables to have economically plausible signs (e.g., interest rates rise after a monetary tightening, output falls). This produces a set of admissible structural models rather than a single one and greatly reduces the risk of misspecification. In practice, identification proceeds by generating many candidate decompositions of $\Sigma_u$ using random rotations and keeping only those whose impulse responses satisfy the chosen sign patterns. This approach originated with Uhlig (2005) and was further developed by Rubio-Ramírez, Waggoner, and Zha (2010) and it is flexible, transparent, and aligns naturally with our probabilistic programming setting.

---

### Why sign restrictions?

Classical SVAR identification schemes (Cholesky ordering, long-run restrictions) often impose **too much structure** or force an arbitrary ordering of variables. **Sign restrictions** (Uhlig, 2005; Rubio-Ramírez, Waggoner and Zha, 2010) are more flexible:

- the researcher only specifies the **direction** of the response of some variables to a shock,
- over a short horizon (e.g. 0–4 quarters),
- without fixing the exact magnitude.

Examples:

- a **monetary tightening** shock: policy rate ↑, output ↓ or non-positive, prices ↓ or non-positive;
- a **fiscal expansion** shock: government spending ↑, output ↑, maybe interest rates ↑.

This is attractive for macro-finance because we usually know the **sign** of the effect better than we know its exact size.

---

### How it works in our framework

Because we already have a posterior over $(B, \Sigma_u)$ from the BVAR, sign restrictions become a **post-processing step**: for each posterior draw we generate candidate structural decompositions and keep only the ones whose IRFs satisfy our sign pattern.

The workflow is:

1. **Draw reduced-form parameters from the posterior**  
   We already have this in the `idata` object.

2. **Generate candidate impact matrices**  
   For a given $\Sigma_u$ we can write $ \Sigma_u = P P' $ (Cholesky, or any square root). Any orthonormal rotation $Q$ gives another valid decomposition
   $$
   C = P Q,
   $$
   so by drawing random orthonormal $Q$ we generate many candidate structural models.  
   In code this is what `make_candidates_from_idata(...)` does: for each posterior draw it builds candidate $C$ matrices (and the associated IRFs) by applying random rotations.

3. **Compute IRFs for each candidate**  
   Using the VAR coefficients and each candidate $C$, we simulate impulse responses over a small horizon, say 0–12 quarters.

4. **Apply the sign restrictions**  
   We define a simple dictionary of desired signs by variable and horizon (e.g. policy_rate: positive at h=0..1, output: non-positive at h=0..4).  
   The function `apply_sign_restrictions(...)` then checks, for each candidate IRF, whether all sign conditions are satisfied. If yes, we **accept** the candidate; otherwise we discard it.

5. **Collect accepted scenarios**  
   The accepted draws are stored in a dictionary (like `accepted_macro`) and can be used just like the Basel scenarios: to build conditional forecasts, to convert to a `df_all`-like long format, and finally to feed into the NII/EVE layer.

---

### Why it’s easy to integrate here

- We already sample the **full posterior** of the VAR, so we don’t need to re-estimate anything for SVAR — we simply rotate each posterior draw.
- The sign-checking step is vectorized and can be written in a few lines of NumPy.
- The output structure (`accepted_macro`) mirrors the one we used for Basel, so later on we can **reuse** the same plotting and the same NII/EVE mapping.

In other words, once the BVAR is estimated, adding structural identification is just: *“generate candidates → filter by signs → store accepted draws.”* This is why sign-restricted SVARs are so popular in applied macro today.



```python
from src import svar
```


```python
idata = az.from_netcdf("results/bvar_results.nc")
cands = svar.make_candidates_from_idata(
    idata=idata,
    p=idata.attrs.get("lags", 2),
    H=20,
    max_draws=5000,
    rotations_per_draw=50,
    require_stable=True,
    seed=123
)

print("Candidates generated:", len(cands["C"]))
print("Shapes B, Sigma, C, IRFs:", cands["B"].shape, cands["Sigma"].shape, cands["C"].shape, cands["IRFs"].shape)
```

    Candidates generated: 214750
    Shapes B, Sigma, C, IRFs: (214750, 8, 16) (214750, 8, 8) (214750, 8, 8) (214750, 21, 8, 8)
    

These are our two sets of sign restrictions for our SVAR for each of our macroeconomic scenarios: fiscal expansion and monetary tightening

### Fiscal Expansion Sign Restrictions

| Variable   | Horizon (q) | Sign | Economic Meaning |
|------------|-------------|------|------------------|
| deficit    | 0–2         | –    | Deficit increases (deficits are negative, so a rise is more negative) |
| level      | 0           | +    | Interest-rate level rises on impact |
| inflation  | 1–4         | +    | Inflation increases within year 1 |
| GDP        | 1–4         | +    | Output rises with a lag |
| debt       | 2–6         | +    | Debt ratio rises over 0.5–2 years |
| (others)   | —           | free | No restriction on slope, curvature, policy |

### Monetary Tightening Sign Restrictions

| Variable      | Horizon (q) | Sign | Economic Meaning |
|---------------|-------------|------|------------------|
| policy rate   | 0           | +    | Policy rate jumps (tightening) |
| slope         | 0–2         | –    | Yield curve flattens |
| inflation     | 1–4         | –    | Disinflation within year 1 |
| GDP           | 1–4         | –    | Output falls with a lag |
| (others)      | —           | free | Level, deficit, debt, curvature unrestricted |


```python
spec_fiscal, spec_monet = svar.build_sign_specs_8var()

accepted_fiscal = svar.apply_sign_restrictions(
    cands, spec_one_shock=spec_fiscal, shock_name="fiscal_expansion",
    max_accept=5000, tol_zero=1e-8
)
print("Fiscal: accepted", accepted_fiscal["accepted"], "of", accepted_fiscal["tried"],
      f"(rate={accepted_fiscal['accept_rate']:.2%})")

accepted_monet = svar.apply_sign_restrictions(
    cands, spec_one_shock=spec_monet, shock_name="monetary_tightening",
    max_accept=5000, tol_zero=1e-8
)
print("Monetary: accepted", accepted_monet["accepted"], "of", accepted_monet["tried"],
      f"(rate={accepted_monet['accept_rate']:.2%})")
```

    Fiscal: accepted 5000 of 95856 (rate=5.22%)
    Monetary: accepted 5000 of 71277 (rate=7.01%)
    


```python
accepted_macro = {'fiscal_expansion' : accepted_fiscal, "monetary_tightening" : accepted_monet}

accF = accepted_macro["fiscal_expansion"]
alphas_1sd_fisc, sig_def_f = svar.alphas_one_sigma(accF, panel, var_order, "gg_deficit_pct_gdp")

# Fiscal expansion, 1σ on deficit
accM = accepted_macro["monetary_tightening"]
alphas_1sd_monet, sig_pol_m = svar.alphas_one_sigma(accM, panel, var_order, "policy_rate")


f_med, f_lo, f_hi, df_irf_fiscal  = svar.summarize_irfs(accepted_fiscal, var_order, H=20, alpha_scaling=alphas_1sd_fisc)
m_med, m_lo, m_hi, df_irf_monet   = svar.summarize_irfs(accepted_monet, var_order, H=20, alpha_scaling=alphas_1sd_monet)
```

Impulse Response Functions (IRFs) show how each variable in a VAR responds over time to a one-unit structural shock, such as a monetary tightening or a fiscal expansion. An IRF traces the dynamic path of the system: it answers “if this shock hits today, how will interest rates, GDP, inflation, and other variables evolve over the next quarters?” Because VARs capture the interactions and propagation mechanisms among variables, IRFs provide a transparent way to study causal effects in macroeconomics. They are the core tool used to interpret structural shocks and to evaluate policy transmission.


```python
svar.plot_irf_panel(f_med, f_lo, f_hi, var_order, shock_label="Fiscal expansion")
plt.show()
```


    
![png](output_27_0.png)
    



```python
svar.plot_irf_panel(m_med, m_lo, m_hi, var_order, shock_label="Monetary tightening")
plt.show()
```


    
![png](output_28_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_fiscal, "fiscal_expansion",
                    var="infl_q_ann", var_idx=0, H=20)
plt.show()
```


    
![png](output_29_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_fiscal, "fiscal_expansion",
                    var="gdp_q_ann", var_idx=0, H=20)
plt.show()
```


    
![png](output_30_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_fiscal, "fiscal_expansion",
                    var="level", var_idx=0, H=20)
plt.show()
```


    
![png](output_31_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_monet, "monetary_tightening",
                    var="infl_q_ann", var_idx=0, H=20)
plt.show()
```


    
![png](output_32_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_monet, "monetary_tightening",
                    var="gdp_q_ann", var_idx=0, H=20)
plt.show()
```


    
![png](output_33_0.png)
    



```python
svar.plot_irf_comparison(cands["IRFs"][:, :, :, 3], df_irf_monet, "monetary_tightening",
                    var="slope_10y_1y", var_idx=0, H=20)
plt.show()
```


    
![png](output_34_0.png)
    



```python
svar.plot_hist_impact(accepted_fiscal, 0, "Fiscal Expansion", "Inflation")
```


    
![png](output_35_0.png)
    



```python
svar.plot_hist_impact(accepted_fiscal, 5, "Fiscal Expansion", "Level")
```


    
![png](output_36_0.png)
    



```python
svar.plot_hist_impact(accepted_monet, 0, "Monetary Tightening", "Inflation")
```


    
![png](output_37_0.png)
    



```python
svar.plot_hist_impact(accepted_monet, 6, "Monetary Tightening", "Slope")
```


    
![png](output_38_0.png)
    



```python
rep_fiscal  = svar.quick_diag_summary(accepted_fiscal,  var_order, "Fiscal expansion")
rep_monet   = svar.quick_diag_summary(accepted_monet,   var_order, "Monetary tightening")
```

    
    === Diagnostics summary: Fiscal expansion ===
    Accepted: 5000 / 95856 (5.22% rate)
    Median impact responses (h=0):
      infl_q_ann          : {'median': np.float64(0.157), 'p10': np.float64(-0.361), 'p90': np.float64(0.877)}
      gdp_q_ann           : {'median': np.float64(-0.058), 'p10': np.float64(-0.941), 'p90': np.float64(0.549)}
      policy_rate         : {'median': np.float64(0.034), 'p10': np.float64(-0.083), 'p90': np.float64(0.147)}
      gg_deficit_pct_gdp  : {'median': np.float64(-0.185), 'p10': np.float64(-0.278), 'p90': np.float64(-0.082)}
      gg_debt_pct_gdp     : {'median': np.float64(0.366), 'p10': np.float64(0.085), 'p90': np.float64(0.618)}
      level               : {'median': np.float64(0.067), 'p10': np.float64(0.013), 'p90': np.float64(0.147)}
      slope_10y_1y        : {'median': np.float64(-0.031), 'p10': np.float64(-0.145), 'p90': np.float64(0.089)}
      curvature_ns_like   : {'median': np.float64(0.028), 'p10': np.float64(-0.051), 'p90': np.float64(0.1)}
    
    === Diagnostics summary: Monetary tightening ===
    Accepted: 5000 / 71277 (7.01% rate)
    Median impact responses (h=0):
      infl_q_ann          : {'median': np.float64(-0.24), 'p10': np.float64(-1.077), 'p90': np.float64(0.328)}
      gdp_q_ann           : {'median': np.float64(-0.19), 'p10': np.float64(-1.015), 'p90': np.float64(0.628)}
      policy_rate         : {'median': np.float64(0.102), 'p10': np.float64(0.023), 'p90': np.float64(0.19)}
      gg_deficit_pct_gdp  : {'median': np.float64(0.089), 'p10': np.float64(-0.069), 'p90': np.float64(0.214)}
      gg_debt_pct_gdp     : {'median': np.float64(-0.16), 'p10': np.float64(-0.506), 'p90': np.float64(0.221)}
      level               : {'median': np.float64(0.007), 'p10': np.float64(-0.103), 'p90': np.float64(0.118)}
      slope_10y_1y        : {'median': np.float64(-0.081), 'p10': np.float64(-0.165), 'p90': np.float64(-0.024)}
      curvature_ns_like   : {'median': np.float64(-0.078), 'p10': np.float64(-0.135), 'p90': np.float64(-0.009)}
    


```python
table_fiscal  = svar.diag_table(accepted_fiscal,  var_order, "Fiscal expansion")
table_monet   = svar.diag_table(accepted_monet,   var_order, "Monetary tightening")

# Combine into one table for display
diag_tables = pd.concat([table_fiscal, table_monet], ignore_index=True)
diag_tables.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shock</th>
      <th>var</th>
      <th>median</th>
      <th>p10</th>
      <th>p90</th>
      <th>accepted</th>
      <th>tried</th>
      <th>accept_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fiscal expansion</td>
      <td>infl_q_ann</td>
      <td>0.16</td>
      <td>-0.36</td>
      <td>0.88</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fiscal expansion</td>
      <td>gdp_q_ann</td>
      <td>-0.06</td>
      <td>-0.94</td>
      <td>0.55</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fiscal expansion</td>
      <td>policy_rate</td>
      <td>0.03</td>
      <td>-0.08</td>
      <td>0.15</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fiscal expansion</td>
      <td>gg_deficit_pct_gdp</td>
      <td>-0.18</td>
      <td>-0.28</td>
      <td>-0.08</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fiscal expansion</td>
      <td>gg_debt_pct_gdp</td>
      <td>0.37</td>
      <td>0.08</td>
      <td>0.62</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Fiscal expansion</td>
      <td>level</td>
      <td>0.07</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fiscal expansion</td>
      <td>slope_10y_1y</td>
      <td>-0.03</td>
      <td>-0.14</td>
      <td>0.09</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fiscal expansion</td>
      <td>curvature_ns_like</td>
      <td>0.03</td>
      <td>-0.05</td>
      <td>0.10</td>
      <td>5000</td>
      <td>95856</td>
      <td>5.22%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Monetary tightening</td>
      <td>infl_q_ann</td>
      <td>-0.24</td>
      <td>-1.08</td>
      <td>0.33</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Monetary tightening</td>
      <td>gdp_q_ann</td>
      <td>-0.19</td>
      <td>-1.01</td>
      <td>0.63</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Monetary tightening</td>
      <td>policy_rate</td>
      <td>0.10</td>
      <td>0.02</td>
      <td>0.19</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Monetary tightening</td>
      <td>gg_deficit_pct_gdp</td>
      <td>0.09</td>
      <td>-0.07</td>
      <td>0.21</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Monetary tightening</td>
      <td>gg_debt_pct_gdp</td>
      <td>-0.16</td>
      <td>-0.51</td>
      <td>0.22</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Monetary tightening</td>
      <td>level</td>
      <td>0.01</td>
      <td>-0.10</td>
      <td>0.12</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Monetary tightening</td>
      <td>slope_10y_1y</td>
      <td>-0.08</td>
      <td>-0.16</td>
      <td>-0.02</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Monetary tightening</td>
      <td>curvature_ns_like</td>
      <td>-0.08</td>
      <td>-0.14</td>
      <td>-0.01</td>
      <td>5000</td>
      <td>71277</td>
      <td>7.01%</td>
    </tr>
  </tbody>
</table>
</div>




```python
impact_M, shocks, vars_ = svar.build_impact_matrix(
    {"Fiscal expansion": table_fiscal, "Monetary tightening": table_monet},
    var_order
)

svar.plot_impact_heatmap(impact_M, shocks, vars_, title="Median (h=0) responses by shock")
```


    
![png](output_41_0.png)
    


## 4) Basel III IRRBB scenarios

In this part of the notebook, we translate the Basel Committee’s *Interest Rate Risk in the Banking Book (IRRBB)* framework into model-consistent yield-curve scenarios.  
The goal is to simulate how the bank’s balance sheet and income would react under the standardized rate shocks used by supervisors.

---

### Basel III IRRBB background

The IRRBB standard defines six supervisory rate scenarios that banks must evaluate for both **Net Interest Income (NII)** and **Economic Value of Equity (EVE)**:

| Scenario | Shape of the shift | Description |
|-----------|-------------------|--------------|
| **Parallel up** | Entire curve +200 bps | Uniform upward move of all maturities |
| **Parallel down** | Entire curve –200 bps | Uniform downward move |
| **Steepener** | Long rates rise more than short rates | Curve steepens |
| **Flattener** | Short rates rise more than long rates | Curve flattens |
| **Short-end up** | Only short maturities rise | Front-end stress |
| **Short-end down** | Only short maturities fall | Front-end easing |

These scenarios are prescribed in the Basel and ECB IRRBB guidelines and serve as the benchmark set for supervisory stress testing across European institutions.

The rationale is simple: by applying stylized but extreme yield-curve shocks, regulators can gauge both the **earnings sensitivity** (NII) and the **valuation sensitivity** (EVE) of a bank’s balance sheet.

---

### Implementation details

The main steps are:

1. **Define the target paths**  
   Functions like `make_ecb_basel6()` generate six predefined level–slope paths corresponding to the Basel shapes (in basis points).

2. **Condition on those paths**  
   The function `basel_conditional_curve_path()` uses the BVAR posterior to draw future trajectories consistent with the imposed yield-path constraints.  
   It solves the linear constraints via Kalman-like updating for each posterior sample.

3. **Store and visualize**  
   The resulting conditional forecasts (median and credible intervals) are saved in long format (`df_all`) and visualized with fan charts to verify the imposed shock shapes.

This approach makes it straightforward to overlay regulatory and structural macro shocks within the same probabilistic forecasting framework.

---

### Intuition

Conceptually, we are telling the model:  
> “Suppose the next few quarters follow the same yield-curve evolution as the Basel flattener (or parallel-up, etc.).  
> What does the model predict for GDP, inflation, and other variables?”

Because these are **conditional expectations** rather than deterministic stress shocks, they preserve the statistical relationships captured in the VAR while imposing the desired regulatory yield-curve shape.

---

> **Code below:**  
> - Define the six Basel yield-curve scenarios.  
> - Generate conditional forecasts for each scenario using `basel_conditional_curve_path()` and `make_ecb_basel6()`.  
> - Combine and visualize results to confirm that level and slope paths match the Basel templates.


```python
from src import basel_scenarios
```


```python
# Load posterior and panel (same as your existing code)
idata = az.from_netcdf("results/bvar_results.nc")  # from bvar.py
panel = pd.read_csv("data/quarterly_panel_modelvars.csv", parse_dates=["date"], index_col="date")

var_order = [
    "infl_q_ann","gdp_q_ann","policy_rate","gg_deficit_pct_gdp","gg_debt_pct_gdp",
    "level","slope_10y_1y","curvature_ns_like"
]
```

### What `basel_conditional_curve_path` Does

`basel_conditional_curve_path` generates model-consistent conditional forecasts for the VAR when you impose future paths for selected variables, such as the Basel III IRRBB shocks on the level, slope, or policy rate. The function takes the posterior draws from the reduced-form BVAR (idata), the historical data up to the last observed quarter, and a set of user-defined path constraints expressed in basis points (e.g. “level +200 bps for the next 4 quarters”, or “short end +50 bps at $h=1$, flat afterward”). It then converts these restrictions into a system of linear equality constraints on the stacked future values of the VAR state.

For each posterior draw of the VAR coefficients and covariance matrix, the function builds the unconditional forecast distribution for the next 
$H$ quarters. This is a multivariate Gaussian characterized by a mean vector and covariance matrix derived from the companion-form VAR. Using the standard Gaussian conditioning formula, the function adjusts this forecast so that the constrained variables exactly match the user-specified paths, while all other variables adjust optimally according to the VAR’s joint dynamics. This produces the conditional mean and conditional covariance of future outcomes under the imposed scenario.

Finally, for each posterior draw, the function simulates several conditional posterior predictive paths, samples from the resulting distribution, and returns all draws as a single long DataFrame containing:

- forecast horizon $h=1,…,H$

- variable name,

- conditional draws,

- posterior draw index,

- scenario label.

The resulting object can be used directly to generate fan charts, scenario comparisons, or as input to downstream modules such as NII/EVE estimation. `make_ecb_basel6` is just a wrapper for all 6 scenarios.


```python
# Common args
kwargs_common = dict(
    idata=idata,
    Yhist_df=panel,           # your historical DataFrame
    var_order=var_order,      # ["infl_q_ann", ..., "level", "slope_10y_1y", ...]
    p=idata.attrs.get("lags", 2),
    H=12,
    draws_posterior=400,
    draws_conditional=200,
)

# 1) Parallel UP: level +200 bps (keep everything else free)
df_parallel_up = basel_scenarios.basel_conditional_curve_path(
    **kwargs_common,
    constraints={
        "level": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": +200}
    },
    scenario_name="Basel_parallel_up"
)

# # 2) Parallel DOWN
# df_parallel_down = basel_scenarios.basel_conditional_curve_path(
#     **kwargs_common,
#     constraints={
#         "level": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": -200}
#     },
#     scenario_name="Basel_parallel_down"
# )

# # 3) Steepener: slope +X bps (optionally keep level unchanged by pinning +0 bps)
# df_steepener = basel_scenarios.basel_conditional_curve_path(
#     **kwargs_common,
#     constraints={
#         "slope_10y_1y": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": +100},
#         # Optional: hold level flat (0 bps) if you want a pure slope move
#         "level": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": 0},
#     },
#     scenario_name="Basel_steepener"
# )

# # 4) Flattener: slope -X bps (optionally level 0)
# df_flattener = basel_scenarios.basel_conditional_curve_path(
#     **kwargs_common,
#     constraints={
#         "slope_10y_1y": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": -100},
#         "level": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": 0},
#     },
#     scenario_name="Basel_flattener"
# )

# # 5) Short-end UP: policy rate +Y bps
# df_short_up = basel_scenarios.basel_conditional_curve_path(
#     **kwargs_common,
#     constraints={
#         "policy_rate": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": +200}
#     },
#     scenario_name="Basel_short_end_up"
# )

# # 6) Short-end DOWN
# df_short_down = basel_scenarios.basel_conditional_curve_path(
#     **kwargs_common,
#     constraints={
#         "policy_rate": {"unit": "pct", "horizons": [1,2,3,4], "delta_bps": -200}
#     },
#     scenario_name="Basel_short_end_down"
# )

```


```python
# Example: median conditional path for 'level' and 'policy_rate'
out = (df_parallel_up.groupby(["var","h"])["yhat"]
             .median()
             .rename("median")
             .reset_index())
```


```python
out[out['var'] == "level"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>h</th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>60</th>
      <td>level</td>
      <td>1</td>
      <td>4.193557</td>
    </tr>
    <tr>
      <th>61</th>
      <td>level</td>
      <td>2</td>
      <td>4.193557</td>
    </tr>
    <tr>
      <th>62</th>
      <td>level</td>
      <td>3</td>
      <td>4.193557</td>
    </tr>
    <tr>
      <th>63</th>
      <td>level</td>
      <td>4</td>
      <td>4.193557</td>
    </tr>
    <tr>
      <th>64</th>
      <td>level</td>
      <td>5</td>
      <td>4.128501</td>
    </tr>
    <tr>
      <th>65</th>
      <td>level</td>
      <td>6</td>
      <td>4.065993</td>
    </tr>
    <tr>
      <th>66</th>
      <td>level</td>
      <td>7</td>
      <td>4.002690</td>
    </tr>
    <tr>
      <th>67</th>
      <td>level</td>
      <td>8</td>
      <td>3.940145</td>
    </tr>
    <tr>
      <th>68</th>
      <td>level</td>
      <td>9</td>
      <td>3.881297</td>
    </tr>
    <tr>
      <th>69</th>
      <td>level</td>
      <td>10</td>
      <td>3.821219</td>
    </tr>
    <tr>
      <th>70</th>
      <td>level</td>
      <td>11</td>
      <td>3.767708</td>
    </tr>
    <tr>
      <th>71</th>
      <td>level</td>
      <td>12</td>
      <td>3.714739</td>
    </tr>
  </tbody>
</table>
</div>




```python
basel_scenarios.plot_basel_conditional_forecasts(out)
```


    
![png](output_49_0.png)
    



    
![png](output_49_1.png)
    



    
![png](output_49_2.png)
    



    
![png](output_49_3.png)
    



    
![png](output_49_4.png)
    



    
![png](output_49_5.png)
    



    
![png](output_49_6.png)
    



    
![png](output_49_7.png)
    



```python
scenarios = basel_scenarios.make_ecb_basel6(
    idata=idata,
    Yhist_df=panel,
    var_order=var_order,
    p=idata.attrs.get("lags", 2),
    H=12,
    draws_posterior=200,
    draws_conditional=100,
    seed_base=123,

    # magnitudes in basis points
    parallel_bps=200.0,
    slope_bps=100.0,
    short_bps=200.0,

    horizons=[1,2,3,4],        # lock in the first year
    level_unit="pct",          # 'level' stored as e.g. 2.35 = 2.35%
    slope_unit="pct",          # slope factor also stored in percent
    short_unit="pct",          # policy rate also in percent
    level_var="level",
    slope_var="slope_10y_1y",
    short_var="policy_rate",
)

df_all = scenarios["all"]
```


```python
basel_scenarios.plot_curve_termsheet(df_all)
```


    
![png](output_51_0.png)
    



    
![png](output_51_1.png)
    



    
![png](output_51_2.png)
    



    
![png](output_51_3.png)
    



    
![png](output_51_4.png)
    



    
![png](output_51_5.png)
    


## 5) Conditional forecasts

Once the BVAR posterior has been estimated and the Basel or macroeconomic scenarios are defined, we can generate **conditional forecasts**.  
These are forecasts that impose specific values or paths for a subset of variables (for example, yield-curve factors), while allowing all other variables to adjust in a way consistent with the estimated model dynamics.

---

### Concept

A conditional forecast answers questions of the form:

> *“Given that the yield curve follows a certain path (e.g., a +200 bps parallel shift), what are the implied model-consistent trajectories for inflation, output, and other variables?”*

Formally, we are computing the conditional distribution:

$$
p(Y_{t+h} \mid Y_t, \text{constraints on some variables})
$$

for $h = 1, \ldots, H$.  
This distribution reflects both the posterior uncertainty in the VAR parameters and the stochastic uncertainty in future shocks, while respecting the imposed conditions.

In this sense, the conditional forecast is a **probabilistic projection under scenario constraints**, not a single deterministic path.

---

### Implementation in our framework

Our BVAR posterior already gives us $B$, $\Sigma \), and all draws of the reduced-form parameters.  
To generate conditional forecasts, we:

1. **Specify the conditioning paths**  
   Each scenario (Basel or macro) defines a set of target values for variables like `level`, `slope_10y_1y`, and `policy_rate` over a given horizon.

2. **Apply linear constraints**  
   Using the algorithm of Waggoner & Zha (1999), we adjust the forecast mean and covariance so that the simulated paths satisfy the imposed constraints while maintaining the original model structure.  
   In practice, this is implemented inside functions such as `make_conditional_forecast()` and `basel_conditional_curve_path()`.

3. **Simulate posterior draws**  
   For each posterior draw of the BVAR, we generate future trajectories consistent with the constraints, creating a full conditional predictive distribution.

4. **Aggregate results**  
   The posterior median and credible intervals across all draws are summarized in long-format DataFrames (e.g., `df_all`) for visualization.

---

### Visualization: fan charts

To interpret these forecasts, we use **fan charts**, which show the central projection (posterior median) and uncertainty bands (e.g., 90% credible intervals).  
Fan charts are a standard visualization in macroeconomic forecasting, making it easy to compare scenarios and gauge the dispersion of possible outcomes.

> **Code below:**  
> - Generate conditional forecasts for each scenario.  
> - Convert results into long-format DataFrames with `make_fan_df_from_paths()`.  
> - Plot fan charts for the variables (e.g. deficit, GDP, inflation) to visualize how the imposed shocks propagate through the system.


```python
from src import conditional_forecasts
```

## What `make_conditional_forecast` Does

`make_conditional_forecast` produces **SVAR-style conditional forecasts** for a *single* identified scenario (e.g. a monetary tightening or fiscal expansion shock). It takes the accepted structural draws from the sign-restricted SVAR (stored in `accepted_all`), reconstructs the VAR dynamics for each draw, applies a one-time structural shock of user-specified size at the beginning of the forecast horizon, and simulates how all variables evolve over the next $h$ quarters. The function therefore uses *structural* shocks, and not reduced-form innovations, to generate forward paths.

The historical panel provides the last $p$ lags needed to initialize the VAR state. For each accepted structural draw, the function extracts the corresponding VAR coefficient matrices $A_1,\dots,A_p$, the intercept vector $c$, and the **impact matrix** (either from the stored IRFs or directly from the $C$ matrix). It then generates a deterministic forecast path for that draw by iterating the VAR forward and injecting the structural shock at $t=0$ (or for multiple periods if `shock_horizon` > 1). If `baseline=True`, the same procedure is run *without* the shock to create a counterfactual.

After simulating up to `n_paths` accepted draws, the function aggregates the resulting trajectories into median forecasts and 10–90% credible bands for each variable and each horizon. It returns both the full array of simulated paths and a tidy DataFrame suitable for **fan charts**, **scenario comparisons**, and **report-quality macro projections**.

In short, `make_conditional_forecast` turns an identified SVAR shock into a coherent, forward-looking scenario for all variables, using only the structural dynamics implied by the sign-restricted BVAR posterior.


```python
# previously defined:
# accepted_macro = {"monetary_tightening": ..., "fiscal_expansion": ...}
# panel, var_order

fan_monet, paths_monet = conditional_forecasts.make_conditional_forecast(
    accepted_all=accepted_macro,
    panel=panel,
    var_order=var_order,
    scenario="monetary_tightening",
    h=12,
    shock_size=1.0,
    shock_horizon=1,
    n_paths=200,
)

# and baseline from the same origin
fan_monet_0, _ = conditional_forecasts.make_conditional_forecast(
    accepted_all=accepted_macro,
    panel=panel,
    var_order=var_order,
    scenario="monetary_tightening",
    h=12,
    baseline=True,   # <-- no shock
)

# plot only inflation and policy rate
conditional_forecasts.plot_conditional_forecast(fan_monet, vars_to_plot=["infl_q_ann","policy_rate","gdp_q_ann"])

```


    
![png](output_55_0.png)
    



    
![png](output_55_1.png)
    



    
![png](output_55_2.png)
    



```python
fan_fiscal, paths_fiscal = conditional_forecasts.make_conditional_forecast(
    accepted_all=accepted_macro,
    panel=panel,
    var_order=var_order,
    scenario="fiscal_expansion",
    h=12,
    shock_size=1.0,
    shock_horizon=1,
    n_paths=200,
)

# and baseline from the same origin
fan_fiscal_0, _ = conditional_forecasts.make_conditional_forecast(
    accepted_all=accepted_macro,
    panel=panel,
    var_order=var_order,
    scenario="fiscal_expansion",
    h=12,
    baseline=True,   # <-- no shock
)

# plot only inflation and policy rate
conditional_forecasts.plot_conditional_forecast(fan_fiscal, vars_to_plot=["gg_deficit_pct_gdp","infl_q_ann","gdp_q_ann"])
```


    
![png](output_56_0.png)
    



    
![png](output_56_1.png)
    



    
![png](output_56_2.png)
    


## 6) Net Interest Income (NII) and Economic Value of Equity (EVE)

Having generated conditional yield-curve paths under both Basel and macro scenarios, we now translate them into **balance-sheet metrics** that measure the bank’s exposure to interest-rate risk in the banking book (IRRBB).

---

### Purpose of NII and EVE

These two measures capture complementary perspectives on a bank’s sensitivity to rate movements:

| Metric | Type | Description | Typical Horizon |
|---------|------|--------------|------------------|
| **NII – Net Interest Income** | Flow | Change in interest income minus expense over a fixed horizon (usually 1 year). Measures short-term earnings impact. | 1 year |
| **EVE – Economic Value of Equity** | Stock | Change in the present value of all future cash flows from assets and liabilities. Measures long-term valuation impact. | Instantaneous |

Regulators use both because a bank might be protected in one dimension (e.g., stable income) but vulnerable in the other (e.g., large valuation losses).

---

### Simplified mapping from yield paths to IRRBB metrics

We adopt a stylized but calibrated approach that keeps the magnitudes realistic and the code transparent.

- **Baseline parameters**  
  - Annual baseline NII = €10 billion  
  - CET1 capital = €40 billion  
  - Duration of equity = 6 years  

- **Calibration targets (ECB IRRBB medians)**  
  - +200 bps parallel up → ΔNII ≈ +6 %, ΔEVE ≈ –12 % of CET1  
  - –200 bps parallel down → ΔNII ≈ –6 %, ΔEVE ≈ +12 %

These values are typical for large euro-area banks and provide a realistic scaling anchor for our toy balance-sheet model.

---

### Algorithmic steps

For each scenario (Basel or macro):

1. **Extract the conditional yield-curve path**  
   We take the projected values of the `level`, `slope_10y_1y`, and `policy_rate` factors for horizons $h = 1,\dots,H$.

2. **Compute yield changes**  
   The first-period change in the level factor (relative to baseline) represents the effective parallel shift in basis points.

3. **Translate to NII response**  
   NII is modeled as proportional to a weighted combination of the short and long rate movements:
   $$
   \Delta \text{NII}_t = \alpha_s \Delta r_{\text{short},t} + \alpha_l \Delta r_{\text{long},t},
   $$
   with positive sensitivity (asset-sensitive bank).  
   The weights are scaled so that a +200 bps parallel up shock yields +6 % NII.

4. **Translate to EVE response**  
   EVE is computed as the present-value effect of a parallel shift:
   $$
   \Delta \text{EVE} = -D_{\text{equity}} \times \Delta r_{\text{level}},
   $$
   where $D_{\text{equity}}$ is the duration of equity (≈ 6 years).  
   Because this simplified formulation depends only on the **average (level)** of the curve, steepener and flattener scenarios produce the same EVE change when their average move is equal.

5. **Aggregate results**  
   The outputs for each scenario are summarized as:
   - **ΔNII (1y, %)** relative to baseline NII,  
   - **ΔEVE (% CET1)** relative to capital.

---

### Interpretation

The results follow the expected supervisory patterns:

- **Parallel up:** NII ↑ (assets reprice faster), EVE ↓ (lower PV of fixed assets)  
- **Parallel down:** NII ↓, EVE ↑  
- **Steepener / Flattener:** smaller magnitudes, signs consistent with curve shape  
- **Short-end shocks:** moderate effects depending on short-term funding exposure  
- **Monetary tightening:** modest NII fall due to short-end increase and margin compression  

If all six Basel scenarios produce NII ↑ and EVE ↓, that simply reflects the assumption of an **asset-sensitive bank** i.e. one whose fixed-rate assets dominate.  
A liability-sensitive configuration (e.g., short-term funded bank) could be implemented later by flipping the sign of short-rate sensitivity.

---

### Visualization and reporting

To communicate results, we report both metrics on a **common percentage scale**, ensuring both NII and EVE bars depart from the same zero line.  
This matches the Basel disclosure format and immediately shows the earnings–valuation trade-off.

> **Code below:**  
> - Compute ΔNII and ΔEVE for all scenarios using `compute_nii_eve_from_curvepaths()` or `scenario_to_nii_eve_objects()`.  
> - Combine results into a summary DataFrame and plot a unified bar chart using `plot_irrbb_all_scenarios()`.

---

### Extensions

While this prototype focuses on level and slope factors, the framework can easily incorporate more detail:

- Separate repricing buckets for assets and liabilities,  
- Behavioral deposit models with non-linear rate pass-through,  
- Tenor-specific EVE weights to distinguish flattener vs. steepener scenarios,  
- Integration of risk-adjusted capital and liquidity metrics.

Such extensions would yield richer NII/EVE dynamics while keeping the same Bayesian forecasting core.



```python
from src import irrbb
```


```python
import importlib
importlib.reload(irrbb)
```




    <module 'src.irrbb' from 'C:\\Users\\thoma\\Desktop\\irrbb_svar_with_code\\src\\irrbb.py'>




```python
# df_all = scenarios["all"] from your Basel conditional BVAR
baseline_level = panel["level"].iloc[-1]   # e.g. 2.19
nii_df, eve_df, fan = irrbb.run_irrbb_simple(
    df_all,
    scenario="Basel_parallel_up",
    baseline_level=baseline_level
)

irrbb.plot_nii_simple(nii_df)
```


    
![png](output_60_0.png)
    



```python
curve_paths = irrbb.extract_conditional_yieldpaths(df_all)
```


```python
yield_sensitivities = {
    "level": 1.0,          # per-bps effect on net interest margin
    "slope_10y_1y": 0.3,
    "policy_rate": 0.7,
    "nii_multiplier": 1.2, # scale to million € or %
    "eve_duration": -25.0  # PVbp equivalent
}
```


```python
nii_eve = irrbb.compute_nii_eve_from_curvepaths(curve_paths, yield_sensitivities)
```


```python
irrbb.build_plots_nii_eve(nii_eve)
```


    
![png](output_64_0.png)
    


    Basel_flattener: ΔEVE @ h=1 = -54.84
    


    
![png](output_64_2.png)
    


    Basel_parallel_down: ΔEVE @ h=1 = -4.84
    


    
![png](output_64_4.png)
    


    Basel_parallel_up: ΔEVE @ h=1 = -104.84
    


    
![png](output_64_6.png)
    


    Basel_short_end_down: ΔEVE @ h=1 = -30.54
    


    
![png](output_64_8.png)
    


    Basel_short_end_up: ΔEVE @ h=1 = -77.11
    


    
![png](output_64_10.png)
    


    Basel_steepener: ΔEVE @ h=1 = -54.84
    


```python
# 1. Load model + data  (you already do this)
idata = az.from_netcdf("results/bvar_results.nc")
panel = pd.read_csv("data/quarterly_panel_modelvars.csv",
                    parse_dates=["date"], index_col="date")

var_order = [
    "infl_q_ann",
    "gdp_q_ann",
    "policy_rate",
    "gg_deficit_pct_gdp",
    "gg_debt_pct_gdp",
    "level",
    "slope_10y_1y",
    "curvature_ns_like",
]

# 2. Build all Basel scenarios with the conditional BVAR
scenarios = basel_scenarios.make_ecb_basel6(
    idata=idata,
    Yhist_df=panel,
    var_order=var_order,
    p=idata.attrs.get("lags", 2),
    H=12,
    draws_posterior=200,
    draws_conditional=100,
    seed_base=123,
    parallel_bps=200.0,
    slope_bps=100.0,
    short_bps=200.0,
    horizons=[1,2,3,4],
    level_unit="pct",
    slope_unit="pct",
    short_unit="pct",
    level_var="level",
    slope_var="slope_10y_1y",
    short_var="policy_rate",
)

df_all = scenarios["all"]
```


```python
# 3. Load or define your term-structure mapping M and buckets
sat = irrbb.toy_sat_params()  # balance sheet setup
tenors_years = sat["buckets_years"]  # tenor grid

# M: shape (n_tenors, 3), mapping [level, slope, curvature] -> Δy by tenor
tenor_cols   = ["yc_spot_1y","yc_spot_5y","yc_spot_10y"]
M = irrbb.fit_factor_to_tenor_map(panel, tenor_cols)
current_curve_levels = panel.iloc[-1,:][tenor_cols].values

cet1_amount = 40 * 1e3  # in millions of €'s

# 4. Run one scenario through the full IRRBB pipeline
out_parallel_up = irrbb.scenario_to_nii_eve_objects(
    df_all=df_all,
    scenario_name="Basel_parallel_up",
    var_order=var_order,
    M=M,
    tenors_years=tenors_years,
    current_curve_levels=current_curve_levels,
    horizon_H=12,
    sat=sat,
    cet1_amount=cet1_amount,
    irrbb_module=irrbb,
    baseline_name=None,       # or "baseline_unshocked" if you generate one
    snap_h_eve=1,
    nii_axis="currency",
    assets_total=None
)

print(out_parallel_up["eve_table"].round(2))
```


    
![png](output_66_0.png)
    


              metric       p10    median       p90
    0     ΔEVE (abs) -86308.08 -76392.19 -66253.98
    1  ΔEVE (% CET1)       NaN   -190.97       NaN
    


```python
accepted_macro = irrbb.make_forecasts_from_irfs(accepted_macro)
last_date = panel.index[-1]

df_macro_all = irrbb.accepted_to_df_all(
    accepted_dict=accepted_macro,
    var_order=var_order,
    last_date=panel.index[-1],
    freq="QE"
)

```


```python
out_monet = irrbb.scenario_to_nii_eve_objects(
    df_all=df_macro_all,
    scenario_name="monetary_tightening",
    var_order=var_order,
    M=M,
    tenors_years=tenors_years,
    current_curve_levels=current_curve_levels,
    horizon_H=12,
    sat=sat,
    cet1_amount=cet1_amount,
    irrbb_module=irrbb,
    baseline_name=None,
    snap_h_eve=1,
)
```


    
![png](output_68_0.png)
    



```python
out_fiscal = irrbb.scenario_to_nii_eve_objects(
    df_all=df_macro_all,
    scenario_name="fiscal_expansion",
    var_order=var_order,
    M=M,
    tenors_years=tenors_years,
    current_curve_levels=current_curve_levels,
    horizon_H=12,
    sat=sat,
    cet1_amount=cet1_amount,
    irrbb_module=irrbb,
    baseline_name=None,
    snap_h_eve=1,
)
```


    
![png](output_69_0.png)
    


Note: the macroeconomic scenarios result in much smaller ΔNII because they are unit shock scenarios instead of pre-specified magnitures e.g. 200 bps from the Basel III scenarios


```python
# nii_eve is your DataFrame from compute_nii_eve_from_curvepaths
# out_monet and out_fiscal are the dicts from scenario_to_nii_eve_objects

irrbb_df = irrbb.build_irrbb_df_from_objects(
    nii_eve_df=nii_eve,
    out_monet=out_monet,
    out_fiscal=out_fiscal
)

# scale the macro shocks figures
# so now the macroeconomic scenarios (originally unit shocks) are scaled to the Basel III ones

irrbb_df.iloc[6:8,1:3] = irrbb_df.iloc[6:8,1:3] / 100

irrbb.plot_irrbb_all_scenarios(
    irrbb_df,
    baseline_nii=(10 * 1e3),
    cet1=cet1_amount
)
```


    
![png](output_71_0.png)
    



```python
irrbb.plot_eve_tornado(irrbb_df[["scenario","ΔEVE"]], cet1_eur=cet1_amount)
```


    
![png](output_72_0.png)
    

