# HMM Regime Detection — Findings (Milestone 4)

_Gaussian HMM on ['log_return', 'vol_21'], **4 states**, full covariance. Fitted on the first 70% of each series; regimes generated over the full history. Causal forward-filtered posteriors are the leakage-safe model input; smoothed Viterbi states are used for interpretation._

## SP500

BIC-optimal n_states on train = **6** (primary model uses 4 for interpretability/comparability).

**Model selection (train):**

|   n_states |   log_likelihood |     AIC |     BIC |
|-----------:|-----------------:|--------:|--------:|
|          2 |         -12512.8 | 25051.5 | 25139.4 |
|          3 |          -9643.4 | 19332.7 | 19488.2 |
|          4 |          -8156.8 | 16383.6 | 16620.3 |
|          5 |          -6624.6 | 13347.2 | 13678.6 |
|          6 |          -5823.4 | 11776.9 | 12216.4 |

**Regime statistics (full series):**

|   regime |   freq |   ann_return |   ann_vol |   avg_duration_days | label         |
|---------:|-------:|-------------:|----------:|--------------------:|:--------------|
|        0 |  0.18  |       -0.053 |     0.327 |              48.265 | Bear-volatile |
|        1 |  0.338 |        0.088 |     0.162 |              28.027 | Bull-volatile |
|        2 |  0.274 |        0.102 |     0.11  |              18.924 | Bull-calm     |
|        3 |  0.208 |        0.181 |     0.076 |              30.645 | Bull-calm     |

**Transition matrix** (rows = from-state, ordered by mean return):

|    |    S0 |    S1 |    S2 |    S3 |
|:---|------:|------:|------:|------:|
| S0 | 0.982 | 0.018 | 0     | 0     |
| S1 | 0.01  | 0.968 | 0.023 | 0     |
| S2 | 0     | 0.029 | 0.946 | 0.025 |
| S3 | 0     | 0.001 | 0.035 | 0.964 |

Implied expected regime duration (days), 1/(1−p_ii): S0=54.7, S1=30.9, S2=18.6, S3=28.0

## GOLD

BIC-optimal n_states on train = **6** (primary model uses 4 for interpretability/comparability).

**Model selection (train):**

|   n_states |   log_likelihood |     AIC |     BIC |
|-----------:|-----------------:|--------:|--------:|
|          2 |         -14086.3 | 28198.7 | 28286.8 |
|          3 |         -11630.5 | 23307   | 23462.9 |
|          4 |          -9702.7 | 19475.4 | 19712.6 |
|          5 |          -8853.3 | 17804.6 | 18136.6 |
|          6 |          -7640.6 | 15411.3 | 15851.7 |

**Regime statistics (full series):**

|   regime |   freq |   ann_return |   ann_vol |   avg_duration_days | label         |
|---------:|-------:|-------------:|----------:|--------------------:|:--------------|
|        0 |  0.17  |       -0.142 |     0.281 |              34.152 | Bear-volatile |
|        1 |  0.186 |        0.034 |     0.075 |              35.917 | Bull-calm     |
|        2 |  0.319 |        0.074 |     0.118 |              22.511 | Bull-calm     |
|        3 |  0.325 |        0.183 |     0.159 |              26.637 | Bull-volatile |

**Transition matrix** (rows = from-state, ordered by mean return):

|    |    S0 |    S1 |    S2 |    S3 |
|:---|------:|------:|------:|------:|
| S0 | 0.973 | 0     | 0.002 | 0.025 |
| S1 | 0     | 0.975 | 0.025 | 0     |
| S2 | 0.005 | 0.017 | 0.955 | 0.023 |
| S3 | 0.012 | 0     | 0.026 | 0.962 |

Implied expected regime duration (days), 1/(1−p_ii): S0=37.4, S1=40.5, S2=22.3, S3=26.0

## BITCOIN

BIC-optimal n_states on train = **5** (primary model uses 4 for interpretability/comparability).

**Model selection (train):**

|   n_states |   log_likelihood |     AIC |     BIC |
|-----------:|-----------------:|--------:|--------:|
|          2 |          -7065.5 | 14157   | 14235.7 |
|          3 |          -5726.5 | 11499.1 | 11638.3 |
|          4 |          -5040.1 | 10150.2 | 10362   |
|          5 |          -4430.1 |  8958.3 |  9254.8 |
|          6 |          -4540.4 |  9210.8 |  9604.2 |

**Regime statistics (full series):**

|   regime |   freq |   ann_return |   ann_vol |   avg_duration_days | label         |
|---------:|-------:|-------------:|----------:|--------------------:|:--------------|
|        0 |  0.067 |       -0.217 |     1.203 |              24.917 | Bear-volatile |
|        1 |  0.17  |       -0.239 |     0.739 |              19.125 | Bear-volatile |
|        2 |  0.373 |        0.546 |     0.531 |              23.577 | Bull-calm     |
|        3 |  0.39  |        0.285 |     0.301 |              40.698 | Bull-calm     |

**Transition matrix** (rows = from-state, ordered by mean return):

|    |    S0 |    S1 |    S2 |    S3 |
|:---|------:|------:|------:|------:|
| S0 | 0.957 | 0.043 | 0     | 0     |
| S1 | 0.012 | 0.944 | 0.041 | 0.003 |
| S2 | 0.003 | 0.018 | 0.958 | 0.021 |
| S3 | 0.002 | 0     | 0.035 | 0.964 |

Implied expected regime duration (days), 1/(1−p_ii): S0=23.3, S1=18.0, S2=23.7, S3=27.6
