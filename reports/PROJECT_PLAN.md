# MRP Project Plan
## Stock Market Regime Detection and Forecasting using Hidden Markov Models and LSTM Networks

**Author:** Ishraq Haque (500947720) · MSc Data Science & Analytics, Toronto Metropolitan University
**Timeline:** ~2–3 months · **Data source:** Financial Modeling Prep (FMP, paid plan)
**Last updated:** 2026-06-04

---

## 1. Research questions (from proposal)

1. Can HMM effectively identify hidden market regimes in financial data?
2. Does incorporating regime information improve forecasting accuracy?
3. How does a hybrid HMM-LSTM model compare to ARIMA and standalone LSTM?

**Hypothesis** — H0: the hybrid model does not improve performance; H1: it does. Validated with a paired t-test (α = 0.05) on per-fold metrics.

## 2. Scope decisions (locked 2026-06-04)

| Decision | Choice | Rationale |
|---|---|---|
| Prediction targets | S&P 500, Gold, Bitcoin | Equity benchmark + safe-haven + high-volatility crypto → tests cross-market generalizability |
| Feature-only signals | VIX, 10Y Treasury yield (TNX) | Strong regime indicators; used as inputs, not separate prediction targets |
| Prediction target variable | Next-day log-return **and** direction (up/down) | Covers MSE/MAE (regression) + directional accuracy (classification) cleanly; avoids inflated R² of raw price level |
| Forecast horizon | 1 trading day (h=1) | Standard; extensible to multi-step later |
| Window length | 60 days | Matches Li 2019 / Jiang 2023 convention |
| Data source | FMP paid | Replaces Yahoo Finance |

## 3. Method overview

```
FMP raw OHLCV ──► cleaning/alignment ──► feature engineering ──┐
                                                               ├─► [A] ARIMA           (baseline)
   log-returns, rolling vol, MAs, RSI/MACD, VIX, TNX ──────────┼─► [B] standalone LSTM (baseline)
                                                               │
   Gaussian HMM (Baum-Welch) on return+vol features ─► regime ─┴─► [C] HMM-LSTM        (proposed)
   states (Viterbi)  ───────────────────────────────────────────► state appended as LSTM feature
```

- **HMM:** GaussianHMM on standardized [log-return, rolling volatility, (optional) VIX] features. Select n_states from {2,3,4,5,6} via BIC + regime interpretability. Viterbi decode → discrete regime per day; also keep state posterior probabilities.
- **Integration:** regime state (one-hot) and/or posterior probabilities concatenated to the LSTM input window (Hu 2024 style). Ablation: state vs. probabilities vs. none.
- **LSTM:** 1–2 layers (~50 units), 60-day window, Adam, MSE loss for returns + auxiliary BCE head for direction (or two separate models). Dropout 0.2.
- **ARIMA:** per-asset (p,d,q) via AIC; rolling refit for fair walk-forward comparison.

## 4. Evaluation protocol

- **Split:** 70/30 chronological (no shuffling). Walk-forward / expanding-window CV for robustness; multiple seeds for LSTM to reduce randomness.
- **Metrics:** MSE, MAE (returns) · Directional accuracy, F1 (direction). Optional: simple long/short strategy Sharpe as an economic-significance check.
- **Significance:** paired t-test on per-fold (or per-seed) metric differences, hybrid vs. each baseline, α = 0.05. Report effect sizes, not just p-values.
- **Leakage guards:** all scaling/feature stats fit on train only; HMM fit on train then applied forward; no future data in windows.

## 5. Milestones (≈10–12 weeks)

| # | Milestone | Output | Wk |
|---|---|---|---|
| M1 | **Data pipeline** — FMP client, collection, cleaning, alignment | `data/processed/*.parquet`, data dictionary | 1 |
| M2 | **EDA** — return distributions, volatility, correlations, regime-like periods | `notebooks/01_eda.ipynb`, figures | 1–2 |
| M3 | **Feature engineering** — log-returns, rolling vol, MAs, RSI/MACD, VIX/TNX merge | `src/features/`, feature matrix | 2 |
| M4 | **HMM regime detection** — fit, n_states selection, regime interpretation | `src/models/hmm.py`, regime plots | 3–4 |
| M5 | **Baselines** — ARIMA + standalone LSTM | `src/models/`, baseline metrics | 4–5 |
| M6 | **Hybrid HMM-LSTM** + ablations | proposed-model metrics | 6–7 |
| M6.1 | **TFT arms** — standalone TFT + HMM-TFT (pytorch-forecasting, per-asset) | `src/models/tft_model.py`, `baseline_tft.py`, `hybrid_tft.py`, TFT metrics | 6 |
| M6.2 | **VQ-VAE regime arms** — VQ-VAE-LSTM + VQ-VAE-TFT (see `MODERNIZED_ARMS_DESIGN.md`) | learned-regime metrics | 7 |
| M7 | **Evaluation + significance testing** across 3 assets | results tables, t-tests | 8 |
| M8 | **Paper draft** (YSGS format) — fill Methodology/Results/Discussion | MRP draft | 9–11 |
| M9 | **Review, polish, reproducibility** (GitHub, README, seeds) | final MRP + repo | 12 |

## 6. Paper structure (YSGS, per sample MRP)

Front matter (title page → author declaration → abstract+keywords → acknowledgements → TOC → list of figures/tables) → **1. Introduction** (background, RQs, variables) → **2. Literature Review** → **3. Exploratory Data Analysis** → **4. Methodology & Experiments** → **5. Results & Discussion** → **6. Conclusion & Future Work** → Appendices (codebook, GitHub link, data fields) → References (IEEE-numbered). Target ~30–35 pages, figure-heavy.

## 7. Reproducibility

Python project, fixed seeds, pinned `requirements.txt`, config-driven, all code on GitHub. API key kept out of source via `.env` (gitignored).

## 8. Key references (foundation)

1. Hu, D. (2024). Forecast Analysis of the Stock Market Based on HMM and LSTM (S&P500). *Dean & Francis*.
2. Liu, M., Huo, J., Wu, Y., Wu, J. (2021). Stock Market Trend Analysis Using HMM and LSTM. *arXiv:2104.09700*.
3. Li, X., Li, Y., Liu, X.-Y., Wang, C. D. (2019). Risk Management via Anomaly Circumvent: Mid-LSTM. *KDD Workshop on Anomaly Detection in Finance*.
4. Jiang, J., Wu, L., Zhao, H., Zhu, H., Zhang, W. (2023). Forecasting movements of stock time series based on hidden state guided deep learning (HMM-ALSTM). *Information Processing & Management*, 60, 103328.
