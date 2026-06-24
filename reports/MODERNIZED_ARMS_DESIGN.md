# Design Spec — Modernized Forecasting Arms (VQ-VAE Regimes + TFT)

**Author:** Ishraq Haque (500947720)
**Date:** 2026-06-18
**Status:** Proposed (awaiting review)
**Relates to:** `reports/PROJECT_PLAN.md` (M6–M7)

---

## 1. Goal

Extend the existing controlled experiment (ARIMA vs standalone LSTM vs HMM-LSTM)
with two modernized components, **added as new arms** rather than replacing the
originals. This preserves all existing baselines and the core "does regime
information help?" comparison, while widening the study into a clean factorial
that also tests *how the forecaster is built* and *how the regime is represented*.

The two additions:
- **Temporal Fusion Transformer (TFT)** — an attention-based forecaster, as an
  alternative backbone to the LSTM. Adds interpretability (variable importance
  across inputs, temporal attention) we can analyze and write about.
- **VQ-VAE regime detector** — a learned discrete latent-state model, as an
  alternative regime representation to the Gaussian HMM.

## 2. Model matrix

Two axes — **regime detector** × **forecaster** — plus ARIMA as the classical anchor.

| Regime ↓ / Forecaster → | LSTM | TFT |
|---|---|---|
| **None** | Standalone LSTM *(exists)* | **Standalone TFT** 🆕 |
| **HMM** | HMM-LSTM *(exists, M6 hybrid)* | **HMM-TFT** 🆕 |
| **VQ-VAE** | **VQ-VAE-LSTM** 🆕 | **VQ-VAE-TFT** 🆕 |

Plus **ARIMA** (classical baseline). **7 models total**, run on all three assets
(S&P 500, Gold, Bitcoin).

The factorial design isolates three effects, each holding the other axis fixed:
1. **Regime value** — None → HMM → VQ-VAE (original RQ2).
2. **Forecaster backbone** — LSTM → TFT.
3. **Regime representation** — HMM → VQ-VAE (new RQ4).

## 3. Research questions (updated)

Existing RQ1–RQ3 are unchanged. Added:

- **RQ4:** Does a learned discrete regime representation (VQ-VAE) capture market
  states more usefully than a Gaussian HMM, measured by downstream forecast gain?
- **RQ3 (extended):** The model comparison now spans classical (ARIMA),
  recurrent (LSTM), and attention-based (TFT) forecasters.

Hypothesis testing is unchanged in form: paired t-tests (α = 0.05) on per-fold /
per-seed metric differences, now applied pairwise across the expanded matrix
(e.g. HMM-TFT vs standalone TFT isolates regime value under the TFT backbone).

## 4. Methodology

### 4.1 Invariants for fair comparison (apply to every arm)
To keep the comparison clean, all arms share:
- **Same target:** next-day log-return (and its sign for direction), horizon = 1 day.
- **Same features:** the 15 engineered features from M3; regime arms add only the
  regime signal on top.
- **Same splits:** 70/30 chronological + walk-forward folds (`src/evaluation/splits.py`).
- **Same metrics:** MSE, RMSE, MAE, directional accuracy, F1 (`src/evaluation/metrics.py`).
- **Same leakage guards:** all scalers/statistics and both regime models fit on
  train only, then applied forward.
- **Same multi-seed protocol** for neural models (ensemble over seeds `[0,1,2]`).

### 4.2 TFT forecaster
- **Library:** `pytorch-forecasting` (PyTorch Lightning) or `darts` TFT. This adds
  a PyTorch dependency alongside the existing TensorFlow stack; the two run in
  separate processes/modules, so there is no runtime conflict. (Decision flagged
  in §7.)
- **Configuration:** single-step (h=1) to match the existing target. Input window
  = 60 days, matching the LSTM. Quantile output is available and optionally gives
  a distributional view of returns, but the primary reported metric stays the
  point forecast for parity with the other arms.
- **Covariate roles:** past observed = the 15 engineered features; static = asset
  id (when pooling); the regime signal enters as a time-varying input for the
  regime arms.
- **Interpretability outputs:** variable-importance weights and temporal attention,
  saved per asset for the Discussion section.

### 4.3 VQ-VAE regime detector
- **Input:** 60-day windows of the regime features (log-return, rolling
  volatility — the same observations the HMM uses), so each window is one training
  sample (~9k per asset). This sidesteps the small-N problem of daily points.
- **Codebook size:** matched to the HMM state count (default 4) so regime
  *cardinality* is controlled and the HMM-vs-VQ-VAE comparison is fair.
- **Causality:** encoder + codebook trained on the train period only; encoding a
  window uses only that window's own trailing 60 days, so test-period codes are
  causal. Mirrors the HMM "fit on train, decode forward" protocol.
- **Output:** a discrete regime code per day (one-hot) and the soft code
  assignment, fed to the forecaster exactly as the HMM regime is.

### 4.4 What feeds what
The regime signal (HMM or VQ-VAE) is concatenated to the forecaster input window,
identical to the current HMM-LSTM integration — the only thing that changes
between arms is which regime source (or none) supplies that signal.

## 5. Evaluation additions
No change to the protocol. The expanded matrix enables new pairwise tests:
- Regime value under each backbone (e.g. HMM-LSTM vs LSTM; HMM-TFT vs TFT).
- Backbone value at each regime level (e.g. TFT vs LSTM with HMM regimes).
- Regime representation (VQ-VAE-X vs HMM-X) — the RQ4 test.
Results go into the existing `reports/results/` tables, with one row per
model × asset, plus the per-seed file.

## 6. Revised milestone plan

The extension slots into M6–M7. Sequenced **TFT-first** so the lower-risk work
lands before the higher-risk VQ-VAE work, allowing graceful degradation.

| # | Milestone | Scope | Risk |
|---|---|---|---|
| M6  | HMM-LSTM hybrid + ablations *(original)* | existing | — |
| M6.1 | **TFT arms** — standalone TFT + HMM-TFT (reuses M4 regimes) | new | Low |
| M6.2 | **VQ-VAE regime** — build + stabilize; VQ-VAE-LSTM + VQ-VAE-TFT arms | new | High |
| M7  | Full evaluation + significance testing across the 7-model matrix | revised | — |

**Graceful degradation:** if M6.2 cannot stabilize within the timeline, the VQ-VAE
arms move to a partial-results / future-work section and the thesis still ships a
complete 5-model study (ARIMA, LSTM, TFT × {none, HMM}).

## 7. Decisions (resolved 2026-06-18)
1. **TFT library:** **`pytorch-forecasting`** (PyTorch Lightning). Most documented,
   strongest community baselines. Adds a PyTorch dependency alongside TensorFlow;
   the two run in separate modules/processes, so there is no runtime conflict.
2. **Pooling:** **one TFT per asset**, matching the current per-asset LSTM/ARIMA
   setup. Keeps the comparison parity clean (every arm is fit per asset); pooling
   is noted only as possible future work.

## 8. Explicitly out of scope (YAGNI)
The following from the broader institutional-grade discussion are **not** part of
this MRP, to protect the timeline and keep the contribution focused:
- BOCPD changepoint gating / real-time risk overlays.
- Diffusion / normalizing-flow risk-sizing layers, CVaR/Kelly position sizing.
- Mamba / state-space backbones.
- Intraday / tick resolution; multi-step (multi-day) price-trajectory forecasting.

These may be cited as future work but will not be implemented.
