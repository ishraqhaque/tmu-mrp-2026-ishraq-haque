# Stock Market Regime Detection and Forecasting using HMM and LSTM Networks

Major Research Paper (MRP) — MSc Data Science & Analytics, Toronto Metropolitan University.
**Ishraq Haque** · Student ID 500947720

A hybrid pipeline that uses **Hidden Markov Models (HMM)** to detect latent market
regimes and **LSTM networks** to forecast next-day movements, benchmarked against
**ARIMA** and a **standalone LSTM**. Data is sourced from **Financial Modeling Prep (FMP)**.

> See [`reports/PROJECT_PLAN.md`](reports/PROJECT_PLAN.md) for the full plan, scope decisions, and milestones.

## Assets

| Role | Assets (FMP symbol) |
|------|---------------------|
| Forecast targets | S&P 500 (`^GSPC`), Gold (`GCUSD`), Bitcoin (`BTCUSD`) |
| Regime-signal features | VIX (`^VIX`), 10Y Treasury yield (`^TNX`) |

## Setup

> **Use Python 3.12.** TensorFlow and `hmmlearn` do not yet ship wheels for
> Python 3.13/3.14. A `.venv` built on Python 3.12 is already created in this repo.

```bash
# 1. Create and activate a virtual environment (3.12 — already done if .venv exists)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
#    Data pipeline only (fast):
pip install pandas requests python-dotenv pyyaml pyarrow tqdm
#    Full stack incl. modeling (TensorFlow, hmmlearn, statsmodels — larger):
pip install -r requirements.txt

# 3. Add your FMP API key
cp .env.example .env
#    then edit .env and paste your key:  FMP_API_KEY=...
```

> Get a key from the [FMP dashboard](https://site.financialmodelingprep.com/developer/docs).
> `.env` is gitignored — your key is never committed.

## Collect data

```bash
# All assets (targets + signals)
python -m src.data.collect

# Subset
python -m src.data.collect --only sp500 gold

# Quick API smoke test
python -m src.data.fmp_client
```

Raw daily OHLCV lands in `data/raw/<asset>.{parquet,csv}` plus a
`_collection_summary.csv` recording row counts and date coverage per asset.

## Project layout

```
config/         config.yaml — assets, dates, model hyperparameters (single source of truth)
src/
  data/         FMP client + collection script
  features/     feature engineering (returns, volatility, indicators)  [M3]
  models/       HMM, ARIMA, LSTM, hybrid HMM-LSTM                       [M4–M6]
  evaluation/   metrics, walk-forward CV, significance tests           [M7]
data/raw/       raw FMP pulls (gitignored)
data/processed/ cleaned, feature-engineered datasets (gitignored)
notebooks/      EDA and analysis notebooks
reports/        PROJECT_PLAN.md, figures, final paper
tests/          unit tests
```

## Reproducibility

Config-driven, fixed random seeds, pinned dependencies. All experiments are
re-runnable from `config/config.yaml`.
