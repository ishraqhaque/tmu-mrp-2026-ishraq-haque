# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#   kernelspec:
#     display_name: Python 3 (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Milestone 4 — HMM Regime Detection
#
# Visualizes the fitted Gaussian HMMs: BIC model selection, regime-colored price
# paths, transition matrices, and the causal (leakage-safe) regime posteriors that
# feed the forecasting models. Models and regime features are produced by
# `src/models/build_regimes.py`; this notebook loads those artifacts.

# %%
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import io                          # noqa: E402
from src.models import hmm_regimes as H           # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED = io.PROCESSED_DIR
cfg = io.load_config()
feat = cfg["hmm"]["observation_features"]
N_STATES = cfg["hmm"]["n_states"]
TARGETS = io.target_keys()
LABELS = {"sp500": "S&P 500", "gold": "Gold", "bitcoin": "Bitcoin"}
# Regime colors ordered bearish -> bullish (state 0 = lowest mean return)
REGIME_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c"]
REGIME_CMAP = ListedColormap(REGIME_COLORS[:N_STATES])

# %% [markdown]
# ## 1. BIC model selection
# Lower BIC = better. BIC keeps improving with more states, but we fix **4** for
# interpretability and cross-asset comparability (sensitivity to 3/5 reported).

# %%
candidates = cfg["hmm"]["n_states_candidates"]
fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4), sharex=True)
for ax, asset in zip(axes, TARGETS):
    df = pd.read_parquet(PROCESSED / f"{asset}.parquet")
    train = df.iloc[: int(len(df) * cfg["split"]["train_frac"])]
    sel = H.select_n_states(train, feat, candidates, cfg["hmm"]["covariance_type"],
                            cfg["hmm"]["n_iter"], cfg["hmm"]["random_state"])
    ax.plot(sel.index, sel["BIC"], "o-", label="BIC")
    ax.plot(sel.index, sel["AIC"], "s--", color="grey", label="AIC", alpha=0.7)
    ax.axvline(N_STATES, color="green", ls=":", lw=1.5, label=f"chosen={N_STATES}")
    ax.set_title(LABELS[asset])
    ax.set_xlabel("n_states")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Information criterion")
fig.suptitle("HMM model selection by information criteria (lower = better)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_bic_selection.png", dpi=150)
plt.close(fig)
print("saved fig10_bic_selection.png")

# %% [markdown]
# ## 2. Regime-colored price paths
# The smoothed Viterbi state painted onto each price series. Red = bearish/volatile
# crisis regime; green = calm bull. Crisis regimes concentrate in 2000–02, 2008, 2020.

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12, 10), sharex=False)
for ax, asset in zip(axes, TARGETS):
    df = pd.read_parquet(PROCESSED / f"{asset}.parquet")
    reg = pd.read_parquet(PROCESSED / f"{asset}_regimes.parquet")["regime"]
    price = df["close"]
    ax.scatter(price.index, price.values, c=reg.values, cmap=REGIME_CMAP,
               s=3, vmin=0, vmax=N_STATES - 1)
    ax.set_yscale("log")
    ax.set_ylabel(f"{LABELS[asset]} (log $)")
axes[0].set_title(f"Price paths colored by HMM regime "
                  f"(red=bear/volatile → green=bull/calm, {N_STATES} states)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig11_regime_price_paths.png", dpi=150)
plt.close(fig)
print("saved fig11_regime_price_paths.png")

# %% [markdown]
# ## 3. Transition matrices
# Strong diagonals confirm persistent regimes (the property an HMM exploits and a
# naive volatility threshold lacks).

# %%
fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4))
for ax, asset in zip(axes, TARGETS):
    hm = joblib.load(PROCESSED / f"hmm_{asset}.joblib")
    trans = H.transition_matrix(hm)
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                cbar=False, ax=ax)
    ax.set_title(f"{LABELS[asset]} transitions")
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
fig.suptitle("HMM transition matrices (S0 = most bearish … S3 = most bullish)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig12_transition_matrices.png", dpi=150)
plt.close(fig)
print("saved fig12_transition_matrices.png")

# %% [markdown]
# ## 4. Causal regime posteriors (the model input)
# Forward-filtered P(regime | past) for the S&P 500 — what the LSTM actually sees.
# These use only information up to each date (no lookahead).

# %%
asset = "sp500"
reg = pd.read_parquet(PROCESSED / f"{asset}_regimes.parquet")
probs = reg[[c for c in reg.columns if c.startswith("regime_p")]]
price = pd.read_parquet(PROCESSED / f"{asset}.parquet")["close"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(price.index, price.values, color="black", lw=0.7)
ax1.set_yscale("log")
ax1.set_ylabel("S&P 500 (log $)")
ax1.set_title("S&P 500 and causal HMM regime posteriors (forward-filtered, no lookahead)")
ax2.stackplot(probs.index, probs.values.T, colors=REGIME_COLORS[:N_STATES],
              labels=[f"S{i}" for i in range(N_STATES)])
ax2.set_ylim(0, 1)
ax2.set_ylabel("P(regime | past)")
ax2.legend(loc="upper left", ncol=N_STATES, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig13_regime_posteriors_sp500.png", dpi=150)
plt.close(fig)
print("saved fig13_regime_posteriors_sp500.png")
print("\nMilestone 4 figures complete.")
