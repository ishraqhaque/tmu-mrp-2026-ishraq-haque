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
# # Milestone 2 — Exploratory Data Analysis
#
# **Stock Market Regime Detection and Forecasting using HMM and LSTM Networks**
#
# This notebook characterises the three forecast targets (S&P 500, Gold, Bitcoin)
# and the two regime signals (VIX, Treasury yield curve). The goal is to surface
# the empirical properties — fat tails, volatility clustering, regime-dependent
# correlation — that motivate a regime-switching (HMM) approach over a single
# stationary model. Every figure is saved to `reports/figures/` and the headline
# statistics are written to `reports/EDA_FINDINGS.md`.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, no display server needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Make `src` importable when run from notebooks/ or project root
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import io  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = io.target_keys()
TARGET_LABELS = {"sp500": "S&P 500", "gold": "Gold", "bitcoin": "Bitcoin"}
PALETTE = {"sp500": "#1f77b4", "gold": "#d4a017", "bitcoin": "#f7931a"}

# Major crisis windows for shading/annotation
CRISES = {
    "Dot-com": ("2000-03-01", "2002-10-01"),
    "GFC": ("2007-10-01", "2009-03-01"),
    "COVID": ("2020-02-15", "2020-04-15"),
}
findings: dict[str, object] = {}
print("Targets:", TARGETS)

# %% [markdown]
# ## 1. Data coverage

# %%
closes = io.load_closes(TARGETS)
rets = io.load_returns(TARGETS, kind="log")
signals = io.load_signals()

coverage = pd.DataFrame(
    {
        "first": closes.apply(lambda s: s.first_valid_index()),
        "last": closes.apply(lambda s: s.last_valid_index()),
        "n_obs": closes.notna().sum(),
    }
)
print(coverage)
findings["coverage"] = coverage

# %% [markdown]
# ## 2. Price history (log scale) with crisis windows
# Long-horizon view motivates why a single regime cannot describe these series:
# each spans multiple bull/bear cycles and systemic shocks.

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 9), sharex=True)
for ax, asset in zip(axes, TARGETS):
    s = closes[asset].dropna()
    ax.plot(s.index, s.values, color=PALETTE[asset], lw=0.9)
    ax.set_yscale("log")
    ax.set_ylabel(f"{TARGET_LABELS[asset]}\n(log $)")
    for name, (lo, hi) in CRISES.items():
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="grey", alpha=0.15)
axes[0].set_title("Price history (log scale) with shaded crisis windows")
axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig01_price_history.png", dpi=150)
plt.close(fig)
print("saved fig01_price_history.png")

# %% [markdown]
# ## 3. Return distributions and non-normality
# Financial returns are leptokurtic (fat-tailed) and skewed — a key reason a
# Gaussian-on-everything model underperforms and regime mixtures help.

# %%
stat_rows = []
fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4))
for ax, asset in zip(axes, TARGETS):
    r = rets[asset].dropna()
    ax.hist(r, bins=120, density=True, color=PALETTE[asset], alpha=0.6)
    xs = np.linspace(r.min(), r.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "k--", lw=1, label="Normal fit")
    ax.set_title(TARGET_LABELS[asset])
    ax.set_xlabel("Daily log return")
    ax.legend(fontsize=8)
    jb_stat, jb_p = stats.jarque_bera(r)
    stat_rows.append(
        {
            "asset": asset,
            "n": len(r),
            "mean": r.mean(),
            "std": r.std(),
            "ann_vol": r.std() * np.sqrt(252),
            "skew": stats.skew(r),
            "excess_kurtosis": stats.kurtosis(r),  # Fisher: normal -> 0
            "min": r.min(),
            "max": r.max(),
            "JB_p": jb_p,
        }
    )
fig.suptitle("Daily log-return distributions vs. Normal fit")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig02_return_distributions.png", dpi=150)
plt.close(fig)

ret_stats = pd.DataFrame(stat_rows).set_index("asset")
print(ret_stats.round(4))
findings["ret_stats"] = ret_stats

# %% [markdown]
# ## 4. QQ plots (tail behaviour)

# %%
fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4))
for ax, asset in zip(axes, TARGETS):
    stats.probplot(rets[asset].dropna(), dist="norm", plot=ax)
    ax.set_title(f"{TARGET_LABELS[asset]} QQ")
    ax.get_lines()[0].set_markerfacecolor(PALETTE[asset])
    ax.get_lines()[0].set_markeredgecolor(PALETTE[asset])
    ax.get_lines()[0].set_markersize(2)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig03_qq_plots.png", dpi=150)
plt.close(fig)
print("saved fig03_qq_plots.png")

# %% [markdown]
# ## 5. Volatility clustering
# Large moves cluster in time (ARCH effect). Rolling 21-day volatility and the
# autocorrelation of squared returns make this explicit — the temporal structure
# the HMM is meant to capture as discrete regimes.

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 9), sharex=True)
for ax, asset in zip(axes, TARGETS):
    rv = rets[asset].rolling(21).std() * np.sqrt(252)
    ax.plot(rv.index, rv.values, color=PALETTE[asset], lw=0.8)
    ax.set_ylabel(f"{TARGET_LABELS[asset]}\nann. vol")
    for lo, hi in CRISES.values():
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="grey", alpha=0.15)
axes[0].set_title("Rolling 21-day annualised volatility")
axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig04_volatility_clustering.png", dpi=150)
plt.close(fig)


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.sum(x * x)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4))
acf_decay = {}
for ax, asset in zip(axes, TARGETS):
    r = rets[asset].dropna().values
    a_sq = acf(r ** 2, 40)
    a_raw = acf(r, 40)
    lags = np.arange(len(a_sq))
    ax.bar(lags, a_sq, color=PALETTE[asset], alpha=0.7, label="squared returns")
    ax.plot(lags, a_raw, "k.-", ms=3, lw=0.6, label="raw returns")
    ci = 1.96 / np.sqrt(len(r))
    ax.axhline(ci, color="red", ls=":", lw=0.8)
    ax.axhline(-ci, color="red", ls=":", lw=0.8)
    ax.set_title(TARGET_LABELS[asset])
    ax.set_xlabel("Lag")
    ax.legend(fontsize=8)
    acf_decay[asset] = float(a_sq[1])  # lag-1 ACF of squared returns
axes[0].set_ylabel("Autocorrelation")
fig.suptitle("ACF: raw vs. squared returns (volatility clustering)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_acf_squared_returns.png", dpi=150)
plt.close(fig)
findings["acf_sq_lag1"] = acf_decay
print("squared-return lag-1 ACF:", {k: round(v, 3) for k, v in acf_decay.items()})

# %% [markdown]
# ## 6. Regime signals: VIX and the yield curve
# VIX (the "fear gauge") and the 10Y-2Y slope are external conditioning signals
# that often lead or coincide with equity regime shifts.

# %%
fig, ax1 = plt.subplots(figsize=(12, 4))
sp = closes["sp500"].dropna()
ax1.plot(sp.index, sp.values, color=PALETTE["sp500"], lw=0.8, label="S&P 500 (log)")
ax1.set_yscale("log")
ax1.set_ylabel("S&P 500 (log $)", color=PALETTE["sp500"])
ax2 = ax1.twinx()
ax2.plot(signals.index, signals["vix"], color="firebrick", lw=0.5, alpha=0.6, label="VIX")
ax2.set_ylabel("VIX", color="firebrick")
ax1.set_title("S&P 500 vs. VIX")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig06_vix_overlay.png", dpi=150)
plt.close(fig)

if "slope_10y_2y" in signals.columns:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    slope = signals["slope_10y_2y"].dropna()
    ax.fill_between(slope.index, slope.values, 0, where=slope.values >= 0,
                    color="steelblue", alpha=0.6)
    ax.fill_between(slope.index, slope.values, 0, where=slope.values < 0,
                    color="crimson", alpha=0.7, label="inverted (10Y<2Y)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Treasury yield-curve slope (10Y − 2Y); inversions in red")
    ax.set_ylabel("Slope (pp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_yield_curve_slope.png", dpi=150)
    plt.close(fig)
    findings["pct_inverted"] = float((slope < 0).mean())
    print(f"yield curve inverted on {findings['pct_inverted']:.1%} of days since 1990")

# %% [markdown]
# ## 7. Regime-dependent cross-asset correlation
# Correlations are not constant — Gold/Bitcoin decouple from equities in calm
# periods but can spike during stress. Rolling 90-day correlation shows this.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
pairs = [("sp500", "gold"), ("sp500", "bitcoin"), ("gold", "bitcoin")]
for a, b in pairs:
    rc = rets[a].rolling(90).corr(rets[b])
    ax.plot(rc.index, rc.values, lw=0.8, label=f"{TARGET_LABELS[a]}–{TARGET_LABELS[b]}")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Rolling 90-day correlation of daily returns")
ax.set_ylabel("Correlation")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig08_rolling_correlations.png", dpi=150)
plt.close(fig)
print("saved fig08_rolling_correlations.png")

# %% [markdown]
# ## 8. A first look at volatility regimes
# Shading the S&P 500 by volatility tercile previews what the HMM should recover:
# persistent low-vol bull stretches punctuated by clustered high-vol crises.

# %%
sp_ret = rets["sp500"].dropna()
rv = (sp_ret.rolling(21).std() * np.sqrt(252)).dropna()
q1, q2 = rv.quantile([1 / 3, 2 / 3])
regime = pd.cut(rv, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
fig, ax = plt.subplots(figsize=(12, 4))
sp_p = closes["sp500"].reindex(rv.index)
ax.plot(sp_p.index, sp_p.values, color="black", lw=0.7)
ax.set_yscale("log")
colors = {"low": "#2ca02c", "mid": "#ff7f0e", "high": "#d62728"}
for lab, col in colors.items():
    mask = regime == lab
    ax.fill_between(sp_p.index, sp_p.min(), sp_p.values, where=mask.values,
                    color=col, alpha=0.12)
ax.set_title("S&P 500 shaded by 21-day volatility tercile (green=low, red=high)")
ax.set_ylabel("S&P 500 (log $)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig09_vol_regimes.png", dpi=150)
plt.close(fig)
findings["high_vol_frac"] = float((regime == "high").mean())
print("saved fig09_vol_regimes.png")

# %% [markdown]
# ## 9. Write findings report

# %%
def fmt(x, p=4):
    return f"{x:.{p}f}"


lines = [
    "# EDA Findings — Milestone 2",
    "",
    "_Auto-generated by `notebooks/01_eda.py`. Figures in `reports/figures/`._",
    "",
    "## Data coverage",
    "",
    coverage.to_markdown(),
    "",
    "## Return distribution statistics (daily log returns)",
    "",
    ret_stats.round(4).to_markdown(),
    "",
    "## Key observations",
    "",
]
for asset in TARGETS:
    rs = ret_stats.loc[asset]
    lines.append(
        f"- **{TARGET_LABELS[asset]}**: annualised vol "
        f"{fmt(rs['ann_vol']*100,1)}%, excess kurtosis {fmt(rs['excess_kurtosis'],1)} "
        f"(Normal = 0 → heavy tails), skew {fmt(rs['skew'],2)}, "
        f"Jarque–Bera p = {fmt(rs['JB_p'],3)} (≈0 ⇒ reject normality). "
        f"Lag-1 ACF of squared returns = {fmt(findings['acf_sq_lag1'][asset],3)} "
        f"(volatility clustering)."
    )
if "pct_inverted" in findings:
    lines.append(
        f"- **Yield curve** inverted (10Y<2Y) on {findings['pct_inverted']*100:.1f}% "
        f"of trading days since 1990 — a recognised pre-recession regime signal."
    )
lines += [
    f"- **S&P 500** spends {findings['high_vol_frac']*100:.0f}% of days in the "
    "top volatility tercile, concentrated in the shaded crisis windows — "
    "direct motivation for an explicit regime-switching model.",
    "",
    "## Implications for modelling",
    "",
    "1. Fat tails + non-normality ⇒ a single Gaussian model is misspecified; "
    "regime mixtures (HMM) better fit the conditional distribution.",
    "2. Strong volatility clustering ⇒ latent persistent states exist for the HMM to recover.",
    "3. Time-varying cross-asset correlation ⇒ regime context should improve "
    "multi-asset forecasting, supporting the HMM-LSTM hypothesis.",
    "",
]
report_path = PROJECT_ROOT / "reports" / "EDA_FINDINGS.md"
report_path.write_text("\n".join(lines))
print(f"Wrote {report_path}")
print("\nEDA complete. Figures:")
for p in sorted(FIG_DIR.glob("fig*.png")):
    print("  ", p.name)
