"""Generate the project methodology flow diagram (fig14_methodology.png).

A top-to-bottom flowchart of the experimental pipeline: data -> preprocessing ->
features -> regime detection -> forecasting models -> evaluation. Colourblind-safe
palette, no reliance on colour alone (labels carry the meaning).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "reports" / "figures" / "fig14_methodology.png"

# Colourblind-safe (Okabe-Ito) tints
C = {
    "data": "#E8F0FB", "prep": "#FBEFE0", "feat": "#E9F5E9",
    "regime": "#F3E8F5", "model": "#FDECEC", "eval": "#E8F6F8",
}
EDGE = "#33373d"


def box(ax, x, y, w, h, text, fc, fontsize=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, edgecolor=EDGE, facecolor=fc, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#15181c", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=EDGE, shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(9.2, 11))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")

ax.text(5, 12.6, "Regime-aware forecasting pipeline",
        ha="center", va="center", fontsize=13, weight="bold", color="#15181c")

# 1. Data sources
box(ax, 0.4, 11.2, 5.6, 0.95,
    "Targets (FMP daily OHLCV)\nS&P 500  ·  Gold  ·  Bitcoin", C["data"], 10, "bold")
box(ax, 6.3, 11.2, 3.3, 0.95,
    "Macro / volatility signals\nVIX  ·  US Treasury 2Y, 10Y", C["data"], 9.5)

# 2. Preprocessing
box(ax, 1.6, 9.85, 6.8, 0.85,
    "Preprocessing — cleaning, trading-calendar alignment, log-returns,\n"
    "leakage-safe handling (Bitcoin clipped to 2014+)", C["prep"], 9.5)

# 3. Features
box(ax, 1.6, 8.5, 6.8, 0.85,
    "Feature engineering (15 features)\n"
    "returns, rolling volatility, MA ratios, RSI(14), MACD, VIX, yield-curve slope",
    C["feat"], 9.5)

# 4. Regime detection
box(ax, 0.6, 6.95, 3.9, 1.0,
    "HMM regime detection\nGaussian HMM (Baum–Welch),\nBIC state selection,\ncausal posteriors",
    C["regime"], 9.5, "bold")
box(ax, 5.5, 6.95, 3.9, 1.0,
    "VQ-VAE regime detection\nlearned discrete codes\n(planned, M6.2)",
    C["regime"], 9.5)
ax.text(5.0, 6.55, "regime fit on TRAIN only → applied forward (no leakage)",
        ha="center", va="center", fontsize=8.5, style="italic", color="#444")

# 5. Forecasting models
box(ax, 0.4, 4.7, 2.9, 1.1,
    "ARIMA\n(classical baseline)", C["model"], 9.5)
box(ax, 3.55, 4.7, 2.9, 1.1,
    "LSTM\nstandalone · +HMM\n· +VQ-VAE", C["model"], 9.5, "bold")
box(ax, 6.7, 4.7, 2.9, 1.1,
    "TFT\nstandalone · +HMM\n· +VQ-VAE", C["model"], 9.5, "bold")
ax.text(5.0, 4.35,
        "controlled comparison: same target, features, splits, seeds — regime signal is the only variable",
        ha="center", va="center", fontsize=8.5, style="italic", color="#444")

# 6. Evaluation
box(ax, 1.3, 2.6, 7.4, 1.15,
    "Evaluation\n"
    "chronological 70/30 split + expanding walk-forward · multi-seed ensembles\n"
    "MSE / RMSE / MAE · directional accuracy / F1 · paired t-test (α = 0.05)",
    C["eval"], 9.5, "bold")

# 7. Output
box(ax, 2.6, 1.1, 4.8, 0.85,
    "Findings — does regime information improve\nshort-horizon forecasting? (RQ1–RQ4)",
    "#EFEFEF", 9.5, "bold")

# arrows
arrow(ax, 3.2, 11.2, 3.2, 10.72)      # targets -> prep
arrow(ax, 7.9, 11.2, 5.2, 10.72)      # signals -> prep
arrow(ax, 5.0, 9.85, 5.0, 9.37)       # prep -> features
arrow(ax, 3.5, 8.5, 2.5, 7.97)        # features -> HMM
arrow(ax, 6.5, 8.5, 7.5, 7.97)        # features -> VQVAE
arrow(ax, 5.0, 8.5, 5.0, 5.82)        # features -> models (straight down)
arrow(ax, 2.5, 6.95, 3.0, 5.82)       # HMM -> models
arrow(ax, 7.5, 6.95, 7.0, 5.82)       # VQVAE -> models
for cx in (1.85, 5.0, 8.15):
    arrow(ax, cx, 4.7, 5.0 if cx == 5.0 else cx, 3.77)  # models -> eval
arrow(ax, 5.0, 2.6, 5.0, 1.97)        # eval -> findings

fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
