"""Fit the primary HMM per asset and emit regime features for forecasting.

For each target: fit a Gaussian HMM (n_states from config) on the TRAIN split
only, then produce over the full series:

* ``regime``        — smoothed Viterbi state (descriptive; for plots/tables)
* ``regime_p0..pK`` — causal forward-filtered posteriors (leakage-safe model input)

Outputs ``data/processed/<asset>_regimes.parquet`` + a fitted-model artifact, and
writes ``reports/HMM_FINDINGS.md`` with per-asset regime statistics, transition
matrices, and the BIC model-selection table.

Run:  ``python -m src.models.build_regimes``
"""

from __future__ import annotations

import warnings

import joblib
import pandas as pd

from src.utils import io
from src.models import hmm_regimes as H

warnings.filterwarnings("ignore")
PROCESSED = io.PROCESSED_DIR


def build() -> None:
    cfg = io.load_config()
    feat = cfg["hmm"]["observation_features"]
    n_states = cfg["hmm"]["n_states"]
    cov = cfg["hmm"]["covariance_type"]
    n_iter = cfg["hmm"]["n_iter"]
    seed = cfg["hmm"]["random_state"]
    train_frac = cfg["split"]["train_frac"]
    candidates = cfg["hmm"]["n_states_candidates"]

    report = [
        "# HMM Regime Detection — Findings (Milestone 4)",
        "",
        f"_Gaussian HMM on {feat}, **{n_states} states**, full covariance. "
        f"Fitted on the first {train_frac:.0%} of each series; regimes generated over "
        "the full history. Causal forward-filtered posteriors are the leakage-safe "
        "model input; smoothed Viterbi states are used for interpretation._",
        "",
    ]

    for asset in io.target_keys():
        df = pd.read_parquet(PROCESSED / f"{asset}.parquet")
        n_tr = int(len(df) * train_frac)
        train = df.iloc[:n_tr]

        # --- model selection table (BIC) on train ---
        sel = H.select_n_states(train, feat, candidates, cov, n_iter, seed)

        # --- primary model fit on train ---
        hm = H.fit_regime_model(train, feat, n_states, cov, n_iter, seed)
        smoothed = H.smoothed_states(hm, df)
        filtered = H.filtered_state_probs(hm, df)
        stats = H.interpret(hm, df, smoothed)
        trans = H.transition_matrix(hm)

        # --- persist regime features (causal posteriors + smoothed state) ---
        out = filtered.copy()
        out["regime"] = smoothed.values
        out.to_parquet(PROCESSED / f"{asset}_regimes.parquet")
        joblib.dump(hm, PROCESSED / f"hmm_{asset}.joblib")

        print(f"{asset:8s}: {n_states} regimes | "
              f"labels={list(stats['label'])} | saved {asset}_regimes.parquet")

        # expected persistence from transition diagonal: 1/(1-p_ii)
        persistence = (1 / (1 - pd.Series(
            {f"S{i}": trans.iloc[i, i] for i in range(n_states)}))).round(1)

        report += [
            f"## {asset.upper()}",
            "",
            f"BIC-optimal n_states on train = **{sel['BIC'].idxmin()}** "
            f"(primary model uses {n_states} for interpretability/comparability).",
            "",
            "**Model selection (train):**",
            "",
            sel[["log_likelihood", "AIC", "BIC"]].round(1).to_markdown(),
            "",
            "**Regime statistics (full series):**",
            "",
            stats[["freq", "ann_return", "ann_vol", "avg_duration_days", "label"]]
            .round(3).to_markdown(),
            "",
            "**Transition matrix** (rows = from-state, ordered by mean return):",
            "",
            trans.round(3).to_markdown(),
            "",
            "Implied expected regime duration (days), 1/(1−p_ii): "
            + ", ".join(f"{k}={v}" for k, v in persistence.items()),
            "",
        ]

    (io.PROJECT_ROOT / "reports" / "HMM_FINDINGS.md").write_text("\n".join(report))
    print("\nWrote reports/HMM_FINDINGS.md")
    print(f"Regime features + models in {PROCESSED}")


if __name__ == "__main__":
    build()
