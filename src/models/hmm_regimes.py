"""Hidden Markov Model regime detection.

Fits a Gaussian HMM on return/volatility observations to recover latent market
regimes, selects the state count by BIC, and exposes two decoders:

* :func:`smoothed_states` — full-sequence Viterbi path, used for *descriptive*
  regime interpretation and visualization (RQ1).
* :func:`filtered_state_probs` — forward-algorithm posteriors that condition only
  on information up to time *t*. This is the **leakage-safe** regime signal fed to
  the forecasting models (RQ2), implemented explicitly so the math is auditable.

States are re-ordered canonically by mean return (state 0 = most bearish) so that
labels are stable across random restarts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
@dataclass
class HMMRegimeModel:
    model: GaussianHMM
    scaler: StandardScaler
    feature_cols: list[str]
    order: np.ndarray            # raw_state_index -> rank (0 = lowest mean return)
    n_states: int

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.feature_cols].values)


# --------------------------------------------------------------------------- #
def _n_params(n_states: int, n_features: int, cov_type: str) -> int:
    """Free-parameter count for BIC/AIC."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    if cov_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        covars = n_states * n_features
    elif cov_type == "spherical":
        covars = n_states
    else:  # tied
        covars = n_features * (n_features + 1) // 2
    return start + trans + means + covars


def fit_regime_model(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    n_states: int,
    cov_type: str = "full",
    n_iter: int = 200,
    seed: int = 42,
) -> HMMRegimeModel:
    """Standardize features (fit on train) and fit a Gaussian HMM."""
    scaler = StandardScaler().fit(df_train[feature_cols].values)
    X = scaler.transform(df_train[feature_cols].values)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cov_type,
        n_iter=n_iter,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X)

    # Canonical ordering by mean return (first feature is log_return by convention)
    raw_states = model.predict(X)
    ret = df_train[feature_cols[0]].values
    means = [ret[raw_states == s].mean() if (raw_states == s).any() else np.nan
             for s in range(n_states)]
    order = np.argsort(np.argsort(means))  # raw state -> rank
    return HMMRegimeModel(model, scaler, list(feature_cols), order, n_states)


def select_n_states(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    candidates: list[int],
    cov_type: str = "full",
    n_iter: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit each candidate state count and score by log-likelihood, AIC, BIC."""
    rows = []
    n_feat = len(feature_cols)
    N = len(df_train)
    for k in candidates:
        hm = fit_regime_model(df_train, feature_cols, k, cov_type, n_iter, seed)
        X = hm._scale(df_train)
        ll = hm.model.score(X)
        p = _n_params(k, n_feat, cov_type)
        rows.append(
            {
                "n_states": k,
                "log_likelihood": ll,
                "n_params": p,
                "AIC": -2 * ll + 2 * p,
                "BIC": -2 * ll + p * np.log(N),
                "converged": bool(hm.model.monitor_.converged),
            }
        )
    return pd.DataFrame(rows).set_index("n_states")


# --------------------------------------------------------------------------- #
def _emission_logprob(hm: HMMRegimeModel, X: np.ndarray) -> np.ndarray:
    """log P(o_t | state i) for every t, i — computed explicitly via scipy."""
    m = hm.model
    n = m.n_components
    logB = np.empty((len(X), n))
    for i in range(n):
        cov = m.covars_[i]
        cov = cov + 1e-6 * np.eye(cov.shape[0])  # jitter for numerical stability
        logB[:, i] = multivariate_normal.logpdf(X, mean=m.means_[i], cov=cov,
                                                 allow_singular=True)
    return logB


def smoothed_states(hm: HMMRegimeModel, df: pd.DataFrame) -> pd.Series:
    """Viterbi most-likely state path (ordered), using the full sequence."""
    X = hm._scale(df)
    raw = hm.model.predict(X)
    ordered = hm.order[raw]
    return pd.Series(ordered, index=df.index, name="regime")


def filtered_state_probs(hm: HMMRegimeModel, df: pd.DataFrame) -> pd.DataFrame:
    """Causal forward-filtered posteriors P(state_t | o_{0..t}); no lookahead.

    Columns are ordered states ``regime_p0 .. regime_p{K-1}``.
    """
    X = hm._scale(df)
    logB = _emission_logprob(hm, X)
    log_pi = np.log(hm.model.startprob_ + 1e-300)
    log_A = np.log(hm.model.transmat_ + 1e-300)
    n = hm.model.n_components
    T = len(X)

    log_alpha = np.empty((T, n))
    log_alpha[0] = log_pi + logB[0]
    log_alpha[0] -= logsumexp(log_alpha[0])           # normalize (filtered posterior)
    for t in range(1, T):
        # predict: P(state_t | o_{<t}) = sum_i alpha_{t-1}(i) A[i,j]
        pred = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
        log_alpha[t] = pred + logB[t]
        log_alpha[t] -= logsumexp(log_alpha[t])       # renormalize each step
    post = np.exp(log_alpha)

    # Reorder columns to canonical (ranked) state order
    ordered = np.empty_like(post)
    for raw_state in range(n):
        ordered[:, hm.order[raw_state]] = post[:, raw_state]
    cols = [f"regime_p{i}" for i in range(n)]
    return pd.DataFrame(ordered, index=df.index, columns=cols)


# --------------------------------------------------------------------------- #
def interpret(hm: HMMRegimeModel, df: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    """Per-regime descriptive statistics with human-readable labels."""
    ret_col = hm.feature_cols[0]
    vol_col = hm.feature_cols[1] if len(hm.feature_cols) > 1 else None
    rows = []
    for s in range(hm.n_states):
        mask = states == s
        sub = df.loc[mask]
        ann_ret = sub[ret_col].mean() * 252
        ann_vol = sub[ret_col].std() * np.sqrt(252)
        rows.append(
            {
                "regime": s,
                "freq": float(mask.mean()),
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "mean_vol_feat": float(sub[vol_col].mean()) if vol_col else np.nan,
                "avg_duration_days": _avg_duration(mask.values),
            }
        )
    stats = pd.DataFrame(rows).set_index("regime")
    med_vol = stats["ann_vol"].median()
    stats["label"] = [
        f"{'Bull' if r.ann_return >= 0 else 'Bear'}-{'volatile' if r.ann_vol > med_vol else 'calm'}"
        for r in stats.itertuples()
    ]
    return stats


def _avg_duration(mask: np.ndarray) -> float:
    """Average consecutive-run length where ``mask`` is True (regime persistence)."""
    if not mask.any():
        return 0.0
    runs, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def transition_matrix(hm: HMMRegimeModel) -> pd.DataFrame:
    """Transition matrix reordered to canonical state order."""
    n = hm.model.n_components
    A = hm.model.transmat_
    reordered = np.empty_like(A)
    inv = np.empty(n, dtype=int)
    for raw in range(n):
        inv[hm.order[raw]] = raw
    for i in range(n):
        for j in range(n):
            reordered[i, j] = A[inv[i], inv[j]]
    idx = [f"S{i}" for i in range(n)]
    return pd.DataFrame(reordered, index=idx, columns=idx)
