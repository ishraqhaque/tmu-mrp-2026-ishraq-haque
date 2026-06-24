"""HMM-LSTM hybrid — the proposed model: standalone LSTM features PLUS causal HMM
regime posteriors.

This is the core RQ2 test: identical LSTM architecture and training to the
standalone baseline, the ONLY difference being the appended HMM regime posteriors.
Any performance gap versus baseline_lstm.py is attributable to the regime signal.

Run:  ``python -m src.models.hybrid_lstm``
"""

from __future__ import annotations

# Load the tensorflow-bearing module FIRST: tensorflow must be imported before
# pandas to avoid a macOS import-order deadlock in model.fit() (see lstm_model.py).
from src.models.lstm_model import run_lstm

import pandas as pd

from src.utils import io
from src.features.build_features import feature_columns

PROCESSED = io.PROCESSED_DIR
RESULTS_DIR = io.PROJECT_ROOT / "reports" / "results"


def run() -> None:
    cfg = io.load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, seed_rows = [], []

    for asset in io.target_keys():
        df = pd.read_parquet(PROCESSED / f"{asset}.parquet")
        feats = feature_columns(df)
        regime = io.load_regime_posteriors(asset, index=df.index)
        X = pd.concat([df[feats], regime], axis=1)       # 15 features + K regime posteriors
        out = run_lstm(X, df["target_return"], cfg, label="hmm_lstm")

        m = out["metrics"]; m["asset"] = asset; rows.append(m)
        for sm in out["seed_metrics"]:
            sm["asset"] = asset; seed_rows.append(sm)
        out["predictions"].to_parquet(PROCESSED / f"pred_hmm_lstm_{asset}.parquet")
        print(f"{asset:8s} HMM-LSTM (n_feat={X.shape[1]}): "
              f"MSE={m['mse']:.3e} MAE={m['mae']:.3e} DirAcc={m['directional_accuracy']:.3f}")

    cols = ["model", "asset", "n", "mse", "rmse", "mae", "directional_accuracy", "f1"]
    pd.DataFrame(rows)[cols].to_csv(RESULTS_DIR / "metrics_hmm_lstm.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS_DIR / "metrics_hmm_lstm_perseed.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR}")
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
