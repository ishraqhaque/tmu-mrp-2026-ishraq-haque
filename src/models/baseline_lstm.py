"""Standalone LSTM baseline — 15 engineered features, no regime information.

This is the direct comparison point for the hybrid (M6): identical architecture and
training, the only difference being the absence of HMM regime posteriors.

Run:  ``python -m src.models.baseline_lstm``
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
        feats = feature_columns(df)                      # 15 features, no regime
        out = run_lstm(df[feats], df["target_return"], cfg, label="lstm")

        m = out["metrics"]
        m["asset"] = asset
        rows.append(m)
        for sm in out["seed_metrics"]:
            sm["asset"] = asset
            seed_rows.append(sm)
        out["predictions"].to_parquet(PROCESSED / f"pred_lstm_{asset}.parquet")
        print(f"{asset:8s} LSTM (n_feat={len(feats)}): "
              f"MSE={m['mse']:.3e} MAE={m['mae']:.3e} DirAcc={m['directional_accuracy']:.3f}")

    cols = ["model", "asset", "n", "mse", "rmse", "mae", "directional_accuracy", "f1"]
    pd.DataFrame(rows)[cols].to_csv(RESULTS_DIR / "metrics_lstm.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS_DIR / "metrics_lstm_perseed.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR}")
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
