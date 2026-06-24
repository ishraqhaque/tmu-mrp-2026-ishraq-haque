"""Standalone TFT baseline — 15 engineered features, no regime information.

The TFT counterpart of baseline_lstm.py and a comparison point for the HMM-TFT
hybrid (RQ3: backbone effect; RQ2: regime effect under the TFT backbone).

Run:  ``python -m src.models.baseline_tft``
"""

from __future__ import annotations

import pandas as pd

from src.models.tft_model import run_tft
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
        out = run_tft(df[feats], df["target_return"], cfg, label="tft")

        m = out["metrics"]; m["asset"] = asset; rows.append(m)
        for sm in out["seed_metrics"]:
            sm["asset"] = asset; seed_rows.append(sm)
        out["predictions"].to_parquet(PROCESSED / f"pred_tft_{asset}.parquet")
        print(f"{asset:8s} TFT (n_feat={len(feats)}): "
              f"MSE={m['mse']:.3e} MAE={m['mae']:.3e} DirAcc={m['directional_accuracy']:.3f}")

    cols = ["model", "asset", "n", "mse", "rmse", "mae", "directional_accuracy", "f1"]
    pd.DataFrame(rows)[cols].to_csv(RESULTS_DIR / "metrics_tft.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS_DIR / "metrics_tft_perseed.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR}")
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
