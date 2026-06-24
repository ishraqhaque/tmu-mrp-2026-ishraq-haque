"""HMM-TFT hybrid — standalone TFT features PLUS causal HMM regime posteriors.

Isolates the value of HMM regime information under the TFT backbone (RQ2), and,
against HMM-LSTM, the value of the attention backbone holding regime fixed (RQ3).
The ONLY difference from baseline_tft.py is the appended regime_p* columns.

Run:  ``python -m src.models.hybrid_tft``
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
        feats = feature_columns(df)
        regime = io.load_regime_posteriors(asset, index=df.index)
        X = pd.concat([df[feats], regime], axis=1)       # 15 features + K regime posteriors
        out = run_tft(X, df["target_return"], cfg, label="hmm_tft")

        m = out["metrics"]; m["asset"] = asset; rows.append(m)
        for sm in out["seed_metrics"]:
            sm["asset"] = asset; seed_rows.append(sm)
        out["predictions"].to_parquet(PROCESSED / f"pred_hmm_tft_{asset}.parquet")
        print(f"{asset:8s} HMM-TFT (n_feat={X.shape[1]}): "
              f"MSE={m['mse']:.3e} MAE={m['mae']:.3e} DirAcc={m['directional_accuracy']:.3f}")

    cols = ["model", "asset", "n", "mse", "rmse", "mae", "directional_accuracy", "f1"]
    pd.DataFrame(rows)[cols].to_csv(RESULTS_DIR / "metrics_hmm_tft.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS_DIR / "metrics_hmm_tft_perseed.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR}")
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    run()
