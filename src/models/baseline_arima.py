"""ARIMA baseline — classical linear benchmark for next-day return forecasting.

Models the daily log-return series directly (returns are already stationary, so
d = 0). Order (p, q) is chosen by AIC on the training window. Test predictions use
a rolling one-step-ahead scheme: the model is fit once, then state is updated with
each realized return via ``append(refit=False)`` — fast and strictly causal.

Run:  ``python -m src.models.baseline_arima``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.utils import io
from src.evaluation import metrics as M
from src.evaluation.splits import chronological_split

warnings.filterwarnings("ignore")
PROCESSED = io.PROCESSED_DIR
RESULTS_DIR = io.PROJECT_ROOT / "reports" / "results"


def select_order(train: np.ndarray, max_p: int, max_q: int, d: int = 0) -> tuple[int, int, int]:
    """Grid-search (p, d, q) by AIC on the training returns."""
    best_aic, best_order = np.inf, (1, d, 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(train, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, (p, d, q)
            except Exception:
                continue
    return best_order


def rolling_forecast(returns: np.ndarray, train_n: int, order: tuple[int, int, int]) -> np.ndarray:
    """One-step-ahead forecasts for positions ``train_n .. len-1`` (causal)."""
    res = ARIMA(returns[:train_n], order=order).fit()
    preds = []
    for i in range(train_n, len(returns)):
        res = res.append([returns[i]], refit=False)   # state update, no re-estimation
        preds.append(float(res.forecast(1)[0]))        # predict return[i+1] = target_return[i]
    return np.array(preds)


def run() -> None:
    cfg = io.load_config()
    acfg = cfg["arima"]
    train_frac = cfg["split"]["train_frac"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for asset in io.target_keys():
        df = pd.read_parquet(PROCESSED / f"{asset}.parquet")
        lr = df["log_return"].values
        y_true = df["target_return"].values
        n = len(df)
        train_idx, test_idx = chronological_split(n, train_frac)
        train_n = len(train_idx)

        order = select_order(lr[:train_n], acfg["max_p"], acfg["max_q"], d=0)
        preds = rolling_forecast(lr, train_n, order)
        yt = y_true[test_idx]

        m = M.evaluate(yt, preds)
        m.update({"model": "arima", "asset": asset, "order": str(order)})
        rows.append(m)

        out = pd.DataFrame({"y_true": yt, "y_pred": preds}, index=df.index[test_idx])
        out.to_parquet(PROCESSED / f"pred_arima_{asset}.parquet")
        print(f"{asset:8s} ARIMA{order}: "
              f"MSE={m['mse']:.3e} MAE={m['mae']:.3e} DirAcc={m['directional_accuracy']:.3f}")

    res_df = pd.DataFrame(rows)[
        ["model", "asset", "order", "n", "mse", "rmse", "mae", "directional_accuracy", "f1"]
    ]
    res_path = RESULTS_DIR / "metrics_arima.csv"
    res_df.to_csv(res_path, index=False)
    print(f"\nSaved {res_path}")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run()
