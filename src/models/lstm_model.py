"""LSTM forecaster — shared core for the standalone baseline and the hybrid.

The same training routine serves both models: the *standalone* baseline is fed the
15 engineered features; the *hybrid* (M6) is fed those plus the 4 causal HMM regime
posteriors. Isolating that single difference is exactly what tests RQ2.

Design
------
* Sliding 60-day windows predict the next-day log return (regression); direction is
  the sign of the prediction, so all models share one comparison basis.
* Feature and target scaling are fit on TRAIN ONLY; windows are strictly causal.
* Each model is trained under several seeds and ensembled (mean prediction) to damp
  run-to-run variance; per-seed predictions are retained for analysis.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF info/warning logs

# IMPORT ORDER IS LOAD-BEARING: tensorflow must be imported before pandas/numpy.
# On macOS (TF 2.21 / py3.12) importing pandas first deadlocks TensorFlow's first
# eager-kernel execution inside model.fit() — the process hangs at 0% CPU forever.
# Keep tensorflow as the first heavy import here, and import this module before
# pandas in any entry point (see baseline_lstm.py / lstm_model.py M5 debug notes).
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation import metrics as M
from src.evaluation.splits import chronological_split

tf.get_logger().setLevel("ERROR")


# --------------------------------------------------------------------------- #
def make_sequences(X: np.ndarray, y: np.ndarray, window: int):
    """Return (sequences, targets, end_positions) for every window end >= window-1."""
    n = len(X)
    seqs = np.stack([X[t - window + 1: t + 1] for t in range(window - 1, n)])
    targs = y[window - 1:]
    ends = np.arange(window - 1, n)
    return seqs, targs, ends


def _build_model(window: int, n_features: int, cfg: dict, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    layers = cfg["layers"]
    model = tf.keras.Sequential(name="lstm_forecaster")
    model.add(tf.keras.layers.Input(shape=(window, n_features)))
    for i in range(layers):
        model.add(tf.keras.layers.LSTM(cfg["units"], return_sequences=(i < layers - 1)))
        model.add(tf.keras.layers.Dropout(cfg["dropout"]))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]), loss="mse")
    return model


def run_lstm(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: dict,
    label: str = "lstm",
) -> dict:
    """Train a multi-seed LSTM ensemble and return test predictions + metrics.

    Parameters
    ----------
    features: aligned numeric feature frame (date-indexed).
    target:   aligned next-day log-return target.
    cfg:      full config dict (uses ``lstm`` and ``split``).
    """
    lcfg, scfg = cfg["lstm"], cfg["split"]
    window = lcfg["window"]
    seeds = lcfg["seeds"]

    X_raw = features.values.astype("float32")
    y_raw = target.values.astype("float32")
    n = len(X_raw)
    train_idx, test_idx = chronological_split(n, scfg["train_frac"])
    train_n = len(train_idx)

    # Scale on train only
    x_scaler = StandardScaler().fit(X_raw[:train_n])
    y_scaler = StandardScaler().fit(y_raw[:train_n].reshape(-1, 1))
    Xs = x_scaler.transform(X_raw).astype("float32")
    ys = y_scaler.transform(y_raw.reshape(-1, 1)).ravel().astype("float32")

    seqs, targs, ends = make_sequences(Xs, ys, window)
    is_train = ends < train_n
    # carve a chronological validation tail (last 12% of train sequences) for early stop
    tr_pos = np.where(is_train)[0]
    n_val = max(1, int(len(tr_pos) * 0.12))
    val_pos = tr_pos[-n_val:]
    fit_pos = tr_pos[:-n_val]
    test_pos = np.where(~is_train)[0]

    X_fit, y_fit = seqs[fit_pos], targs[fit_pos]
    X_val, y_val = seqs[val_pos], targs[val_pos]
    X_test = seqs[test_pos]
    test_dates = features.index[ends[test_pos]]
    y_test_true = y_raw[ends[test_pos]]

    per_seed = {}
    for seed in seeds:
        model = _build_model(window, Xs.shape[1], lcfg, seed)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
        model.fit(
            X_fit, y_fit,
            validation_data=(X_val, y_val),
            epochs=lcfg["epochs"], batch_size=lcfg["batch_size"],
            callbacks=[es], verbose=0,
        )
        pred_scaled = model.predict(X_test, verbose=0).ravel()
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        per_seed[seed] = pred
        tf.keras.backend.clear_session()

    ensemble = np.mean(np.column_stack(list(per_seed.values())), axis=1)
    m = M.evaluate(y_test_true, ensemble)
    m["model"] = label
    seed_metrics = [
        {**M.evaluate(y_test_true, p), "model": label, "seed": s}
        for s, p in per_seed.items()
    ]
    preds_df = pd.DataFrame(
        {"y_true": y_test_true, "y_pred": ensemble, **{f"seed_{s}": per_seed[s] for s in seeds}},
        index=test_dates,
    )
    return {"metrics": m, "seed_metrics": seed_metrics, "predictions": preds_df}
