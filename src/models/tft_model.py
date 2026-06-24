"""Temporal Fusion Transformer forecaster — shared core for the TFT arms.

Mirrors src/models/lstm_model.run_lstm: same inputs, same return structure, so the
standalone-TFT and HMM-TFT arms slot into the existing evaluation tables unchanged.
One model per asset; next-day log-return target; 60-day encoder; multi-seed ensemble.
Leakage-safe: target normalization is fit by pytorch-forecasting on the training
rows only, and every test prediction uses only its own trailing encoder window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

from src.evaluation import metrics as M
from src.evaluation.splits import chronological_split


def make_long_frame(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Build the pytorch-forecasting long frame: integer time_idx, single series id,
    target column, and feature covariates. Returns (frame, feature_column_names)."""
    feat_cols = list(features.columns)
    df = features.copy().reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["series"] = "a"
    df["target_return"] = target.to_numpy(dtype="float32")
    df[feat_cols] = df[feat_cols].astype("float32")
    return df, feat_cols


def _build_datasets(df, feat_cols, window, train_n, val_n):
    fit_cutoff = train_n - val_n               # rows [0, fit_cutoff) train the weights
    # Normalizer + weights fit on the fit-split only; val tail (fit_cutoff..train_n)
    # drives early stopping; test block is everything from train_n on.
    training = TimeSeriesDataSet(
        df[df.time_idx < fit_cutoff],
        time_idx="time_idx", target="target_return", group_ids=["series"],
        max_encoder_length=window, max_prediction_length=1,
        time_varying_unknown_reals=feat_cols + ["target_return"],
        time_varying_known_reals=["time_idx"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True, add_target_scales=True,
    )
    # Validation is capped at the train boundary so early stopping never sees the
    # test period: val targets span only [fit_cutoff, train_n). Test targets start
    # at train_n. (Without the cap, from_dataset(df, ...) would emit val targets all
    # the way to n-1, leaking test signal into the stopping decision.)
    validation = TimeSeriesDataSet.from_dataset(
        training, df[df.time_idx < train_n], predict=False,
        stop_randomization=True, min_prediction_idx=fit_cutoff)
    test = TimeSeriesDataSet.from_dataset(
        training, df, predict=False, stop_randomization=True, min_prediction_idx=train_n)
    return training, validation, test


def run_tft(features: pd.DataFrame, target: pd.Series, cfg: dict, label: str = "tft") -> dict:
    """Train a multi-seed TFT ensemble and return test predictions + metrics.

    Parameters
    ----------
    features: aligned numeric feature frame (date-indexed).
    target:   aligned next-day log-return target.
    cfg:      full config dict (uses ``tft`` and ``split``).
    label:    model name recorded in the metrics rows.
    """
    tcfg, scfg = cfg["tft"], cfg["split"]
    window = tcfg["window"]
    seeds = tcfg["seeds"]

    df, feat_cols = make_long_frame(features, target)
    n = len(df)
    train_idx, test_idx = chronological_split(n, scfg["train_frac"])
    train_n = len(train_idx)
    val_n = max(window + 1, int(train_n * 0.12))   # chronological val tail of train

    training, validation, test = _build_datasets(df, feat_cols, window, train_n, val_n)
    train_dl = training.to_dataloader(train=True, batch_size=tcfg["batch_size"], num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=tcfg["batch_size"] * 4, num_workers=0)
    test_dl = test.to_dataloader(train=False, batch_size=512, num_workers=0)

    per_seed = {}
    test_time = None
    for seed in seeds:
        pl.seed_everything(seed, workers=True)
        tft = TemporalFusionTransformer.from_dataset(
            training, learning_rate=tcfg["learning_rate"], hidden_size=tcfg["hidden_size"],
            attention_head_size=tcfg["attention_head_size"], dropout=tcfg["dropout"],
            hidden_continuous_size=tcfg["hidden_continuous_size"],
        )
        es = EarlyStopping(monitor="val_loss", patience=tcfg["patience"], mode="min")
        trainer = pl.Trainer(
            max_epochs=tcfg["max_epochs"], accelerator="cpu", callbacks=[es],
            enable_progress_bar=False, enable_model_summary=False, logger=False,
            enable_checkpointing=False, gradient_clip_val=0.1,
        )
        trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        # Prediction returns a Prediction namedtuple with fields: output, x, index,
        # decoder_lengths, y.  When return_index=True, out.index is a DataFrame
        # with columns ["time_idx", "series"] — the integer row index of the
        # predicted step (decoder step 0) for each sample.
        out = tft.predict(test_dl, mode="prediction", return_index=True,
                          trainer_kwargs={"logger": False})
        preds = np.asarray(out.output).ravel().astype(float)
        if test_time is None:
            test_time = out.index["time_idx"].to_numpy()
        per_seed[seed] = preds

    test_dates = features.index[test_time]
    y_true = target.to_numpy(dtype=float)[test_time]

    ensemble = np.mean(np.column_stack([per_seed[s] for s in seeds]), axis=1)
    m = M.evaluate(y_true, ensemble)
    m["model"] = label
    seed_metrics = [
        {**M.evaluate(y_true, per_seed[s]), "model": label, "seed": s}
        for s in seeds
    ]
    preds_df = pd.DataFrame(
        {"y_true": y_true, "y_pred": ensemble, **{f"seed_{s}": per_seed[s] for s in seeds}},
        index=test_dates,
    )
    return {"metrics": m, "seed_metrics": seed_metrics, "predictions": preds_df}
