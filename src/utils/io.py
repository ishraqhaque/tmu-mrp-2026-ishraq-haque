"""Shared data-loading helpers used by EDA, feature engineering, and modeling.

Single place to read raw FMP pulls, compute returns, and assemble aligned
panels so every downstream stage uses identical definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Bitcoin's pre-2014 FMP history is near-zero and illiquid; start analysis where
# real, liquid pricing exists. Documented in PROJECT_PLAN / Methodology.
ASSET_MIN_DATE = {"bitcoin": "2014-01-01"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def target_keys() -> list[str]:
    return list(load_config().get("targets", {}).keys())


def signal_keys() -> list[str]:
    return list(load_config().get("signals", {}).keys())


def load_raw(asset: str, apply_min_date: bool = True) -> pd.DataFrame:
    """Load one asset's raw daily frame from parquet (date-indexed, ascending)."""
    path = RAW_DIR / f"{asset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.collect` first.")
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    if apply_min_date and asset in ASSET_MIN_DATE:
        df = df[df.index >= pd.Timestamp(ASSET_MIN_DATE[asset])]
    return df


def get_close(asset: str) -> pd.Series:
    """Closing price series for a target asset."""
    s = load_raw(asset)["close"].rename(asset)
    return s


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns of a price series."""
    return np.log(prices / prices.shift(1)).rename(prices.name)


def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().rename(prices.name)


def load_closes(assets: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Aligned wide frame of closing prices across the given assets (outer join)."""
    assets = list(assets) if assets is not None else target_keys()
    cols = {a: get_close(a) for a in assets}
    return pd.concat(cols, axis=1).sort_index()


def load_returns(assets: Optional[Iterable[str]] = None, kind: str = "log") -> pd.DataFrame:
    """Aligned wide frame of daily returns across the given assets."""
    closes = load_closes(assets)
    fn = log_returns if kind == "log" else simple_returns
    return closes.apply(fn)


def load_signals() -> pd.DataFrame:
    """VIX level and Treasury yields (2Y, 10Y) plus the 10Y-2Y slope, aligned."""
    out = {}
    cfg = load_config().get("signals", {})
    if "vix" in cfg:
        out["vix"] = load_raw("vix", apply_min_date=False)["close"]
    if "treasury" in cfg:
        tr = load_raw("treasury", apply_min_date=False)
        if "year10" in tr:
            out["y10"] = tr["year10"]
        if "year2" in tr:
            out["y2"] = tr["year2"]
    df = pd.concat(out, axis=1).sort_index()
    if {"y10", "y2"}.issubset(df.columns):
        df["slope_10y_2y"] = df["y10"] - df["y2"]
    return df


def load_regime_posteriors(asset: str, index: Optional[pd.Index] = None) -> pd.DataFrame:
    """Causal HMM regime posteriors (``regime_p0..pK``) for one asset.

    The descriptive ``regime`` (Viterbi) column is dropped — only the leakage-safe
    forward-filtered posteriors are model inputs. When ``index`` is given, the
    frame is reindexed to it (the regime file is built from the same processed
    frame, so this is an exact alignment).
    """
    path = PROCESSED_DIR / f"{asset}_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.models.build_regimes` first.")
    post = pd.read_parquet(path)
    post = post[[c for c in post.columns if c.startswith("regime_p")]]
    if index is not None:
        post = post.reindex(index)
    return post
