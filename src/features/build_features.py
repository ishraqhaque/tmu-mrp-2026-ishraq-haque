"""Feature engineering: turn raw FMP OHLCV into the modeling matrix.

Design principles
-----------------
* **Leakage-safe.** Every feature at day *t* uses only information available up
  to and including *t*. Prediction targets are day *t+1*. Any train-only scaling
  happens later in the modeling stage, never here.
* **Transparent indicators.** RSI and MACD are implemented from first principles
  (no third-party TA black box) so the methodology is fully auditable.
* **Config-driven.** Windows, indicators, and merged signals come from
  ``config/config.yaml`` so experiments are reproducible.

Run:  ``python -m src.features.build_features``
Output: ``data/processed/<asset>.parquet`` + ``_feature_dictionary.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import io

PROCESSED_DIR = io.PROCESSED_DIR


# --------------------------------------------------------------------------- #
# Technical indicators (first-principles implementations)
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index. Uses EWMA smoothing (alpha = 1/window)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi_14")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal, and histogram, normalised by price for scale invariance."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = (ema_fast - ema_slow) / close          # normalise so it is comparable across assets
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def compute_features(asset: str, cfg: dict) -> pd.DataFrame:
    """Build the leakage-safe feature + target matrix for one target asset."""
    fcfg = cfg["features"]
    raw = io.load_raw(asset)                      # respects Bitcoin 2014+ clip
    close = raw["close"].rename("close")

    feat = pd.DataFrame(index=raw.index)
    feat["close"] = close

    # --- price/return features (all backward-looking) ---
    lr = io.log_returns(close).rename("log_return")
    feat["log_return"] = lr
    feat["ret_5d"] = np.log(close / close.shift(5))          # 5-day momentum

    for w in fcfg["rolling_vol_windows"]:
        feat[f"vol_{w}"] = lr.rolling(w).std()

    for w in fcfg["moving_average_windows"]:
        feat[f"ma{w}_ratio"] = close / close.rolling(w).mean() - 1.0   # distance from MA

    if fcfg.get("use_rsi", True):
        feat["rsi_14"] = rsi(close)
    if fcfg.get("use_macd", True):
        feat = feat.join(macd(close))

    # --- volume (OFF by default) ---
    # FMP volume coverage is inconsistent across series (absent before 1993 for
    # Gold and 2016 for Bitcoin), which would force dropping 2-3 years of
    # otherwise-usable price history. Excluded to keep a uniform feature set and
    # maximize the sample; re-enable via features.use_volume in config.
    if fcfg.get("use_volume", False) and "volume" in raw.columns and (raw["volume"] > 0).mean() > 0.5:
        vol = raw["volume"].replace(0, np.nan)
        feat["vol_z_21"] = (vol - vol.rolling(21).mean()) / vol.rolling(21).std()

    # --- merged macro / volatility signals (regime context) ---
    signals = io.load_signals()
    merge = fcfg.get("merge_signals", [])
    sig = pd.DataFrame(index=signals.index)
    if "vix" in merge and "vix" in signals:
        sig["vix"] = signals["vix"]
        sig["vix_chg"] = signals["vix"].diff()
    if "treasury" in merge or "tnx" in merge:
        if "y10" in signals:
            sig["y10"] = signals["y10"]
        if "slope_10y_2y" in signals:
            sig["slope_10y_2y"] = signals["slope_10y_2y"]
            sig["slope_chg"] = signals["slope_10y_2y"].diff()
    if not sig.empty:
        # Align to the asset's trading calendar; forward-fill so weekend crypto
        # rows inherit the most recent known macro state (never future data).
        sig = sig.reindex(feat.index).ffill()
        feat = feat.join(sig)

    # --- prediction targets (next day => shift -1) ---
    feat["target_return"] = lr.shift(-1)
    feat["target_direction"] = (feat["target_return"] > 0).astype("Int64")

    # Drop warm-up NaNs (rolling windows) and the final row (no next-day target).
    feat = feat.dropna()
    feat["target_direction"] = feat["target_direction"].astype(int)
    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns = everything except raw close and the targets."""
    exclude = {"close", "target_return", "target_direction"}
    return [c for c in df.columns if c not in exclude]


def build_all() -> None:
    cfg = io.load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    dict_lines = ["# Feature Dictionary", "", "Generated by `src/features/build_features.py`.", ""]

    for asset in io.target_keys():
        df = compute_features(asset, cfg)
        df.to_parquet(PROCESSED_DIR / f"{asset}.parquet")
        df.to_csv(PROCESSED_DIR / f"{asset}.csv")
        feats = feature_columns(df)
        up_rate = df["target_direction"].mean()
        summary.append(
            {
                "asset": asset,
                "rows": len(df),
                "n_features": len(feats),
                "start": df.index.min().date(),
                "end": df.index.max().date(),
                "up_rate": round(up_rate, 4),
            }
        )
        print(f"{asset:8s} {len(df):6d} rows  {len(feats)} features  "
              f"{df.index.min().date()}→{df.index.max().date()}  up={up_rate:.3f}")

    # Feature dictionary uses the last asset's columns (identical schema across assets).
    descriptions = {
        "log_return": "Daily log return, ln(C_t / C_{t-1})",
        "ret_5d": "5-day cumulative log return (momentum)",
        "vol_5": "Rolling 5-day std of log returns (short-term volatility)",
        "vol_21": "Rolling 21-day std of log returns (monthly volatility)",
        "ma10_ratio": "Close / 10-day MA − 1 (distance from short MA)",
        "ma50_ratio": "Close / 50-day MA − 1 (distance from long MA)",
        "rsi_14": "Wilder RSI(14), momentum oscillator in [0,100]",
        "macd": "MACD line (EMA12−EMA26)/Close",
        "macd_signal": "9-period EMA of MACD line",
        "macd_hist": "MACD − signal",
        "vol_z_21": "21-day z-score of trading volume",
        "vix": "CBOE Volatility Index level (regime signal)",
        "vix_chg": "Daily change in VIX",
        "y10": "10-Year Treasury par yield (%)",
        "slope_10y_2y": "10Y−2Y yield-curve slope (pp); negative = inverted",
        "slope_chg": "Daily change in yield-curve slope",
        "target_return": "TARGET: next-day log return (t+1)",
        "target_direction": "TARGET: 1 if next-day return > 0 else 0",
    }
    dict_lines.append("| Column | Description |")
    dict_lines.append("|--------|-------------|")
    for col in df.columns:
        dict_lines.append(f"| `{col}` | {descriptions.get(col, '—')} |")
    # Documentation lives in reports/ (data/processed/ is gitignored).
    (io.PROJECT_ROOT / "reports" / "FEATURE_DICTIONARY.md").write_text("\n".join(dict_lines))

    print("\nSummary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved processed data to {PROCESSED_DIR}")
    print("Feature dictionary -> reports/FEATURE_DICTIONARY.md")


if __name__ == "__main__":
    build_all()
