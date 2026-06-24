"""Collect raw daily history for all configured assets from FMP.

Reads ``config/config.yaml``, pulls each target and signal symbol, and writes
one parquet + CSV per asset into ``data/raw/``. Run from the project root:

    python -m src.data.collect
    python -m src.data.collect --only sp500 gold        # subset
    python -m src.data.collect --format csv             # csv only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.fmp_client import FMPClient, FMPError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def collect(only: list[str] | None = None, fmt: str = "both") -> None:
    cfg = load_config()
    data_cfg = cfg["data"]
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    raw_dir = PROJECT_ROOT / data_cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Flatten targets + signals into one {key: symbol} map with a role tag.
    universe: dict[str, dict] = {}
    for key, meta in cfg.get("targets", {}).items():
        universe[key] = {**meta, "role": "target"}
    for key, meta in cfg.get("signals", {}).items():
        universe[key] = {**meta, "role": "signal"}

    if only:
        missing = [k for k in only if k not in universe]
        if missing:
            raise SystemExit(f"Unknown asset key(s): {missing}. Known: {list(universe)}")
        universe = {k: v for k, v in universe.items() if k in only}

    client = FMPClient()
    summary: list[dict] = []

    for key, meta in universe.items():
        endpoint = meta.get("endpoint", "eod")
        symbol = meta.get("fmp_symbol", f"treasury-rates:{meta.get('tenors')}")
        print(f"→ {key:9s} ({symbol})  [{meta['role']}] …", end=" ", flush=True)
        try:
            if endpoint == "treasury":
                df = client.get_treasury_rates(start=start, end=end, tenors=meta.get("tenors"))
            else:
                df = client.get_daily_history(meta["fmp_symbol"], start=start, end=end)
        except FMPError as err:
            print(f"FAILED: {err}")
            summary.append({"asset": key, "symbol": symbol, "rows": 0, "status": "FAILED"})
            continue

        if fmt in ("parquet", "both"):
            df.to_parquet(raw_dir / f"{key}.parquet")
        if fmt in ("csv", "both"):
            df.to_csv(raw_dir / f"{key}.csv")

        first, last = df.index.min().date(), df.index.max().date()
        print(f"{len(df):>6,} rows  {first} → {last}")
        summary.append(
            {
                "asset": key,
                "symbol": symbol,
                "role": meta["role"],
                "rows": len(df),
                "start": str(first),
                "end": str(last),
                "status": "OK",
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_path = raw_dir / "_collection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nCollection summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw data to {raw_dir}")
    print(f"Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw daily data from FMP.")
    parser.add_argument("--only", nargs="*", help="Subset of asset keys to fetch.")
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "both"],
        default="both",
        help="Output format (default: both).",
    )
    args = parser.parse_args()
    collect(only=args.only, fmt=args.format)


if __name__ == "__main__":
    main()
