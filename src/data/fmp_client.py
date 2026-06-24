"""Thin client for the Financial Modeling Prep (FMP) historical-price API.

Replaces Yahoo Finance as the project's data source. Handles the FMP ticker
conventions (``^`` for indices, ``*USD`` for commodities/crypto), retries on
transient errors, and returns tidy daily OHLCV DataFrames.

Docs: https://site.financialmodelingprep.com/developer/docs
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # read FMP_API_KEY from .env if present

# FMP's legacy /api/v3/historical-price-full endpoint was retired in 2024; the
# current "stable" API is the only one accessible to new keys.
FMP_STABLE = "https://financialmodelingprep.com/stable"

# Columns we keep from the stable EOD payload, in canonical order. The stable
# endpoint returns a flat list of records (no `adjClose`); for indices,
# commodities and crypto there are no splits/dividends, so close == adjClose.
_KEEP_COLS = ["date", "open", "high", "low", "close", "adjClose", "volume"]


class FMPError(RuntimeError):
    """Raised when the FMP API returns an error or no usable data."""


class FMPClient:
    """Minimal FMP historical-price client.

    Parameters
    ----------
    api_key:
        FMP API key. Falls back to the ``FMP_API_KEY`` environment variable.
    max_retries:
        Number of retry attempts on network / 429 / 5xx errors.
    pause:
        Base seconds to sleep between retries (exponential backoff).
    """

    # The stable EOD endpoint returns at most this many rows per request, so
    # full history must be assembled by paginating backward in time.
    PAGE_LIMIT = 5000

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 4,
        pause: float = 1.5,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise FMPError(
                "No FMP API key found. Set FMP_API_KEY in your environment or .env file. "
                "See .env.example."
            )
        self.max_retries = max_retries
        self.pause = pause
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: dict) -> dict | list:
        """GET with retry/backoff. Returns parsed JSON."""
        params = {**params, "apikey": self.api_key}
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:  # rate limited
                    raise FMPError("HTTP 429 rate limit")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, FMPError) as err:
                last_err = err
                if attempt < self.max_retries:
                    sleep_s = self.pause * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
        raise FMPError(f"FMP request failed after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    def _paginate_backward(
        self,
        fetch_chunk,
        start: Optional[str],
        end: Optional[str],
        label: str,
    ) -> pd.DataFrame:
        """Walk backward in time, ``PAGE_LIMIT`` rows at a time, until ``start``.

        ``fetch_chunk(to_date)`` must return a date-indexed DataFrame of the most
        recent rows ending on or before ``to_date``. The stable API caps each
        response at ~5,000 rows, so a single call cannot reach the 1990s; we
        repeatedly query with ``to`` set just before the earliest row seen.
        """
        start_ts = pd.Timestamp(start) if start else None
        cursor_to = end  # None => server default (today)
        frames: list[pd.DataFrame] = []
        seen_earliest: Optional[pd.Timestamp] = None

        while True:
            chunk = fetch_chunk(cursor_to)
            if chunk.empty:
                break
            frames.append(chunk)
            chunk_earliest = chunk.index.min()

            # Stop if the page wasn't full (reached the start of available data)
            if len(chunk) < self.PAGE_LIMIT:
                break
            # Stop if we've gone back far enough
            if start_ts is not None and chunk_earliest <= start_ts:
                break
            # Stop if pagination is not advancing (safety against infinite loop)
            if seen_earliest is not None and chunk_earliest >= seen_earliest:
                break
            seen_earliest = chunk_earliest
            cursor_to = (chunk_earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            time.sleep(0.2)  # be polite between paginated calls

        if not frames:
            raise FMPError(f"{label}: no data returned (check symbol / plan access)")

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    # ------------------------------------------------------------------ #
    def get_daily_history(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch full daily OHLCV history for ``symbol`` (paginated to full depth).

        Parameters
        ----------
        symbol:
            FMP symbol, e.g. ``^GSPC``, ``GCUSD``, ``BTCUSD``, ``^VIX``.
        start, end:
            ISO ``YYYY-MM-DD`` bounds (inclusive). ``None`` => no bound on that side.

        Returns
        -------
        DataFrame indexed by ``date`` (DatetimeIndex, ascending) with columns
        ``open, high, low, close, adjClose, volume`` (``adjClose`` mirrors
        ``close`` for non-dividend instruments like indices/commodities/crypto).
        """

        def fetch_chunk(to_date: Optional[str]) -> pd.DataFrame:
            params: dict = {"symbol": symbol}
            if start:
                params["from"] = start
            if to_date:
                params["to"] = to_date
            payload = self._get(f"{FMP_STABLE}/historical-price-eod/full", params)
            if isinstance(payload, dict):
                msg = payload.get("Error Message") or payload.get("message") or str(payload)
                raise FMPError(f"{symbol}: {msg}")
            if not payload:
                return pd.DataFrame()
            df = pd.DataFrame(payload)
            if "adjClose" not in df.columns and "close" in df.columns:
                df["adjClose"] = df["close"]
            keep = [c for c in _KEEP_COLS if c in df.columns]
            df = df[keep].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

        return self._paginate_backward(fetch_chunk, start, end, label=symbol)

    # ------------------------------------------------------------------ #
    def get_treasury_rates(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        tenors: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch daily U.S. Treasury par yields from the ``treasury-rates`` endpoint.

        Used as the 10-Year-yield regime signal in place of the ``^TNX`` index,
        which is gated behind a higher FMP tier. Returns the requested ``tenors``
        (default ``year2`` + ``year10``) so the 10Y-2Y curve slope can be derived.
        """
        tenors = tenors or ["year2", "year10"]

        def fetch_chunk(to_date: Optional[str]) -> pd.DataFrame:
            params: dict = {}
            if start:
                params["from"] = start
            if to_date:
                params["to"] = to_date
            payload = self._get(f"{FMP_STABLE}/treasury-rates", params)
            if isinstance(payload, dict):
                msg = payload.get("Error Message") or payload.get("message") or str(payload)
                raise FMPError(f"treasury-rates: {msg}")
            if not payload:
                return pd.DataFrame()
            df = pd.DataFrame(payload)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            keep = [t for t in tenors if t in df.columns]
            df = df[keep].copy()
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

        return self._paginate_backward(fetch_chunk, start, end, label="treasury-rates")


if __name__ == "__main__":  # quick smoke test: python -m src.data.fmp_client
    client = FMPClient()
    sample = client.get_daily_history("^GSPC", start="2024-01-01", end="2024-02-01")
    print(sample.tail())
