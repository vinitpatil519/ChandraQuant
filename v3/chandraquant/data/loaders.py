"""Market data with a snapshot fallback, so a demo never dies on a bad network.

Policy: try a live Yahoo refresh with a hard timeout; on any failure - no network,
rate limit, schema change - fall back silently to the committed snapshot in
``data/snapshot/``. The caller is told which happened via ``DataStatus`` so the UI can
show a ``live`` or ``cached`` badge honestly rather than pretending.

Verified 2026-08-28: Yahoo serves ^NSEI, ^NSEBANK and ^CNXIT only from 2007-09-17,
NOT from 1996 as the v1 documents claim. That still covers the GFC, COVID and
Russia-Ukraine crisis windows. See ``backfill.py`` for the extension attempt.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..config import RAW_DIR, SNAPSHOT_DIR, TICKER_KEYS, get_ticker

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_START = "1996-01-01"
FETCH_TIMEOUT_S = 8.0


@dataclass
class DataStatus:
    """Where the data actually came from - surfaced in the UI, never guessed."""

    source: str          # "live" | "cache" | "snapshot"
    as_of: pd.Timestamp
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def is_live(self) -> bool:
        return self.source == "live"

    def badge(self) -> str:
        dot = "*" if self.is_live else "o"
        label = {"live": "live", "cache": "cached", "snapshot": "snapshot"}[self.source]
        return f"{dot} {label} ({self.end.date()})"


def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns for a single ticker in recent versions."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        elif set(OHLCV) & set(lvl0):
            df.columns = lvl0
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns=str.title)
    keep = [c for c in OHLCV if c in df.columns]
    df = df[keep].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df


def _fetch_yahoo(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            symbol, start=start, progress=False, auto_adjust=False, threads=False
        )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"Yahoo returned no rows for {symbol}")
    return _flatten(raw, symbol)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    # Index series occasionally carry zero-volume placeholder rows; keep them but make
    # the zeros explicit so volume features do not silently divide by zero.
    if "Volume" in df:
        df["Volume"] = df["Volume"].fillna(0.0)
    return df


def raw_path(key: str) -> str:
    return str(RAW_DIR / f"{key}.parquet")


def snapshot_path(key: str) -> str:
    return str(SNAPSHOT_DIR / f"{key}.parquet")


def load_ohlcv(
    ticker_key: str,
    refresh: bool = True,
    start: str = DEFAULT_START,
) -> tuple[pd.DataFrame, DataStatus]:
    """Load daily OHLCV for one index, preferring live data but never requiring it."""
    entry = get_ticker(ticker_key)
    key, symbol = entry["key"], entry["yahoo"]

    df = None
    source = "snapshot"

    if refresh:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_fetch_yahoo, symbol, start)
                df = _clean(future.result(timeout=FETCH_TIMEOUT_S))
            source = "live"
            df.to_parquet(raw_path(key))
        except (FutureTimeout, Exception):
            df = None

    if df is None:
        from pathlib import Path

        for path, tag in ((raw_path(key), "cache"), (snapshot_path(key), "snapshot")):
            if Path(path).exists():
                df = _clean(pd.read_parquet(path))
                source = tag
                break

    if df is None or df.empty:
        raise FileNotFoundError(
            f"No data for {key}: live fetch failed and no snapshot exists. "
            f"Run `python scripts/refresh_data.py --all` once while online."
        )

    status = DataStatus(
        source=source,
        as_of=pd.Timestamp(date.today()),
        rows=len(df),
        start=df.index[0],
        end=df.index[-1],
    )
    return df, status


def write_snapshot(ticker_key: str, df: pd.DataFrame) -> str:
    """Persist the committed offline fallback."""
    entry = get_ticker(ticker_key)
    path = snapshot_path(entry["key"])
    df.to_parquet(path)
    return path


def load_all(refresh: bool = True) -> dict[str, tuple[pd.DataFrame, DataStatus]]:
    return {k: load_ohlcv(k, refresh=refresh) for k in TICKER_KEYS}
