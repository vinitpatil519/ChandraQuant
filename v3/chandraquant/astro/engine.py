"""The Jyotisha engine - assembles every astro module into one per-ticker matrix.

Order matters: events needs lunar, shadbala needs panchanga + grahas + aspects +
the daily Lagna from muhurta, and composites needs all of them plus the natal-relative
frames. This module owns that wiring so nothing else has to.

Everything produced here is a deterministic function of (date, ticker natal chart).
No market data is touched, so the whole matrix can be computed years into the future -
which is what makes the forward calendar in the dashboard an honest forecast rather
than a backfit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, get_ticker
from . import (
    ashtakavarga,
    aspects,
    composites,
    dasha,
    ephemeris,
    events,
    grahas,
    lunar,
    muhurta,
    natal,
    panchanga,
    shadbala,
    solar,
)

# Columns that are labels rather than numbers. Kept for display and the narrative
# engine, excluded from the model matrix by features/astro_bridge.py.
CATEGORICAL_SUFFIXES = (
    "_name", "_lord", "_gana", "_activity", "_deity", "_archetype",
    "_five_fold", "ritu_name", "masa_name",
)


def is_categorical(col: str) -> bool:
    return any(col.endswith(s) or col == s for s in CATEGORICAL_SUFFIXES)


def _cache_path(ticker_key: str, dates: pd.DatetimeIndex) -> Path:
    payload = f"{ticker_key}|{dates[0]}|{dates[-1]}|{len(dates)}"
    return CACHE_DIR / f"astro_{ticker_key}_{hashlib.md5(payload.encode()).hexdigest()[:12]}.parquet"


def build(
    dates,
    ticker_key: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Full Jyotisha feature matrix for one index over `dates`."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize().unique().sort_values()
    cache = _cache_path(ticker_key, dates)
    if use_cache and cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) == len(dates):
            return cached

    entry = get_ticker(ticker_key)
    pos = ephemeris.graha_positions(
        dates, local_time=entry["transit_time"], tz=entry["timezone"]
    )

    pan = panchanga.compute(pos)
    lun = lunar.compute(pos)
    sol = solar.compute(pos)
    grh = grahas.compute(pos)
    asp = aspects.compute(pos)
    evt = events.compute(pos, lun)
    nat = natal.compute(pos, ticker_key)
    dsh = dasha.compute(pos, ticker_key)
    av = ashtakavarga.compute(pos, ticker_key)
    muh = muhurta.compute(
        pos,
        latitude=entry["latitude"],
        longitude=entry["longitude"],
    )
    sbl = shadbala.compute(pos, muh["lagna"].to_numpy(), pan, grh, asp)
    cmp_ = composites.compute(pan, lun, sol, grh, asp, evt, nat, dsh, av, sbl)

    frames = {
        "pan": pan, "lun": lun, "sol": sol, "grh": grh, "asp": asp, "evt": evt,
        "nat": nat, "dsh": dsh, "av": av, "muh": muh, "sbl": sbl, "cmp": cmp_,
    }
    # Drop the raw ephemeris longitudes we already re-express as cyclic encodings,
    # but keep the sidereal longitudes themselves - they are genuinely informative.
    keep_pos = pos[[c for c in pos.columns if c.endswith("_lon") or c == "ayanamsa"]]

    out = pd.concat([keep_pos] + list(frames.values()), axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index.name = "date"

    if use_cache:
        out.to_parquet(cache)
    return out


def numeric_matrix(astro: pd.DataFrame) -> pd.DataFrame:
    """Model-ready view: numeric columns only, categoricals dropped."""
    cols = [c for c in astro.columns if not is_categorical(c)]
    num = astro[cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN or constant - they carry no information.
    num = num.loc[:, num.notna().any()]
    nunique = num.nunique(dropna=True)
    return num.loc[:, nunique > 1]


def summary(ticker_key: str, when) -> dict:
    """Everything the dashboard and narrative engine need for a single date."""
    ts = pd.Timestamp(when).normalize()
    window = pd.date_range(ts - pd.Timedelta(days=400), ts, freq="D")
    astro = build(window, ticker_key)
    row = astro.loc[ts]
    chart = natal.natal_chart(ticker_key)
    return {
        "date": ts,
        "ticker": ticker_key,
        "chart": chart,
        "row": row,
        "dasha": dasha.describe(ticker_key, ts),
        "sav": ashtakavarga.describe(ticker_key),
    }
