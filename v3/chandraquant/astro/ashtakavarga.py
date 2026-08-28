"""Ashtakavarga - the classical numeric transit-strength system.

Ashtakavarga is Jyotisha's own quantitative model, and it is the most naturally
"quant" thing in the entire tradition. For each graha, eight contributors (the seven
grahas plus the Lagna) donate a benefic point - a *bindu* - to a specific set of
houses counted from wherever that contributor sits in the natal chart. Summing all
seven charts yields the Sarvashtakavarga: a 0-56 score for every rashi whose grand
total is always exactly 337.

The operative rule is transit-based and directly usable: a graha transiting a rashi
that holds many bindus delivers results; the same graha through a low-bindu rashi
does not. So SAV converts "where are the planets" into a per-index numeric strength
surface - and because the bindus are computed from the natal chart, the surface is
different for NIFTY, BANKNIFTY and CNXIT.

Classical thresholds: 30+ bindus is a strong rashi, below 25 is weak.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from ..config import ashtakavarga_cfg
from . import ayanamsa as ay
from .natal import natal_chart

BAV_GRAHAS = ["Surya", "Chandra", "Mangala", "Budha", "Guru", "Shukra", "Shani"]


@functools.lru_cache(maxsize=16)
def compute_charts(ticker_key: str) -> dict[str, np.ndarray]:
    """Bhinnashtakavarga (per-graha, 12 rashis) and Sarvashtakavarga for an index."""
    cfg = ashtakavarga_cfg()
    chart = natal_chart(ticker_key)

    # Natal rashi of each contributor, including the Lagna.
    ref_rashi = {g: int(ay.rashi_index(lon)) for g, lon in chart["grahas"].items()}
    ref_rashi["Lagna"] = chart["lagna_rashi"]

    charts: dict[str, np.ndarray] = {}
    for graha in BAV_GRAHAS:
        bindus = np.zeros(12, dtype=int)
        for contributor, houses in cfg["tables"][graha].items():
            base = ref_rashi[contributor]
            for h in houses:
                bindus[(base + int(h) - 1) % 12] += 1
        charts[graha] = bindus

    sav = np.zeros(12, dtype=int)
    for graha in BAV_GRAHAS:
        sav += charts[graha]
    charts["SAV"] = sav
    return charts


def compute(positions: pd.DataFrame, ticker_key: str) -> pd.DataFrame:
    """Transit Ashtakavarga features."""
    charts = compute_charts(ticker_key)
    sav = charts["SAV"]
    idx = positions.index
    cols: dict[str, np.ndarray] = {}

    weighted = np.zeros(len(idx), dtype=float)
    for graha in BAV_GRAHAS:
        rashi = ay.rashi_index(positions[f"{graha}_lon"].to_numpy())
        # SAV of the rashi this graha is currently transiting.
        sav_here = sav[rashi]
        cols[f"av_{graha}_sav"] = sav_here
        # The graha's own bhinnashtakavarga score in the rashi it occupies - the
        # classical "does this transit deliver" test.
        cols[f"av_{graha}_bav"] = charts[graha][rashi]
        cols[f"av_{graha}_strong"] = (charts[graha][rashi] >= 4).astype(int)
        weighted += sav_here

    cols["av_sav_mean"] = weighted / len(BAV_GRAHAS)
    # Chandra and Guru are the two transits classical texts weight most heavily.
    cols["av_chandra_sav"] = cols["av_Chandra_sav"]
    cols["av_guru_sav"] = cols["av_Guru_sav"]
    cols["av_shani_sav"] = cols["av_Shani_sav"]
    # Net: benefic transits through rich rashis minus malefic transits through poor ones.
    cols["av_benefic_support"] = (
        cols["av_Guru_sav"] + cols["av_Shukra_sav"] + cols["av_Chandra_sav"]
    ) / 3.0
    cols["av_malefic_support"] = (cols["av_Shani_sav"] + cols["av_Mangala_sav"]) / 2.0
    cols["av_net"] = cols["av_benefic_support"] - cols["av_malefic_support"]

    return pd.DataFrame(cols, index=idx)


def describe(ticker_key: str) -> pd.DataFrame:
    """The SAV table for display - 12 rashis with their bindu counts and band."""
    cfg = ashtakavarga_cfg()
    charts = compute_charts(ticker_key)
    from ..config import graha_cfg

    rashi_names = [r["name"] for r in graha_cfg()["rashis"]]
    bands = cfg["sav_bands"]

    def band_of(v: int) -> str:
        for b in bands:
            if v <= int(b["max"]):
                return b["label"]
        return bands[-1]["label"]

    return pd.DataFrame(
        {
            "rashi": rashi_names,
            "sav": charts["SAV"],
            "band": [band_of(int(v)) for v in charts["SAV"]],
        }
    )
