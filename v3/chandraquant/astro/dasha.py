"""Vimshottari Dasha - the 120-year planetary period chain.

Vimshottari is the dominant predictive framework in Jyotisha, and it is the single
best macro-regime labeller in this project. Unlike a transit, which lasts days, a
Mahadasha runs 6-20 years: it partitions a quarter-century of index history into a
handful of interpretable eras, each ruled by one graha with a distinct market
character. "BANKNIFTY is in Shani Mahadasha / Budha Antardasha" is a *structural*
statement about the regime, and it is the sentence the narrative engine leads with.

Mechanics. The chain is seeded by the natal Moon's nakshatra: its Vimshottari lord
owns the first Mahadasha, and the portion of that dasha already elapsed at birth is
the fraction of the nakshatra the Moon had traversed. Thereafter the nine lords run
in fixed order - Ketu 7, Shukra 20, Surya 6, Chandra 10, Mangala 7, Rahu 18, Guru 16,
Shani 19, Budha 17 - summing to 120 years.

Each Mahadasha subdivides into nine Antardashas in the same order, starting from the
Mahadasha lord itself, proportioned by the same year-weights; each Antardasha
subdivides again into Pratyantardashas. So the chain gives three nested regime labels
at three different timescales from a single natal fact.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from ..config import nakshatra_cfg
from . import ayanamsa as ay
from .natal import natal_chart

# Vimshottari uses the sidereal solar year.
DAYS_PER_YEAR = 365.25

# Market character of each dasha lord, used by the narrative engine.
LORD_CHARACTER = {
    "Ketu": ("detachment and severance", -0.35, "abrupt dislocations, thin conviction, sudden exits"),
    "Shukra": ("valuation and appetite", 0.55, "premium multiples, risk appetite, sustained bid"),
    "Surya": ("authority and policy", 0.20, "state-driven, headline-sensitive, top-down"),
    "Chandra": ("sentiment and liquidity", 0.35, "flow-driven, emotional, fast rotation"),
    "Mangala": ("aggression and volatility", -0.20, "sharp two-way moves, breakouts and breakdowns"),
    "Rahu": ("leverage and speculation", 0.30, "manias, derivatives excess, foreign flows"),
    "Guru": ("expansion and credit", 0.75, "the classical bull era - credit and re-rating"),
    "Shani": ("contraction and discipline", -0.45, "grinding, patient, de-rating - accumulation over momentum"),
    "Budha": ("commerce and information", 0.40, "efficient price discovery, news-reactive, tactical"),
}


@functools.lru_cache(maxsize=16)
def _vimshottari_params() -> tuple[tuple[str, ...], dict[str, int], float]:
    cfg = nakshatra_cfg()["vimshottari"]
    return tuple(cfg["order"]), dict(cfg["years"]), float(cfg["total_years"])


def _sequence_from(lord: str) -> list[str]:
    """The nine lords in Vimshottari order, rotated to begin at `lord`."""
    order, _, _ = _vimshottari_params()
    i = order.index(lord)
    return list(order[i:]) + list(order[:i])


@functools.lru_cache(maxsize=16)
def build_chain(ticker_key: str, horizon_years: float = 140.0) -> pd.DataFrame:
    """Build the full Mahadasha / Antardasha / Pratyantardasha chain for an index.

    Returns one row per Pratyantardasha with start/end timestamps and the three lords.
    """
    order, years, total = _vimshottari_params()
    chart = natal_chart(ticker_key)
    birth = pd.Timestamp(chart["natal_utc"]).tz_localize(None)

    seed_lord = chart["moon_nakshatra_lord"]
    elapsed_fraction = chart["moon_nak_fraction"]

    # The first Mahadasha is already partly spent at birth; wind the clock back so the
    # chain starts at that lord's true (notional) commencement.
    md_start = birth - pd.Timedelta(days=years[seed_lord] * elapsed_fraction * DAYS_PER_YEAR)

    rows = []
    cursor_md = md_start
    md_sequence = _sequence_from(seed_lord)
    limit = birth + pd.Timedelta(days=horizon_years * DAYS_PER_YEAR)
    cycle = 0
    while cursor_md < limit and cycle < 4:
        for md_lord in md_sequence:
            md_days = years[md_lord] * DAYS_PER_YEAR
            md_end = cursor_md + pd.Timedelta(days=md_days)

            cursor_ad = cursor_md
            for ad_lord in _sequence_from(md_lord):
                ad_days = md_days * years[ad_lord] / total
                ad_end = cursor_ad + pd.Timedelta(days=ad_days)

                cursor_pd = cursor_ad
                for pd_lord in _sequence_from(ad_lord):
                    pd_days = ad_days * years[pd_lord] / total
                    pd_end = cursor_pd + pd.Timedelta(days=pd_days)
                    rows.append(
                        {
                            "start": cursor_pd,
                            "end": pd_end,
                            "md": md_lord,
                            "ad": ad_lord,
                            "pd": pd_lord,
                            "md_start": cursor_md,
                            "md_end": md_end,
                            "ad_start": cursor_ad,
                            "ad_end": ad_end,
                        }
                    )
                    cursor_pd = pd_end
                cursor_ad = ad_end
            cursor_md = md_end
            if cursor_md >= limit:
                break
        md_sequence = list(order)
        cycle += 1

    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def compute(positions: pd.DataFrame, ticker_key: str) -> pd.DataFrame:
    """Dasha features for each trading day."""
    chain = build_chain(ticker_key)
    idx = positions.index
    n = len(idx)

    starts = chain["start"].to_numpy()
    pos = np.searchsorted(starts, idx.to_numpy(), side="right") - 1
    pos = np.clip(pos, 0, len(chain) - 1)
    sel = chain.iloc[pos].reset_index(drop=True)

    cols: dict[str, np.ndarray] = {}
    order, years, _ = _vimshottari_params()
    lord_to_int = {lord: i for i, lord in enumerate(order)}

    for level in ("md", "ad", "pd"):
        lords = sel[level].to_numpy()
        cols[f"dasha_{level}_lord"] = lords
        cols[f"dasha_{level}_idx"] = np.array([lord_to_int[l] for l in lords])
        cols[f"dasha_{level}_bias"] = np.array([LORD_CHARACTER[l][1] for l in lords])
        for name, arr in ay.cyclic_encode(
            cols[f"dasha_{level}_idx"], 9.0, harmonics=1, prefix=f"dasha_{level}_"
        ).items():
            cols[name] = arr

    # Progress through each level, plus days to the next change - a dasha sandhi
    # (junction) is classically a moment of regime instability.
    days = idx.to_numpy().astype("datetime64[ns]")
    for level, sfx in (("md", "md"), ("ad", "ad")):
        start = sel[f"{level}_start"].to_numpy().astype("datetime64[ns]")
        end = sel[f"{level}_end"].to_numpy().astype("datetime64[ns]")
        span = (end - start) / np.timedelta64(1, "D")
        elapsed = (days - start) / np.timedelta64(1, "D")
        cols[f"dasha_{sfx}_progress"] = np.clip(elapsed / np.maximum(span, 1.0), 0.0, 1.0)
        cols[f"dasha_{sfx}_days_remaining"] = (end - days) / np.timedelta64(1, "D")
        # Sandhi: the last / first 5% of a period.
        prog = cols[f"dasha_{sfx}_progress"]
        cols[f"dasha_{sfx}_sandhi"] = ((prog < 0.05) | (prog > 0.95)).astype(int)

    pd_start = sel["start"].to_numpy().astype("datetime64[ns]")
    pd_end = sel["end"].to_numpy().astype("datetime64[ns]")
    pd_span = (pd_end - pd_start) / np.timedelta64(1, "D")
    cols["dasha_pd_progress"] = np.clip(
        ((days - pd_start) / np.timedelta64(1, "D")) / np.maximum(pd_span, 1.0), 0.0, 1.0
    )

    # Combined regime bias: the Mahadasha sets the era, the Antardasha modulates it,
    # the Pratyantardasha only colours the immediate weeks.
    cols["dasha_combined_bias"] = (
        0.50 * cols["dasha_md_bias"]
        + 0.35 * cols["dasha_ad_bias"]
        + 0.15 * cols["dasha_pd_bias"]
    )

    # Lord-pair interaction: does the Antardasha lord support or fight the Mahadasha lord?
    cols["dasha_md_ad_same"] = (sel["md"].to_numpy() == sel["ad"].to_numpy()).astype(int)

    df = pd.DataFrame(cols, index=idx)
    return df


def describe(ticker_key: str, when) -> dict:
    """Human-readable dasha state for one moment - used by the narrative engine."""
    chain = build_chain(ticker_key)
    ts = pd.Timestamp(when)
    row = chain[(chain["start"] <= ts) & (chain["end"] > ts)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "md": r["md"],
        "ad": r["ad"],
        "pd": r["pd"],
        "md_character": LORD_CHARACTER[r["md"]][0],
        "md_reading": LORD_CHARACTER[r["md"]][2],
        "ad_character": LORD_CHARACTER[r["ad"]][0],
        "ad_reading": LORD_CHARACTER[r["ad"]][2],
        "md_start": r["md_start"],
        "md_end": r["md_end"],
        "ad_start": r["ad_start"],
        "ad_end": r["ad_end"],
        "md_years_remaining": (r["md_end"] - ts).days / DAYS_PER_YEAR,
        "ad_days_remaining": (r["ad_end"] - ts).days,
    }
