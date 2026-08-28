"""Panchanga - the five limbs of Vedic time: Tithi, Vara, Nakshatra, Yoga, Karana.

Each limb is a deterministic function of the Sun and Moon's sidereal longitudes, so
every column produced here is known arbitrarily far in advance. That is the single
genuinely unusual property of this feature family: unlike every technical indicator,
it does not have to wait for the market to print.

Classical significance, condensed:
  Tithi     lunar day, 12 deg of Chandra-Surya elongation. The Rikta tithis (4th, 9th,
            14th of each paksha) are barred for commerce in muhurta texts.
  Vara      weekday and its ruling graha.
  Nakshatra the Moon's mansion, 13 deg 20'. Carries gana, lord and an activity class
            that maps directly onto market character (trending / range / volatile).
  Yoga      the 27 nitya yogas from the SUM of the luminaries' longitudes. Nine are
            classically inauspicious; Vyatipata and Vaidhriti are the worst.
  Karana    half a tithi. Vishti (Bhadra) is the classical "transact nothing" window
            and is used as a hard gate on new long entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import karana_cfg, nakshatra_cfg, tithi_cfg, yoga_cfg, graha_cfg
from . import ayanamsa as ay


def _lookup(entries: list[dict], key: str) -> dict:
    return {e["index"]: e[key] for e in entries}


def _nak_table() -> list[dict]:
    return nakshatra_cfg()["nakshatras"]


def _karana_for_slot(slot: np.ndarray) -> np.ndarray:
    """Map karana slot 0..59 onto the 11 karana identities.

    Surya Siddhanta sequence: Kimstughna occupies slot 0; the seven chara karanas
    repeat eight times across slots 1..56; Shakuni, Chatushpada and Naga close the
    lunar month at slots 57, 58, 59.
    """
    slot = np.asarray(slot, dtype=int)
    out = np.empty_like(slot)
    out[:] = -1
    is_first = slot == 0
    is_last = slot >= 57
    is_chara = ~is_first & ~is_last

    out[is_first] = 10                       # Kimstughna
    out[is_chara] = (slot[is_chara] - 1) % 7  # Bava..Vishti -> 0..6
    out[is_last] = 7 + (slot[is_last] - 57)   # Shakuni, Chatushpada, Naga -> 7, 8, 9
    return out


def compute(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute the five limbs from an ephemeris frame (see ephemeris.graha_positions)."""
    tcfg, ncfg, ycfg, kcfg, gcfg = tithi_cfg(), nakshatra_cfg(), yoga_cfg(), karana_cfg(), graha_cfg()
    sun = positions["Surya_lon"].to_numpy()
    moon = positions["Chandra_lon"].to_numpy()
    idx = positions.index

    df = pd.DataFrame(index=idx)

    # --- Tithi ---------------------------------------------------------------------
    elong = np.mod(moon - sun, 360.0)
    df["elongation"] = elong
    tithi_exact = elong / ay.DEG_PER_TITHI
    tithi_i = tithi_exact.astype(int)
    df["tithi_index"] = tithi_i
    df["tithi_frac"] = tithi_exact - tithi_i            # progress through the tithi
    df["tithi_num"] = (tithi_i % 15) + 1                # 1..15 within the paksha
    df["paksha_shukla"] = (tithi_i < 15).astype(int)

    t_entries = tcfg["tithis"]
    t_name = _lookup(t_entries, "display")
    t_five = _lookup(t_entries, "five_fold")
    t_bias = _lookup(t_entries, "market_bias")
    df["tithi_name"] = [t_name[i] for i in tithi_i]
    df["tithi_five_fold"] = [t_five[i] for i in tithi_i]
    df["tithi_bias"] = [t_bias[i] for i in tithi_i]
    df["tithi_is_rikta"] = (df["tithi_five_fold"] == "Rikta").astype(int)
    df["tithi_five_fold_quality"] = df["tithi_five_fold"].map(
        {k: v["quality"] for k, v in tcfg["five_fold_classes"].items()}
    )
    df["paksha_bias"] = np.where(
        df["paksha_shukla"] == 1,
        tcfg["paksha"]["Shukla"]["bias"],
        tcfg["paksha"]["Krishna"]["bias"],
    )
    df["is_purnima"] = (tithi_i == 14).astype(int)
    df["is_amavasya"] = (tithi_i == 29).astype(int)

    # Paksha-bala: Chandra's strength waxes to full at Purnima and vanishes at Amavasya.
    df["paksha_bala"] = (1.0 - np.cos(np.radians(elong))) / 2.0

    # --- Vara ----------------------------------------------------------------------
    dow = idx.dayofweek.to_numpy()
    df["vara"] = dow
    vara_lords = {int(k): v for k, v in gcfg["vara_lords"].items()}
    df["vara_lord"] = [vara_lords[int(d)] for d in dow]

    # --- Nakshatra -----------------------------------------------------------------
    nak_i = ay.nakshatra_index(moon)
    df["nakshatra_index"] = nak_i
    df["nakshatra_frac"] = ay.nakshatra_fraction(moon)
    df["pada"] = ay.pada_index(moon)
    df["pada_global"] = ay.pada_global(moon)

    n_entries = _nak_table()
    for col, key in (
        ("nakshatra_name", "display"),
        ("nakshatra_lord", "lord"),
        ("nakshatra_gana", "gana"),
        ("nakshatra_activity", "activity"),
        ("nakshatra_deity", "deity"),
        ("nakshatra_archetype", "archetype"),
    ):
        table = _lookup(n_entries, key)
        df[col] = [table[i] for i in nak_i]
    nb = _lookup(n_entries, "market_bias")
    df["nakshatra_bias"] = [nb[i] for i in nak_i]

    gana_score = {"Deva": 1.0, "Manushya": 0.0, "Rakshasa": -1.0}
    df["gana_score"] = df["nakshatra_gana"].map(gana_score)
    for cls in ("chara", "sthira", "ugra", "mridu", "kshipra", "tikshna", "mishra"):
        df[f"nak_is_{cls}"] = (df["nakshatra_activity"] == cls).astype(int)

    # --- Yoga ----------------------------------------------------------------------
    yoga_exact = np.mod(sun + moon, 360.0) / ay.DEG_PER_YOGA
    yoga_i = yoga_exact.astype(int)
    df["yoga_index"] = yoga_i
    df["yoga_frac"] = yoga_exact - yoga_i
    y_entries = ycfg["yogas"]
    y_name, y_mal, y_bias = (
        _lookup(y_entries, "display"),
        _lookup(y_entries, "malefic"),
        _lookup(y_entries, "market_bias"),
    )
    df["yoga_name"] = [y_name[i] for i in yoga_i]
    df["yoga_is_malefic"] = [int(y_mal[i]) for i in yoga_i]
    df["yoga_bias"] = [y_bias[i] for i in yoga_i]

    # --- Karana --------------------------------------------------------------------
    slot = (elong / ay.DEG_PER_KARANA).astype(int)
    kar_i = _karana_for_slot(slot)
    df["karana_slot"] = slot
    df["karana_index"] = kar_i
    k_entries = kcfg["karanas"]
    k_name, k_vishti, k_bias = (
        _lookup(k_entries, "display"),
        _lookup(k_entries, "vishti"),
        _lookup(k_entries, "market_bias"),
    )
    df["karana_name"] = [k_name[i] for i in kar_i]
    df["karana_is_vishti"] = [int(k_vishti[i]) for i in kar_i]
    df["karana_bias"] = [k_bias[i] for i in kar_i]

    # --- Cyclic embeddings ----------------------------------------------------------
    # Angular variables wrap; projecting onto the unit circle keeps 359 deg adjacent
    # to 1 deg, and the second/third harmonics expose sub-cycle structure to the trees.
    for values, period, prefix, harm in (
        (elong, 360.0, "tithi_", 3),
        (moon, 360.0, "moon_", 3),
        (sun, 360.0, "sun_", 2),
        (nak_i + df["nakshatra_frac"].to_numpy(), 27.0, "nak_", 2),
        (yoga_i + df["yoga_frac"].to_numpy(), 27.0, "yoga_", 2),
        (slot, 60.0, "karana_", 1),
        (dow, 7.0, "vara_", 1),
    ):
        for name, arr in ay.cyclic_encode(values, period, harmonics=harm, prefix=prefix).items():
            df[name] = arr

    # --- Composite panchanga auspiciousness (Muhurta Shuddhi) ------------------------
    df["panchanga_score"] = (
        0.25 * df["tithi_bias"]
        + 0.30 * df["nakshatra_bias"]
        + 0.20 * df["yoga_bias"]
        + 0.15 * df["karana_bias"]
        + 0.10 * df["paksha_bias"]
    )
    return df
