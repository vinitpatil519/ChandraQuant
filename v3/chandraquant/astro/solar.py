"""Surya cycles - Sankranti ingress, Ayana, Ritu and the solar month.

Where Chandra supplies the fast oscillation, Surya supplies the annual frame. In
mundane astrology Surya is the sovereign - state authority, policy, and the structural
"year" of the market. The Sankranti (the Sun's ingress into a new sidereal rashi) is
the single most widely observed astronomical event in the Indian calendar; Makar
Sankranti, when Surya enters Makara, is a national festival and a classical marker of
the turn toward growth (Uttarayana, the northward course).

These features are slow, which is exactly the point: they give the model an annual
seasonal frame that is orthogonal to price and known decades in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ayanamsa as ay

# The six ritus, each spanning two sidereal rashis starting from Mesha.
RITU_NAMES = [
    ("Vasanta", "spring - fresh growth, risk appetite returns"),
    ("Grishma", "summer - heat and exhaustion, momentum overextends"),
    ("Varsha", "monsoon - turbulence and reversal, flows disrupted"),
    ("Sharad", "autumn - clarity and harvest, the festive bid"),
    ("Hemanta", "pre-winter - consolidation and stock-taking"),
    ("Shishira", "winter - contraction, minimum activity"),
]

# Solar (sidereal) months, named for the rashi the Sun occupies.
MASA_NAMES = [
    "Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
    "Tula", "Vrischika", "Dhanu", "Makara", "Kumbha", "Meena",
]

MAKARA_RASHI = 9   # Makar Sankranti - start of Uttarayana in the sidereal frame
MESHA_RASHI = 0    # Mesha Sankranti - the Vedic solar new year
KARKA_RASHI = 3    # Karka Sankranti - start of Dakshinayana


def _days_to_next_change(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """For each row, days until the next change in a step-valued series."""
    n = len(values)
    out = np.full(n, np.nan)
    change_idx = np.flatnonzero(np.diff(values) != 0) + 1
    if len(change_idx) == 0:
        return out
    day_num = (dates - dates[0]).days.to_numpy().astype(float)
    ptr = 0
    for i in range(n):
        while ptr < len(change_idx) and change_idx[ptr] <= i:
            ptr += 1
        if ptr < len(change_idx):
            out[i] = day_num[change_idx[ptr]] - day_num[i]
    return out


def _days_since_change(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    change_idx = np.flatnonzero(np.diff(values) != 0) + 1
    if len(change_idx) == 0:
        return out
    day_num = (dates - dates[0]).days.to_numpy().astype(float)
    ptr = -1
    for i in range(n):
        while ptr + 1 < len(change_idx) and change_idx[ptr + 1] <= i:
            ptr += 1
        if ptr >= 0:
            out[i] = day_num[i] - day_num[change_idx[ptr]]
    return out


def compute(positions: pd.DataFrame) -> pd.DataFrame:
    """Solar cycle features from an ephemeris frame."""
    sun = positions["Surya_lon"].to_numpy()
    sun_trop = positions["Surya_trop"].to_numpy()
    idx = positions.index
    df = pd.DataFrame(index=idx)

    rashi = ay.rashi_index(sun)
    df["surya_rashi"] = rashi
    df["surya_rashi_degree"] = ay.rashi_degree(sun)
    df["surya_nakshatra"] = ay.nakshatra_index(sun)

    # --- Sankranti (solar ingress) ---------------------------------------------------
    df["days_to_sankranti"] = _days_to_next_change(rashi, idx)
    df["days_since_sankranti"] = _days_since_change(rashi, idx)
    # An ingress window: the classical Punya Kala around the exact crossing.
    df["sankranti_window"] = (
        (df["days_to_sankranti"] <= 2) | (df["days_since_sankranti"] <= 2)
    ).astype(int)
    df["is_makar_sankranti"] = (
        (rashi == MAKARA_RASHI) & (df["days_since_sankranti"] <= 1)
    ).astype(int)
    df["is_mesha_sankranti"] = (
        (rashi == MESHA_RASHI) & (df["days_since_sankranti"] <= 1)
    ).astype(int)

    # --- Ayana (solar course) ---------------------------------------------------------
    # Sidereal Uttarayana: Makara through Mithuna. The northward course is classically
    # the auspicious half of the year for undertakings (Shubha Kala).
    df["is_uttarayana"] = (((rashi - MAKARA_RASHI) % 12) < 6).astype(int)
    # Tropical declination - the true astronomical seasonal driver.
    eps = np.radians(23.4392911)
    df["surya_declination"] = np.degrees(
        np.arcsin(np.sin(eps) * np.sin(np.radians(sun_trop)))
    )

    # --- Ritu (season) and Masa (solar month) ------------------------------------------
    ritu = (rashi // 2) % 6
    df["ritu_index"] = ritu
    df["ritu_name"] = [RITU_NAMES[r][0] for r in ritu]
    df["masa_index"] = rashi
    df["masa_name"] = [MASA_NAMES[r] for r in rashi]

    # --- Cyclic embeddings ---------------------------------------------------------------
    for values, period, prefix, harm in (
        (sun, 360.0, "surya_", 3),
        (rashi + df["surya_rashi_degree"].to_numpy() / 30.0, 12.0, "masa_", 2),
        (ritu, 6.0, "ritu_", 1),
    ):
        for name, arr in ay.cyclic_encode(values, period, harmonics=harm, prefix=prefix).items():
            df[name] = arr

    return df
