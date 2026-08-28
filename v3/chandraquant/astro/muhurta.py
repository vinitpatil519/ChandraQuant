"""Muhurta - electional timing: Lagna, Hora, Rahu Kaal and the auspiciousness score.

Muhurta is the branch of Jyotisha concerned with *when to act*, which makes it the
closest classical analogue to trade timing. Where the panchanga describes the day,
muhurta scores it: is this a moment to commit capital or to stand aside?

Three components are computed here, all keyed to the NSE session (09:15-15:30 IST):

  Lagna    the rising sign at the opening bell. It advances a full zodiac each day, so
           unlike every other slow feature it varies meaningfully day to day and gives
           the daily chart a distinct orientation.
  Hora     the planetary hour. Days divide into 24 horas ruled in Chaldean order, and
           the hora ruling the open colours the session.
  Rahu Kaal / Yamaghanta / Gulika - inauspicious eighth-parts of the daylight span,
           different for each weekday. Rahu Kaal in particular is widely observed by
           Indian retail traders, which makes it a rare case where the astrological
           belief could plausibly be self-fulfilling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import graha_cfg
from . import ayanamsa as ay
from .natal import ascendant_tropical

# NSE session, in hours after local midnight.
SESSION_OPEN_H = 9.25    # 09:15
SESSION_CLOSE_H = 15.5   # 15:30

# Approximate Mumbai daylight span. Sunrise/sunset vary by ~1h across the year; a
# sinusoidal model is well inside the resolution these eighth-part windows need.
MEAN_SUNRISE_H = 6.4
MEAN_SUNSET_H = 18.7
DAYLIGHT_SWING_H = 0.55


def _sunrise_sunset(day_of_year: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local sunrise / sunset in decimal hours for Mumbai."""
    phase = 2.0 * np.pi * (day_of_year - 172) / 365.25  # peak at summer solstice
    rise = MEAN_SUNRISE_H - DAYLIGHT_SWING_H * np.cos(phase)
    setg = MEAN_SUNSET_H + DAYLIGHT_SWING_H * np.cos(phase)
    return rise, setg


def compute(
    positions: pd.DataFrame,
    latitude: float = 18.9388,
    longitude: float = 72.8354,
    session_open_h: float = SESSION_OPEN_H,
) -> pd.DataFrame:
    """Muhurta features for the trading session."""
    cfg = graha_cfg()
    idx = positions.index
    n = len(idx)
    cols: dict[str, np.ndarray] = {}

    jd_tt = positions["jd_tt"].to_numpy()
    jd_ut = jd_tt - 69.2 / 86400.0
    ayan = positions["ayanamsa"].to_numpy()

    # --- Lagna at the opening bell ---------------------------------------------------
    asc_trop = ascendant_tropical(jd_ut, latitude, longitude)
    lagna = np.mod(asc_trop - ayan, 360.0)
    cols["lagna"] = lagna
    cols["lagna_rashi"] = ay.rashi_index(lagna)
    cols["lagna_nakshatra"] = ay.nakshatra_index(lagna)
    cols["lagna_degree"] = ay.rashi_degree(lagna)
    for name, arr in ay.cyclic_encode(lagna, 360.0, harmonics=2, prefix="lagna_").items():
        cols[name] = arr

    # Lagna lord and whether the rising sign is a movable / fixed / dual sign - the
    # classical read on whether the day's character will shift or hold.
    rashis = cfg["rashis"]
    quality = {r["index"]: r["quality"] for r in rashis}
    lord = {r["index"]: r["lord"] for r in rashis}
    cols["lagna_lord"] = np.array([lord[int(r)] for r in cols["lagna_rashi"]])
    lq = np.array([quality[int(r)] for r in cols["lagna_rashi"]])
    cols["lagna_is_chara"] = (lq == "chara").astype(int)
    cols["lagna_is_sthira"] = (lq == "sthira").astype(int)
    cols["lagna_is_dvisvabhava"] = (lq == "dvisvabhava").astype(int)

    # --- Hora (planetary hour) at the open ---------------------------------------------
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy().astype(float)
    rise, setg = _sunrise_sunset(doy)

    hora_order = list(cfg["hora_order"])          # Chaldean: Shani Guru Mangala Surya Shukra Budha Chandra
    vara_lords = {int(k): v for k, v in cfg["vara_lords"].items()}
    # The first hora of a day is ruled by that day's lord; subsequent horas step
    # backwards through the Chaldean order.
    hours_since_rise = np.maximum(session_open_h - rise, 0.0)
    hora_index = np.floor(hours_since_rise).astype(int)
    hora_lords = []
    for d, hi in zip(dow, hora_index):
        day_lord = vara_lords[int(d)]
        start = hora_order.index(day_lord)
        hora_lords.append(hora_order[(start + int(hi)) % 7])
    cols["hora_lord"] = np.array(hora_lords)
    cols["hora_index"] = hora_index
    benefic_horas = {"Guru", "Shukra", "Budha", "Chandra"}
    cols["hora_is_benefic"] = np.array(
        [1 if h in benefic_horas else 0 for h in hora_lords]
    )

    # --- Rahu Kaal / Yamaghanta / Gulika ------------------------------------------------
    daylight = setg - rise
    part = daylight / 8.0
    for label, key in (
        ("rahu_kaal", "rahu_kaal_part"),
        ("yamaganda", "yamaganda_part"),
        ("gulika", "gulika_part"),
    ):
        part_map = {int(k): int(v) for k, v in cfg[key].items()}
        part_no = np.array([part_map[int(d)] for d in dow])
        start_h = rise + (part_no - 1) * part
        end_h = start_h + part
        cols[f"{label}_start"] = start_h
        cols[f"{label}_end"] = end_h
        # Does the window overlap the trading session at all, and how much of it?
        overlap = np.maximum(
            0.0, np.minimum(end_h, SESSION_CLOSE_H) - np.maximum(start_h, session_open_h)
        )
        cols[f"{label}_in_session"] = (overlap > 0).astype(int)
        cols[f"{label}_session_fraction"] = overlap / (SESSION_CLOSE_H - session_open_h)
        # Does it land on the open specifically - the most watched case?
        cols[f"{label}_at_open"] = (
            (start_h <= session_open_h) & (end_h > session_open_h)
        ).astype(int)

    # --- Abhijit muhurta: the auspicious 8th muhurta straddling local noon --------------
    midday = (rise + setg) / 2.0
    abhijit_half = daylight / 30.0
    cols["abhijit_start"] = midday - abhijit_half
    cols["abhijit_end"] = midday + abhijit_half
    cols["abhijit_in_session"] = (
        (cols["abhijit_start"] < SESSION_CLOSE_H) & (cols["abhijit_end"] > session_open_h)
    ).astype(int)

    cols["daylight_hours"] = daylight
    cols["sunrise_h"] = rise
    cols["sunset_h"] = setg

    return pd.DataFrame(cols, index=idx)
