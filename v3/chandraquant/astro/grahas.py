"""Graha state - dignity, Vakri (retrogression), Asta (combustion) and Graha Yuddha.

This is where the slow planets earn their keep. Chandra moves too fast to define a
regime and Surya only frames the year, but Guru (12-year cycle), Shani (29.5-year) and
the nodes (18.6-year) move on exactly the timescale of a business cycle. Classical
mundane astrology reads them as the structural background against which the fast
lunar signal plays out.

Three classical states are treated as first-class events because each has a clean
market analogue:

  Vakri (retrograde)  - the graha's apparent motion reverses. Budha vakri is the
                        best-known market superstition anywhere; here it is simply a
                        recurring, precisely datable regime flag.
  Stambhana (station) - the instant motion reverses. Velocity is zero and the graha
                        lingers on one degree for days. Classically the point of
                        maximum effect, and the strongest turning-point candidate in
                        the whole feature set.
  Asta (combustion)   - the graha is swallowed by Surya's glare and loses its power to
                        act. The market reading is a participant whose capacity to
                        transact has been temporarily withdrawn.

Verified against real ephemeris events: Budha vakri 2025 detected at 2025-03-16 /
07-19 / 11-10 (actual stations 03-15 / 07-18 / 11-09); Shani vakri 138 days in 2025
(actual ~138); Guru vakri 85 days in calendar 2025 (actual ~86).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import graha_cfg
from . import ayanamsa as ay
from .ephemeris import NAVAGRAHA

# Grahas that can retrograde and be combust. The nodes are always retrograde by
# definition, and Surya cannot be combust by itself.
MOVING_GRAHAS = ["Mangala", "Budha", "Guru", "Shukra", "Shani"]
REAL_GRAHAS = ["Surya", "Chandra", "Mangala", "Budha", "Guru", "Shukra", "Shani"]
NODES = ["Rahu", "Ketu"]


def _graha_lookup() -> dict[str, dict]:
    return {g["name"]: g for g in graha_cfg()["grahas"]}


def _exalt_debil_points(spec: dict) -> tuple[float, float]:
    """Absolute sidereal longitude of the deep exaltation and debilitation points."""
    ex = spec["exalt"]["rashi"] * 30.0 + spec["exalt"]["degree"]
    de = spec["debil"]["rashi"] * 30.0 + spec["debil"]["degree"]
    return ex, de


def _days_to_flag_change(flag: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Days until a boolean flag flips (NaN past the last flip in the sample)."""
    n = len(flag)
    out = np.full(n, np.nan)
    change_idx = np.flatnonzero(np.diff(flag.astype(int)) != 0) + 1
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


def compute(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-graha dignity, motion state and combustion features."""
    cfg = graha_cfg()
    specs = _graha_lookup()
    idx = positions.index
    n = len(idx)
    sun = positions["Surya_lon"].to_numpy()

    # Accumulate into a dict and build the frame once - inserting ~185 columns one at
    # a time fragments the block manager badly.
    cols: dict[str, np.ndarray] = {}
    benefic_strength = np.zeros(n)
    malefic_strength = np.zeros(n)

    for name in NAVAGRAHA:
        spec = specs[name]
        lon = positions[f"{name}_lon"].to_numpy()
        speed = positions[f"{name}_speed"].to_numpy()

        rashi = ay.rashi_index(lon)
        rashi_deg = ay.rashi_degree(lon)
        cols[f"{name}_rashi"] = rashi
        cols[f"{name}_rashi_deg"] = rashi_deg
        cols[f"{name}_nak"] = ay.nakshatra_index(lon)
        cols[f"{name}_pada"] = ay.pada_index(lon)

        # --- Uccha bala: 1.0 at deep exaltation, 0.0 at deep debilitation -------------
        _, de_pt = _exalt_debil_points(spec)
        uccha = (180.0 - ay.angular_separation(lon, de_pt)) / 180.0
        cols[f"{name}_uccha_bala"] = uccha

        cols[f"{name}_is_exalted"] = (rashi == spec["exalt"]["rashi"]).astype(int)
        cols[f"{name}_is_debilitated"] = (rashi == spec["debil"]["rashi"]).astype(int)
        own = spec.get("own_rashis") or []
        cols[f"{name}_is_own"] = (
            np.isin(rashi, own).astype(int) if own else np.zeros(n, dtype=int)
        )

        mt = spec.get("moolatrikona")
        if mt:
            cols[f"{name}_is_moolatrikona"] = (
                (rashi == mt["rashi"]) & (rashi_deg >= mt["from_deg"]) & (rashi_deg < mt["to_deg"])
            ).astype(int)
        else:
            cols[f"{name}_is_moolatrikona"] = np.zeros(n, dtype=int)

        # --- Motion state ---------------------------------------------------------------
        cols[f"{name}_speed"] = speed
        mean_speed = abs(float(spec["mean_speed"]))
        cols[f"{name}_speed_norm"] = speed / mean_speed if mean_speed else np.zeros(n)

        if name in MOVING_GRAHAS:
            vakri = speed < 0
            cols[f"{name}_vakri"] = vakri.astype(int)
            # Stambhana: apparent motion has effectively halted.
            cols[f"{name}_stambhana"] = (np.abs(speed) < 0.12 * mean_speed).astype(int)
            cols[f"{name}_days_to_station"] = _days_to_flag_change(vakri, idx)
            # Cheshta bala proxy: retrograde motion is classically *strengthening*.
            cols[f"{name}_cheshta"] = np.where(vakri, 1.0, np.clip(speed / mean_speed, 0, 1.5))
        elif name in NODES:
            cols[f"{name}_vakri"] = np.ones(n, dtype=int)
            cols[f"{name}_stambhana"] = np.zeros(n, dtype=int)
            cols[f"{name}_cheshta"] = np.ones(n)
        else:
            cols[f"{name}_vakri"] = np.zeros(n, dtype=int)
            cols[f"{name}_stambhana"] = np.zeros(n, dtype=int)
            cols[f"{name}_cheshta"] = np.clip(speed / mean_speed, 0, 1.5)

        # --- Asta (combustion) ------------------------------------------------------------
        orb = spec.get("combust_orb_deg")
        if orb and name != "Surya":
            sep = ay.angular_separation(lon, sun)
            eff_orb = np.where(speed < 0, spec.get("combust_orb_retro_deg", orb), orb)
            cols[f"{name}_sun_sep"] = sep
            cols[f"{name}_asta"] = (sep < eff_orb).astype(int)
            # Depth of combustion: 1.0 at exact conjunction, 0 at the orb edge.
            asta_depth = np.clip(1.0 - sep / eff_orb, 0.0, 1.0)
            cols[f"{name}_asta_depth"] = asta_depth
        else:
            cols[f"{name}_sun_sep"] = (
                np.zeros(n) if name == "Surya" else np.full(n, np.nan)
            )
            cols[f"{name}_asta"] = np.zeros(n, dtype=int)
            asta_depth = np.zeros(n)
            cols[f"{name}_asta_depth"] = asta_depth

        # --- Aggregate benefic / malefic pressure -------------------------------------
        # Weight each graha by its natural strength and current dignity, discounting
        # whatever is combust and therefore unable to act.
        weight = float(spec["naisargika_bala"]) / 60.0
        contrib = weight * uccha * (1.0 - 0.6 * asta_depth)
        if spec["nature"] == "benefic":
            benefic_strength += contrib
        elif spec["nature"] == "malefic":
            malefic_strength += contrib
        else:  # Budha is conditionally benefic - count it at half weight.
            benefic_strength += 0.5 * contrib

        cols.update(ay.cyclic_encode(lon, 360.0, harmonics=1, prefix=f"{name}_"))

    # Chandra's benefic capacity is governed by paksha-bala: a dark Moon is a malefic.
    moon = positions["Chandra_lon"].to_numpy()
    cols["chandra_paksha_bala"] = (1.0 - np.cos(np.radians(np.mod(moon - sun, 360.0)))) / 2.0

    cols["benefic_strength"] = benefic_strength
    cols["malefic_strength"] = malefic_strength
    cols["graha_balance"] = benefic_strength - malefic_strength

    # --- Graha Yuddha (planetary war) ---------------------------------------------------
    # Two non-luminary grahas within one degree. Classically one is defeated and loses
    # its capacity to deliver results.
    war_orb = float(cfg["graha_yuddha_orb_deg"])
    war = np.zeros(n, dtype=int)
    for i, a in enumerate(MOVING_GRAHAS):
        for b in MOVING_GRAHAS[i + 1:]:
            sep = ay.angular_separation(
                positions[f"{a}_lon"].to_numpy(), positions[f"{b}_lon"].to_numpy()
            )
            war |= (sep < war_orb).astype(int)
    cols["graha_yuddha"] = war

    # --- Counts ---------------------------------------------------------------------------
    cols["n_vakri"] = sum(cols[f"{g}_vakri"] for g in MOVING_GRAHAS)
    cols["n_asta"] = sum(cols[f"{g}_asta"] for g in MOVING_GRAHAS)
    cols["n_exalted"] = sum(cols[f"{g}_is_exalted"] for g in REAL_GRAHAS)
    cols["n_debilitated"] = sum(cols[f"{g}_is_debilitated"] for g in REAL_GRAHAS)

    return pd.DataFrame(cols, index=idx)
