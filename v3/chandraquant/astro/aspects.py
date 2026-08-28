"""Drishti (Vedic aspects), continuous angular kernels, and the named market yogas.

Two aspect systems run side by side, deliberately:

1. **Parashari drishti** - the classical Vedic scheme. Every graha aspects the 7th
   house from itself fully; Mangala additionally aspects the 4th and 8th, Guru the
   5th and 9th, Shani the 3rd and 10th. It is whole-sign and discrete, which makes it
   interpretable but coarse.

2. **Continuous orb-decayed kernels** - conjunction, sextile, square, trine and
   opposition with smooth falloff. These behave far better numerically: a tree model
   can split on "how exact is the square" rather than on a step function, and the
   approach and separation of an aspect become visible as a ramp.

The named yogas (Shrapit, Angarak, Gajakesari, Grahan, the Guru-Shani Great
Conjunction) are the combinations mundane astrology explicitly ties to market
behaviour. Each emits both a boolean and a continuous strength.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..config import graha_cfg
from . import ayanamsa as ay
from .ephemeris import NAVAGRAHA

# Pairs worth carrying explicitly. All 36 unordered navagraha pairs are computed for
# raw separation; the aspect decomposition is kept for the pairs with real classical
# or cyclical meaning, to keep the feature count sane.
KEY_PAIRS = [
    ("Guru", "Shani"), ("Guru", "Rahu"), ("Shani", "Rahu"), ("Mangala", "Shani"),
    ("Mangala", "Rahu"), ("Guru", "Shukra"), ("Surya", "Shani"), ("Surya", "Guru"),
    ("Chandra", "Guru"), ("Chandra", "Shani"), ("Chandra", "Rahu"), ("Chandra", "Mangala"),
    ("Budha", "Shukra"), ("Budha", "Surya"), ("Shukra", "Shani"), ("Mangala", "Guru"),
]

BENEFICS = ["Guru", "Shukra"]
MALEFICS = ["Shani", "Mangala", "Rahu", "Ketu"]


def _aspect_strength(sep: np.ndarray, angle: float, orb: float, falloff: float) -> np.ndarray:
    """Smooth orb-decayed strength of one aspect: 1.0 exact, 0.0 at the orb edge."""
    return np.clip(1.0 - np.abs(sep - angle) / orb, 0.0, 1.0) ** falloff


def _drishti_strength(house: np.ndarray, graha: str, cfg: dict) -> np.ndarray:
    """Parashari aspect strength cast by `graha` onto a body sitting `house` away."""
    table = dict(cfg["drishti"]["default"])
    table.update({int(k): v for k, v in cfg["drishti"]["special"].get(graha, {}).items()})
    out = np.zeros(len(house), dtype=float)
    for h, strength in table.items():
        out = np.where(house == int(h), float(strength), out)
    return out


def compute(positions: pd.DataFrame) -> pd.DataFrame:
    cfg = graha_cfg()
    idx = positions.index
    n = len(idx)
    cols: dict[str, np.ndarray] = {}

    lons = {g: positions[f"{g}_lon"].to_numpy() for g in NAVAGRAHA}
    naisargika = {g["name"]: float(g["naisargika_bala"]) / 60.0 for g in cfg["grahas"]}

    # --- Raw separations for every unordered pair ------------------------------------
    for a, b in itertools.combinations(NAVAGRAHA, 2):
        cols[f"sep_{a}_{b}"] = ay.angular_separation(lons[a], lons[b])

    # --- Continuous aspect decomposition for the key pairs ---------------------------
    falloff = float(cfg["falloff"])
    siderograph = np.zeros(n)
    for a, b in KEY_PAIRS:
        sep = cols[f"sep_{a}_{b}"] if f"sep_{a}_{b}" in cols else cols[f"sep_{b}_{a}"]
        pair_signal = np.zeros(n)
        for asp in cfg["western_aspects"]:
            s = _aspect_strength(sep, float(asp["angle"]), float(asp["orb"]), falloff)
            cols[f"asp_{a}_{b}_{asp['name']}"] = s
            pair_signal += float(asp["polarity"]) * s
        cols[f"aspect_{a}_{b}"] = pair_signal
        # Bradley-siderograph weighting: slower, heavier bodies dominate the tide.
        weight = 1.0 / (1.0 + naisargika.get(a, 0.5) + naisargika.get(b, 0.5))
        siderograph += weight * pair_signal

    # --- Kala Taranga: the normalised cosmic tide ------------------------------------
    sd = siderograph.std()
    cols["siderograph_raw"] = siderograph
    cols["kala_taranga"] = (siderograph - siderograph.mean()) / (sd if sd > 1e-9 else 1.0)

    # --- Parashari drishti received by each graha ------------------------------------
    for target in NAVAGRAHA:
        ben = np.zeros(n)
        mal = np.zeros(n)
        total = np.zeros(n)
        for caster in NAVAGRAHA:
            if caster == target:
                continue
            house = ay.house_from(lons[caster], lons[target])
            strength = _drishti_strength(house, caster, cfg)
            total += strength
            if caster in BENEFICS:
                ben += strength
            elif caster in MALEFICS:
                mal += strength
        cols[f"drishti_{target}_benefic"] = ben
        cols[f"drishti_{target}_malefic"] = mal
        cols[f"drishti_{target}_net"] = ben - mal
        cols[f"drishti_{target}_total"] = total

    # --- Named yogas with mundane market lore ----------------------------------------
    for spec in cfg["named_yogas"]:
        a, b = spec["pair"]
        name = spec["name"]
        sep = ay.angular_separation(lons[a], lons[b])
        if spec.get("kendra"):
            # Gajakesari: Guru in a kendra (1/4/7/10) from Chandra.
            house = ay.house_from(lons[b], lons[a])
            flag = np.isin(house, [1, 4, 7, 10]).astype(int)
            strength = flag.astype(float)
        else:
            orb = float(spec["orb"])
            flag = (sep <= orb).astype(int)
            strength = np.clip(1.0 - sep / orb, 0.0, 1.0)
        cols[f"yoga_{name}"] = flag
        cols[f"yoga_{name}_strength"] = strength * float(spec["polarity"])

    # --- Kemadruma: the isolated Moon -------------------------------------------------
    # No graha (excluding Surya and the nodes) in the 2nd or 12th from Chandra: the
    # crowd is structurally unsupported, a liquidity vacuum.
    kem_cfg = cfg["kemadruma"]
    excluded = set(kem_cfg["exclude"]) | {"Chandra"}
    occupied = np.zeros(n, dtype=bool)
    for g in NAVAGRAHA:
        if g in excluded:
            continue
        house = ay.house_from(lons["Chandra"], lons[g])
        occupied |= np.isin(house, kem_cfg["houses_checked"])
    cols["yoga_Kemadruma"] = (~occupied).astype(int)

    # Sunapha / Anapha / Durudhara: grahas flanking Chandra - the opposite of Kemadruma.
    in_2nd = np.zeros(n, dtype=bool)
    in_12th = np.zeros(n, dtype=bool)
    for g in NAVAGRAHA:
        if g in excluded:
            continue
        house = ay.house_from(lons["Chandra"], lons[g])
        in_2nd |= house == 2
        in_12th |= house == 12
    cols["yoga_Sunapha"] = (in_2nd & ~in_12th).astype(int)
    cols["yoga_Anapha"] = (in_12th & ~in_2nd).astype(int)
    cols["yoga_Durudhara"] = (in_2nd & in_12th).astype(int)

    # --- Aggregate aspect climate ------------------------------------------------------
    benefic_yogas = ["GuruShukra", "GajaKesari", "BudhaAditya"]
    malefic_yogas = ["ShaniRahu", "MangalaRahu", "ChandraRahu", "ChandraKetu", "ShaniMangala"]
    cols["yoga_benefic_pressure"] = sum(
        np.abs(cols[f"yoga_{y}_strength"]) for y in benefic_yogas if f"yoga_{y}_strength" in cols
    )
    cols["yoga_malefic_pressure"] = sum(
        np.abs(cols[f"yoga_{y}_strength"]) for y in malefic_yogas if f"yoga_{y}_strength" in cols
    )
    cols["yoga_net_pressure"] = cols["yoga_benefic_pressure"] - cols["yoga_malefic_pressure"]

    return pd.DataFrame(cols, index=idx)
