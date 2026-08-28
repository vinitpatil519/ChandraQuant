"""Shadbala - the six-fold strength of the grahas, in virupas and rupas.

Shadbala is classical Jyotisha's answer to "how much can this planet actually do right
now". A graha may sit in a favourable house and still be powerless; Shadbala scores it
across six independent sources of strength and sums them. 60 virupas make one rupa,
and each graha has a classical minimum rupa threshold below which its promises fail.

The six sources, and what each measures:

  Sthana bala   positional - exaltation, house type (kendra/panapara/apoklima),
                odd-even sign preference
  Dig bala      directional - each graha is strongest in one specific quarter of the
                chart (Surya and Mangala in the 10th, Guru and Budha in the 1st,
                Chandra and Shukra in the 4th, Shani in the 7th)
  Kala bala     temporal - day/night preference, paksha, weekday and hora rulership
  Cheshta bala  motional - retrogression is *strengthening*, which is the opposite of
                the popular reading and worth noting
  Naisargika    natural, fixed - Surya strongest, Shani weakest
  Drik bala     aspectual - the net benefic minus malefic drishti received

Simplification declared honestly: Saptavargaja bala requires the seven divisional
charts (D1-D9), which are not computed here. Its contribution is approximated by the
dignity terms already present in Sthana bala. Everything else follows Parashara.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import graha_cfg
from . import ayanamsa as ay

SHADBALA_GRAHAS = ["Surya", "Chandra", "Mangala", "Budha", "Guru", "Shukra", "Shani"]

# Classical minimum required strength, in rupas.
REQUIRED_RUPAS = {
    "Surya": 5.0, "Chandra": 6.0, "Mangala": 5.0, "Budha": 7.0,
    "Guru": 6.5, "Shukra": 5.5, "Shani": 5.0,
}

# Diurnal / nocturnal preference for Nathonnatha bala.
DAY_STRONG = {"Surya", "Guru", "Shukra"}
NIGHT_STRONG = {"Chandra", "Mangala", "Shani"}

# Male (odd-sign preferring) vs female (even-sign preferring) grahas for Ojhayugma bala.
MALE_GRAHAS = {"Surya", "Mangala", "Guru", "Budha"}
FEMALE_GRAHAS = {"Chandra", "Shukra"}

KENDRA = [1, 4, 7, 10]
PANAPARA = [2, 5, 8, 11]

OBLIQUITY_DEG = 23.4392911


def _dig_bala(lon: np.ndarray, lagna: np.ndarray, best_house: int) -> np.ndarray:
    """Directional strength: 60 virupas at the graha's strongest house cusp, 0 opposite."""
    best_point = np.mod(lagna + (best_house - 1) * 30.0, 360.0)
    # Strength falls linearly with angular distance from the strongest point.
    sep = ay.angular_separation(lon, best_point)
    return 60.0 * (1.0 - sep / 180.0)


def compute(
    positions: pd.DataFrame,
    lagna: np.ndarray,
    panchanga_df: pd.DataFrame,
    grahas_df: pd.DataFrame,
    aspects_df: pd.DataFrame,
) -> pd.DataFrame:
    """Shadbala for the seven classical grahas."""
    cfg = graha_cfg()
    specs = {g["name"]: g for g in cfg["grahas"]}
    idx = positions.index
    n = len(idx)
    cols: dict[str, np.ndarray] = {}

    lagna = np.asarray(lagna, dtype=float)
    paksha_bala_frac = panchanga_df["paksha_bala"].to_numpy()
    vara_lord = panchanga_df["vara_lord"].to_numpy()
    sun_trop = positions["Surya_trop"].to_numpy()

    total_rupas = np.zeros(n)
    benefic_rupas = np.zeros(n)
    malefic_rupas = np.zeros(n)

    for graha in SHADBALA_GRAHAS:
        spec = specs[graha]
        lon = positions[f"{graha}_lon"].to_numpy()
        speed = positions[f"{graha}_speed"].to_numpy()
        rashi = ay.rashi_index(lon)
        house = ay.house_from(lagna, lon)

        # --- 1. Sthana bala ---------------------------------------------------------
        uccha = grahas_df[f"{graha}_uccha_bala"].to_numpy() * 60.0
        kendradi = np.where(
            np.isin(house, KENDRA), 60.0, np.where(np.isin(house, PANAPARA), 30.0, 15.0)
        )
        odd_sign = (rashi % 2) == 0  # rashi index 0 = Mesha, an odd sign
        if graha in MALE_GRAHAS:
            ojhayugma = np.where(odd_sign, 15.0, 0.0)
        elif graha in FEMALE_GRAHAS:
            ojhayugma = np.where(~odd_sign, 15.0, 0.0)
        else:
            ojhayugma = np.full(n, 7.5)
        # Drekkana bala: male grahas in the first third of a sign, etc.
        third = (ay.rashi_degree(lon) // 10.0).astype(int)
        if graha in MALE_GRAHAS:
            drekkana = np.where(third == 0, 15.0, 0.0)
        elif graha in FEMALE_GRAHAS:
            drekkana = np.where(third == 2, 15.0, 0.0)
        else:
            drekkana = np.where(third == 1, 15.0, 0.0)
        sthana = uccha + kendradi + ojhayugma + drekkana

        # --- 2. Dig bala -------------------------------------------------------------
        dig = _dig_bala(lon, lagna, int(spec["dig_bala_house"]))

        # --- 3. Kala bala ------------------------------------------------------------
        # Nathonnatha: the market open is a daytime moment, so day-strong grahas are
        # near full and night-strong grahas near their floor.
        if graha in DAY_STRONG:
            nathonnatha = np.full(n, 48.0)
        elif graha in NIGHT_STRONG:
            nathonnatha = np.full(n, 12.0)
        else:
            nathonnatha = np.full(n, 60.0)   # Budha is strong at all times

        # Paksha bala: benefics gain with the waxing Moon, malefics with the waning.
        if spec["nature"] == "benefic":
            paksha = 60.0 * paksha_bala_frac
        elif spec["nature"] == "malefic":
            paksha = 60.0 * (1.0 - paksha_bala_frac)
        else:
            paksha = np.full(n, 30.0)
        if graha == "Chandra":
            # Chandra's paksha bala is her own illumination, doubled in the classical rule.
            paksha = 60.0 * paksha_bala_frac

        # Dina bala: the lord of the weekday gets the full measure.
        dina = np.where(vara_lord == graha, 45.0, 0.0)

        # Ayana bala: strength from declination relative to the celestial equator.
        trop = positions[f"{graha}_trop"].to_numpy()
        decl = np.degrees(
            np.arcsin(np.sin(np.radians(OBLIQUITY_DEG)) * np.sin(np.radians(trop)))
        )
        northward = decl >= 0
        if graha in {"Surya", "Mangala", "Guru", "Shukra"}:
            ayana = 60.0 * (0.5 + 0.5 * decl / 24.0)
        elif graha in {"Chandra", "Shani"}:
            ayana = 60.0 * (0.5 - 0.5 * decl / 24.0)
        else:  # Budha is strong in both ayanas
            ayana = np.full(n, 60.0)
        ayana = np.clip(ayana, 0.0, 60.0)

        kala = nathonnatha + paksha + dina + ayana

        # --- 4. Cheshta bala ----------------------------------------------------------
        # Retrogression is classically strengthening: a vakri graha is at its most
        # insistent. Direct motion scales with how far the graha is from mean speed.
        mean_speed = abs(float(spec["mean_speed"]))
        if graha in {"Surya", "Chandra"}:
            cheshta = np.full(n, 30.0)   # the luminaries never retrograde
        else:
            rel = speed / mean_speed if mean_speed else np.zeros(n)
            cheshta = np.where(speed < 0, 60.0, np.clip(60.0 * (1.0 - rel), 0.0, 60.0))

        # --- 5. Naisargika bala --------------------------------------------------------
        naisargika = np.full(n, float(spec["naisargika_bala"]))

        # --- 6. Drik bala ---------------------------------------------------------------
        drik_col = f"drishti_{graha}_net"
        drik = aspects_df[drik_col].to_numpy() * 15.0 if drik_col in aspects_df else np.zeros(n)

        # --- Total ------------------------------------------------------------------------
        virupas = sthana + dig + kala + cheshta + naisargika + drik
        rupas = virupas / 60.0

        cols[f"sb_{graha}_sthana"] = sthana
        cols[f"sb_{graha}_dig"] = dig
        cols[f"sb_{graha}_kala"] = kala
        cols[f"sb_{graha}_cheshta"] = cheshta
        cols[f"sb_{graha}_drik"] = drik
        cols[f"sb_{graha}_rupas"] = rupas
        # Ishta / Kashta: does the graha clear its classical minimum?
        cols[f"sb_{graha}_ratio"] = rupas / REQUIRED_RUPAS[graha]
        cols[f"sb_{graha}_sufficient"] = (rupas >= REQUIRED_RUPAS[graha]).astype(int)

        total_rupas += rupas
        if spec["nature"] == "benefic":
            benefic_rupas += rupas
        elif spec["nature"] == "malefic":
            malefic_rupas += rupas
        else:
            benefic_rupas += 0.5 * rupas

    cols["sb_total_rupas"] = total_rupas
    cols["sb_benefic_rupas"] = benefic_rupas
    cols["sb_malefic_rupas"] = malefic_rupas
    cols["sb_net_rupas"] = benefic_rupas - malefic_rupas
    cols["sb_n_sufficient"] = sum(cols[f"sb_{g}_sufficient"] for g in SHADBALA_GRAHAS)
    # Graha Shakti Index: benefic minus malefic strength, standardised later.
    cols["graha_shakti_raw"] = benefic_rupas - malefic_rupas

    return pd.DataFrame(cols, index=idx)
