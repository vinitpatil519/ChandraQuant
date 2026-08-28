"""Lahiri (Chitrapaksha) ayanamsa and the sidereal-zodiac primitives.

Skyfield returns *tropical* ecliptic longitudes (equinox of date). Jyotisha works in
the *sidereal* zodiac, which trails the tropical one by the ayanamsa - the accumulated
general precession since the sidereal zero point was defined.

Lahiri is the ayanamsa adopted by the Indian Calendar Reform Committee and used by
every mainstream Indian panchanga. Its anchor is 23 deg 15' 00.65" at 1956-01-01
(JD 2435553.5); from there we accumulate IAU-2006 general precession in longitude.
This reproduces Swiss Ephemeris' SE_SIDM_LAHIRI to a few arc-seconds across 1900-2100,
which is four orders of magnitude tighter than the 13 deg 20' width of a nakshatra.
"""

from __future__ import annotations

import numpy as np

# --- Sidereal zero-point anchor (Swiss Ephemeris SE_SIDM_LAHIRI) --------------------
_LAHIRI_JD0 = 2435553.5        # 1956-01-01 TT
_LAHIRI_AYAN0 = 23.250182      # degrees at that instant
_J2000 = 2451545.0
_T0 = (_LAHIRI_JD0 - _J2000) / 36525.0

# Degrees per zodiacal division
DEG_PER_RASHI = 30.0
DEG_PER_NAKSHATRA = 360.0 / 27.0        # 13 deg 20'
DEG_PER_PADA = DEG_PER_NAKSHATRA / 4.0  # 3 deg 20'
DEG_PER_TITHI = 12.0
DEG_PER_KARANA = 6.0
DEG_PER_YOGA = 360.0 / 27.0

NUM_RASHIS = 12
NUM_NAKSHATRAS = 27
NUM_TITHIS = 30
NUM_YOGAS = 27
NUM_KARANA_SLOTS = 60


def _general_precession_arcsec(T: np.ndarray | float) -> np.ndarray | float:
    """IAU 2006 accumulated general precession in longitude, arcseconds.

    T is Julian centuries from J2000.0 (TT).
    """
    return (
        5028.796195 * T
        + 1.1054348 * T**2
        + 0.00007964 * T**3
        - 0.000023857 * T**4
        - 0.0000000383 * T**5
    )


_PREC_AT_ANCHOR = _general_precession_arcsec(_T0)


def lahiri_ayanamsa(jd_tt: np.ndarray | float) -> np.ndarray | float:
    """Lahiri ayanamsa in degrees for one or many Julian Days (TT)."""
    T = (np.asarray(jd_tt, dtype=float) - _J2000) / 36525.0
    delta_arcsec = _general_precession_arcsec(T) - _PREC_AT_ANCHOR
    return _LAHIRI_AYAN0 + delta_arcsec / 3600.0


def to_sidereal(tropical_lon_deg, jd_tt):
    """Convert tropical ecliptic longitude(s) to Lahiri sidereal, wrapped to [0, 360)."""
    return np.mod(np.asarray(tropical_lon_deg, dtype=float) - lahiri_ayanamsa(jd_tt), 360.0)


def to_tropical(sidereal_lon_deg, jd_tt):
    """Inverse of :func:`to_sidereal`."""
    return np.mod(np.asarray(sidereal_lon_deg, dtype=float) + lahiri_ayanamsa(jd_tt), 360.0)


# --- Zodiacal decomposition ---------------------------------------------------------


def rashi_index(lon_deg):
    """0-based rashi (sign) index: 0 = Mesha ... 11 = Meena."""
    return (np.asarray(lon_deg, dtype=float) // DEG_PER_RASHI).astype(int) % NUM_RASHIS


def rashi_degree(lon_deg):
    """Degrees elapsed within the current rashi, in [0, 30)."""
    return np.mod(np.asarray(lon_deg, dtype=float), DEG_PER_RASHI)


def nakshatra_index(lon_deg):
    """0-based nakshatra index: 0 = Ashwini ... 26 = Revati."""
    return (np.asarray(lon_deg, dtype=float) // DEG_PER_NAKSHATRA).astype(int) % NUM_NAKSHATRAS


def nakshatra_fraction(lon_deg):
    """Fraction of the current nakshatra already traversed, in [0, 1)."""
    return np.mod(np.asarray(lon_deg, dtype=float), DEG_PER_NAKSHATRA) / DEG_PER_NAKSHATRA


def pada_index(lon_deg):
    """1-based pada (quarter) within the nakshatra: 1..4."""
    frac = np.mod(np.asarray(lon_deg, dtype=float), DEG_PER_NAKSHATRA)
    return (frac // DEG_PER_PADA).astype(int) + 1


def pada_global(lon_deg):
    """0-based index into the 108 padas of the zodiac."""
    return (np.asarray(lon_deg, dtype=float) // DEG_PER_PADA).astype(int) % 108


def navamsa_index(lon_deg):
    """0-based navamsa (D9) sign index - each pada maps to one navamsa sign."""
    return (np.asarray(lon_deg, dtype=float) // DEG_PER_PADA).astype(int) % NUM_RASHIS


# --- Angular helpers ----------------------------------------------------------------


def angular_separation(a_deg, b_deg):
    """Unsigned separation in [0, 180] between two ecliptic longitudes."""
    diff = np.mod(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float), 360.0)
    return np.where(diff > 180.0, 360.0 - diff, diff)


def signed_elongation(a_deg, b_deg):
    """Signed a-minus-b elongation wrapped to [0, 360)."""
    return np.mod(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float), 360.0)


def house_from(reference_deg, target_deg):
    """Whole-sign house of `target` counted from the sign of `reference`. Returns 1..12.

    This is the Vedic counting convention: the reference's own sign is house 1.
    """
    ref_sign = rashi_index(reference_deg)
    tgt_sign = rashi_index(target_deg)
    return ((tgt_sign - ref_sign) % NUM_RASHIS) + 1


def cyclic_encode(values, period: float, harmonics: int = 2, prefix: str = "") -> dict:
    """sin/cos harmonic embedding of a cyclic variable.

    Angular variables have a discontinuity at the wrap point (359 deg and 1 deg are
    adjacent but numerically far apart). Projecting onto the unit circle removes it,
    and extra harmonics let a tree model resolve sub-cycle structure.
    """
    arr = np.asarray(values, dtype=float)
    theta = 2.0 * np.pi * arr / period
    out = {}
    for h in range(1, harmonics + 1):
        suffix = "" if h == 1 else f"{h}"
        out[f"{prefix}sin{suffix}"] = np.sin(h * theta)
        out[f"{prefix}cos{suffix}"] = np.cos(h * theta)
    return out
