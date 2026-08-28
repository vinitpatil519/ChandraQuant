"""Chandra mechanics - the fast hand of the Jyotisha clock.

The Moon completes a full circuit of the zodiac in 27.32 days and a full phase cycle
in 29.53. It is by far the fastest classical graha, which makes it the only one whose
state changes on a timescale comparable to a swing trade. In mundane astrology Chandra
governs *manas* - the collective mind - which maps onto crowd sentiment and short-cycle
liquidity, so its geometry carries most of the tradeable astro signal.

Beyond phase, this module extracts the mechanics that classical texts treat as
destabilising: Gandanta (the water-fire junctions), Sandhi (sign boundaries), the
Moon's varying speed (Sighra / Manda gati), its latitude (Chandra Shara, which governs
eclipse possibility) and its varying distance (perigee / apogee).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import nakshatra_cfg
from . import ayanamsa as ay

# Gandanta zones are centred on the water->fire sign junctions, expressed as sidereal
# longitude: Meena/Mesha (0 deg), Karka/Simha (120 deg), Vrischika/Dhanu (240 deg).
GANDANTA_CENTRES = np.array([0.0, 120.0, 240.0])

# Mean lunar distance in AU, used to normalise perigee / apogee.
MEAN_MOON_DIST_AU = 0.00257


def _circ_dist(a: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Minimum absolute circular distance from each `a` to any of `centres`, degrees."""
    d = np.abs(((a[:, None] - centres[None, :]) + 180.0) % 360.0 - 180.0)
    return d.min(axis=1)


def compute(positions: pd.DataFrame) -> pd.DataFrame:
    """Lunar mechanics features from an ephemeris frame."""
    moon = positions["Chandra_lon"].to_numpy()
    sun = positions["Surya_lon"].to_numpy()
    lat = positions["Chandra_lat"].to_numpy()
    dist = positions["Chandra_dist"].to_numpy()
    speed = positions["Chandra_speed"].to_numpy()
    elong = np.mod(moon - sun, 360.0)

    df = pd.DataFrame(index=positions.index)

    # --- Phase ---------------------------------------------------------------------
    # Illuminated fraction of the disc: 0 at Amavasya, 1 at Purnima.
    df["moon_phase_angle"] = elong
    df["moon_illumination"] = (1.0 - np.cos(np.radians(elong))) / 2.0
    # Signed distance from the nearest syzygy, +1 at Purnima and -1 at Amavasya.
    df["moon_syzygy_bias"] = -np.cos(np.radians(elong))
    # Absolute nearness to *either* syzygy - both are classical turning points.
    df["moon_syzygy_proximity"] = np.abs(np.cos(np.radians(elong)))

    # --- Chandra Shara (celestial latitude) and eclipse geometry ---------------------
    df["chandra_shara"] = lat
    df["chandra_shara_abs"] = np.abs(lat)
    # An eclipse needs a syzygy AND the Moon near a node (small |latitude|). This is a
    # smooth 0-1 likelihood, not a detection; discrete eclipses live in events.py.
    node_nearness = np.clip(1.0 - np.abs(lat) / 1.6, 0.0, 1.0)
    syzygy_nearness = np.abs(np.cos(np.radians(elong))) ** 6
    df["eclipse_potential"] = node_nearness * syzygy_nearness

    # --- Gati (speed) ---------------------------------------------------------------
    # The Moon runs 11.8-15.4 deg/day. Sighra gati (swift) classically accelerates
    # whatever is underway; Manda gati (slow) stalls it.
    df["moon_speed"] = speed
    df["moon_speed_norm"] = (speed - 13.176) / 1.2
    df["moon_is_sighra"] = (speed > 13.6).astype(int)
    df["moon_is_manda"] = (speed < 12.6).astype(int)

    # --- Distance (perigee / apogee) -------------------------------------------------
    df["moon_dist"] = dist
    df["moon_dist_norm"] = (dist - MEAN_MOON_DIST_AU) / (MEAN_MOON_DIST_AU * 0.055)
    df["moon_is_perigee"] = (df["moon_dist_norm"] < -1.2).astype(int)
    df["moon_is_apogee"] = (df["moon_dist_norm"] > 1.2).astype(int)
    # Supermoon analogue: perigee coinciding with syzygy - maximum tidal forcing.
    df["moon_tidal_force"] = (MEAN_MOON_DIST_AU / dist) ** 3 * df["moon_syzygy_proximity"]

    # --- Gandanta ---------------------------------------------------------------------
    orb = float(nakshatra_cfg()["gandanta_orb_deg"])
    gd = _circ_dist(moon, GANDANTA_CENTRES)
    df["gandanta_distance"] = gd
    df["moon_in_gandanta"] = (gd <= orb).astype(int)
    # Smooth ramp so the model sees the approach, not just the knot itself.
    df["gandanta_intensity"] = np.clip(1.0 - gd / (orb * 3.0), 0.0, 1.0)

    # --- Sandhi (sign junction) -------------------------------------------------------
    deg_in_rashi = ay.rashi_degree(moon)
    edge = np.minimum(deg_in_rashi, 30.0 - deg_in_rashi)
    df["moon_sandhi_distance"] = edge
    df["moon_in_sandhi"] = (edge <= 1.0).astype(int)

    # --- Nakshatra transition mechanics -----------------------------------------------
    nak_frac = ay.nakshatra_fraction(moon)
    df["nak_progress"] = nak_frac
    # Days until the Moon leaves the current nakshatra, at its current speed.
    remaining_deg = (1.0 - nak_frac) * ay.DEG_PER_NAKSHATRA
    df["days_to_nak_change"] = remaining_deg / np.maximum(speed, 1e-6)
    # The final pada before a nakshatra change is a classical "void" - unsupported action.
    df["moon_last_pada"] = (ay.pada_index(moon) == 4).astype(int)
    df["moon_void_like"] = ((nak_frac > 0.90) | (nak_frac < 0.10)).astype(int)

    # --- Chandra Kriya / Avastha / Vela ------------------------------------------------
    # Three classical subdivisions of the Chandra-Surya relationship (Jataka Parijata),
    # implemented as equal divisions of the synodic cycle: 60, 12 and 36 states.
    df["chandra_kriya"] = (elong / 6.0).astype(int)      # 60 states
    df["chandra_avastha"] = (elong / 30.0).astype(int)   # 12 states
    df["chandra_vela"] = (elong / 10.0).astype(int)      # 36 states

    # --- Cyclic embeddings --------------------------------------------------------------
    for values, period, prefix, harm in (
        (elong, 360.0, "phase_", 4),
        (lat, 10.8, "shara_", 1),
        (df["chandra_kriya"].to_numpy(), 60.0, "kriya_", 1),
        (df["chandra_avastha"].to_numpy(), 12.0, "avastha_", 1),
    ):
        for name, arr in ay.cyclic_encode(values, period, harmonics=harm, prefix=prefix).items():
            df[name] = arr

    # --- Chandra Bala Index contribution -------------------------------------------------
    # Composite of the destabilising terms; consumed by composites.py.
    df["lunar_instability"] = (
        0.40 * df["gandanta_intensity"]
        + 0.25 * df["eclipse_potential"]
        + 0.20 * df["moon_in_sandhi"]
        + 0.15 * df["moon_void_like"]
    )
    return df
