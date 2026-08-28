"""Discrete celestial events - Grahana (eclipses), Gochara ingress, and stations.

Continuous features tell the model where the sky *is*. Event features tell it what
just happened and what is about to. Both matter, but events matter more for regime
detection, because classical mundane astrology treats them as the moments when a
regime actually turns:

  Grahana (eclipse)   - the luminaries obscured at a node. The single most feared
                        configuration in Jyotisha; markets have their own eclipse
                        folklore, and the +/- window around one is a genuine
                        volatility cluster.
  Gochara (ingress)   - a graha crossing into a new rashi. Shani ingresses every ~2.5
                        years, Guru every ~1 year, Rahu-Ketu every ~1.5. These carve
                        the timeline into macro regime bands.
  Stambhana (station) - direction reversal, handled in grahas.py; here we add the
                        days-to / days-since ramps.

Every event is emitted three ways: a flag, a signed days-to/days-since ramp, and a
window indicator. Ramps matter - a model that only sees a one-day spike cannot learn
anticipation, which is the entire point of a forward-computable feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


from . import ayanamsa as ay
from .ephemeris import luminaries_at_utc

# Slow grahas whose rashi changes define macro regime bands.
INGRESS_GRAHAS = ["Guru", "Shani", "Rahu"]

# Ecliptic limits. An eclipse occurs only if the Moon's ecliptic latitude at the exact
# syzygy is inside these bounds - this is the real astronomical criterion, not a
# heuristic. Solar: an eclipse is certain inside 1.40 deg; between 1.40 and 1.58 only a
# grazing polar partial is possible, so we use the tighter limit - it drops the two
# 2026 borderline cases (beta 1.56, 1.49) while keeping every catalogued eclipse.
# Lunar: the Moon clips the penumbra out to ~1.55 deg and the umbra out to ~1.03 deg.
SOLAR_ECLIPTIC_LIMIT_DEG = 1.40
LUNAR_PENUMBRAL_LIMIT_DEG = 1.55
LUNAR_UMBRAL_LIMIT_DEG = 1.03

ECLIPSE_WINDOWS = (1, 3, 7)


def detect_eclipses(dates: pd.DatetimeIndex, positions: pd.DataFrame) -> pd.DataFrame:
    """Locate true eclipses by refining each syzygy and testing the ecliptic limit.

    Daily sampling is far too coarse to judge an eclipse: the Moon moves ~13 deg a day
    and its latitude swings through the whole node crossing in under 48 hours. So we
    bracket every conjunction and opposition, interpolate the exact instant, evaluate
    the Moon's latitude *there*, and apply the classical ecliptic limits.

    Returns a frame indexed by eclipse date with columns ``kind`` and ``moon_lat``.
    """
    elong = np.mod(
        positions["Chandra_lon"].to_numpy() - positions["Surya_lon"].to_numpy(), 360.0
    )
    day_num = (dates - dates[0]).days.to_numpy().astype(float)

    # Bracket syzygies: conjunction is the 360->0 wrap, opposition the upward 180 cross.
    candidates: list[tuple[float, str]] = []
    d_conj = elong  # zero at conjunction
    d_oppo = np.mod(elong - 180.0, 360.0)  # zero at opposition
    for series, kind in ((d_conj, "solar"), (d_oppo, "lunar")):
        wrapped = series > 180.0
        # A crossing is where the series wraps from just-below-360 to just-above-0.
        for i in range(len(series) - 1):
            if wrapped[i] and not wrapped[i + 1]:
                a = series[i] - 360.0  # negative, approaching zero
                b = series[i + 1]      # positive, just past
                if b - a <= 0:
                    continue
                frac = -a / (b - a)
                if 0.0 <= frac <= 1.0:
                    candidates.append((day_num[i] + frac, kind))

    if not candidates:
        return pd.DataFrame(columns=["kind", "moon_lat"])

    base_utc = pd.Timestamp(dates[0]).tz_localize("UTC")
    instants = [base_utc + pd.Timedelta(days=float(d)) for d, _ in candidates]
    lum = luminaries_at_utc(instants)
    lat = lum["moon_lat"].to_numpy()

    rows = []
    for (day_off, kind), beta in zip(candidates, lat):
        limit = SOLAR_ECLIPTIC_LIMIT_DEG if kind == "solar" else LUNAR_PENUMBRAL_LIMIT_DEG
        if abs(beta) <= limit:
            when = pd.Timestamp(dates[0]) + pd.Timedelta(days=float(day_off))
            rows.append({"date": when.normalize(), "kind": kind, "moon_lat": float(beta)})

    if not rows:
        return pd.DataFrame(columns=["kind", "moon_lat"])
    out = pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date")
    return out


def _signed_days_to_events(dates: pd.DatetimeIndex, event_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Days until the next event and since the previous one (NaN outside the sample)."""
    day_num = (dates - dates[0]).days.to_numpy().astype(float)
    ev_days = day_num[event_mask]
    n = len(dates)
    to_next = np.full(n, np.nan)
    since_prev = np.full(n, np.nan)
    if len(ev_days) == 0:
        return to_next, since_prev
    pos = np.searchsorted(ev_days, day_num, side="left")
    has_next = pos < len(ev_days)
    to_next[has_next] = ev_days[pos[has_next]] - day_num[has_next]
    prev_pos = np.searchsorted(ev_days, day_num, side="right") - 1
    has_prev = prev_pos >= 0
    since_prev[has_prev] = day_num[has_prev] - ev_days[prev_pos[has_prev]]
    return to_next, since_prev


def _step_change_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    mask[1:] = np.diff(values) != 0
    return mask


def compute(positions: pd.DataFrame, lunar_df: pd.DataFrame) -> pd.DataFrame:
    """Event features. Requires the lunar frame for the eclipse potential series."""
    idx = positions.index
    n = len(idx)
    cols: dict[str, np.ndarray] = {}

    # --- Grahana (eclipses) -----------------------------------------------------------
    eclipses = detect_eclipses(idx, positions)
    eclipse_mask = idx.isin(eclipses.index)
    solar_mask = idx.isin(eclipses.index[eclipses["kind"] == "solar"]) if len(eclipses) else np.zeros(n, dtype=bool)
    lunar_mask = idx.isin(eclipses.index[eclipses["kind"] == "lunar"]) if len(eclipses) else np.zeros(n, dtype=bool)
    # Central eclipses (small |beta|) are the ones classical texts treat as potent.
    if len(eclipses):
        central = eclipses.index[eclipses["moon_lat"].abs() <= LUNAR_UMBRAL_LIMIT_DEG]
        central_mask = idx.isin(central)
    else:
        central_mask = np.zeros(n, dtype=bool)

    cols["is_eclipse"] = eclipse_mask.astype(int)
    cols["is_solar_eclipse"] = solar_mask.astype(int)
    cols["is_lunar_eclipse"] = lunar_mask.astype(int)
    cols["is_central_eclipse"] = central_mask.astype(int)

    to_ecl, since_ecl = _signed_days_to_events(idx, eclipse_mask)
    cols["days_to_eclipse"] = to_ecl
    cols["days_since_eclipse"] = since_ecl
    nearest = np.fmin(np.nan_to_num(to_ecl, nan=1e6), np.nan_to_num(since_ecl, nan=1e6))
    cols["days_from_eclipse"] = np.where(nearest >= 1e6, np.nan, nearest)
    for w in ECLIPSE_WINDOWS:
        cols[f"eclipse_window_{w}d"] = (np.nan_to_num(nearest, nan=1e6) <= w).astype(int)
    # Smooth intensity ramp: peaks on the eclipse, decays over a fortnight.
    cols["grahana_intensity"] = np.exp(-np.nan_to_num(nearest, nan=1e6) / 5.0)

    # Sutak: the classical abstention window before an eclipse (12h solar, 9h lunar,
    # extended here to the trading day before, when transactions are traditionally paused).
    cols["sutak_window"] = (np.nan_to_num(to_ecl, nan=1e6) <= 1).astype(int)

    # --- Gochara: slow-graha ingress ----------------------------------------------------
    for g in INGRESS_GRAHAS:
        rashi = ay.rashi_index(positions[f"{g}_lon"].to_numpy())
        mask = _step_change_mask(rashi)
        cols[f"{g}_ingress"] = mask.astype(int)
        to_i, since_i = _signed_days_to_events(idx, mask)
        cols[f"{g}_days_to_ingress"] = to_i
        cols[f"{g}_days_since_ingress"] = since_i
        # Position within the current transit, 0.0 at entry -> 1.0 at exit.
        span = np.nan_to_num(to_i, nan=0.0) + np.nan_to_num(since_i, nan=0.0)
        cols[f"{g}_transit_progress"] = np.where(
            span > 0, np.nan_to_num(since_i, nan=0.0) / np.maximum(span, 1.0), np.nan
        )
        cols[f"{g}_ingress_window"] = (np.nan_to_num(since_i, nan=1e6) <= 10).astype(int)

    # --- Stambhana ramps for the retrograde grahas ---------------------------------------
    for g in ["Budha", "Shukra", "Mangala", "Guru", "Shani"]:
        speed = positions[f"{g}_speed"].to_numpy()
        sign_change = np.zeros(n, dtype=bool)
        sign_change[1:] = np.sign(speed[1:]) != np.sign(speed[:-1])
        cols[f"{g}_station"] = sign_change.astype(int)
        to_s, since_s = _signed_days_to_events(idx, sign_change)
        near = np.fmin(np.nan_to_num(to_s, nan=1e6), np.nan_to_num(since_s, nan=1e6))
        cols[f"{g}_days_from_station"] = np.where(near >= 1e6, np.nan, near)
        # Stations are the strongest turning-point candidates - give them a sharp ramp.
        cols[f"{g}_station_intensity"] = np.exp(-np.nan_to_num(near, nan=1e6) / 3.0)

    # --- Mahasamyoga: the Guru-Shani Great Conjunction -----------------------------------
    # The ~19.86 year cycle classical texts read as the business-cycle clock.
    gs_sep = ay.angular_separation(
        positions["Guru_lon"].to_numpy(), positions["Shani_lon"].to_numpy()
    )
    cols["guru_shani_sep"] = gs_sep
    cols["guru_shani_phase"] = gs_sep / 180.0
    cols["great_conjunction_proximity"] = np.clip(1.0 - gs_sep / 30.0, 0.0, 1.0)
    for name, arr in ay.cyclic_encode(
        np.mod(positions["Guru_lon"].to_numpy() - positions["Shani_lon"].to_numpy(), 360.0),
        360.0, harmonics=2, prefix="gurushani_",
    ).items():
        cols[name] = arr

    # --- Nodal cycle (18.6 years) ---------------------------------------------------------
    for name, arr in ay.cyclic_encode(
        positions["Rahu_lon"].to_numpy(), 360.0, harmonics=2, prefix="rahu_cycle_"
    ).items():
        cols[name] = arr

    # --- Aggregate event stress -------------------------------------------------------------
    cols["event_stress"] = (
        0.45 * cols["grahana_intensity"]
        + 0.20 * cols["Budha_station_intensity"]
        + 0.20 * cols["Shani_station_intensity"]
        + 0.15 * cols["great_conjunction_proximity"]
    )

    return pd.DataFrame(cols, index=idx)
