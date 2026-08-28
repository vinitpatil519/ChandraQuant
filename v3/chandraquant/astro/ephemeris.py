"""Skyfield / JPL DE440s wrapper producing sidereal graha positions.

Everything downstream in the Jyotisha engine reads from :func:`graha_positions`.
Positions are *apparent* geocentric ecliptic coordinates of date (light-time and
aberration corrected, matching Indian panchanga convention), then converted to the
Lahiri sidereal zodiac.

Rahu and Ketu are the lunar nodes. Indian panchangas conventionally use the MEAN
node, so that is the default; the osculating true node is available for comparison.
"""

from __future__ import annotations

import functools
import hashlib
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
from skyfield.api import Loader
from skyfield.framelib import ecliptic_frame

from ..config import CACHE_DIR, EPHEM_DIR
from . import ayanamsa as ay

EPHEM_FILE = "de440s.bsp"

# A complete de440s.bsp is ~31.2 MB. If the download is interrupted, Skyfield finalises
# whatever arrived, then reuses that truncated file forever - and jplephem fails on it
# with "buffer is too small for requested array", which tells the user nothing. Anything
# materially under the real size is treated as a partial download and re-fetched.
MIN_EPHEM_BYTES = 25_000_000

# Vedic name -> DE440s target. Barycentres are used for the outer planets, which is
# standard practice and well inside a milli-degree for our purposes.
GRAHA_TARGETS = {
    "Surya": "sun",
    "Chandra": "moon",
    "Budha": "mercury barycenter",
    "Shukra": "venus barycenter",
    "Mangala": "mars barycenter",
    "Guru": "jupiter barycenter",
    "Shani": "saturn barycenter",
}

# Slow outers, carried only for the Kala-Taranga aspect composite (not classical grahas).
OUTER_TARGETS = {
    "Aruna": "uranus barycenter",
    "Varuna": "neptune barycenter",
}

# The nine classical grahas, in the traditional order.
NAVAGRAHA = ["Surya", "Chandra", "Mangala", "Budha", "Guru", "Shukra", "Shani", "Rahu", "Ketu"]

# Grahas whose positions come straight from the ephemeris (nodes are computed).
EPHEMERIS_GRAHAS = list(GRAHA_TARGETS)


@functools.lru_cache(maxsize=1)
def _loader() -> Loader:
    return Loader(str(EPHEM_DIR), verbose=False)


@functools.lru_cache(maxsize=1)
def _kernel():
    """The DE440s kernel. Downloaded on first use (~32 MB, covers 1849-2150).

    The first call on a fresh checkout fetches the ephemeris from NASA, which can take
    anywhere from thirty seconds to several minutes depending on the link. Skyfield is
    silent by default, so without this notice the app looks hung on its very first run -
    the worst possible first impression. Announce it, and show Skyfield's progress bar.
    """
    path = EPHEM_DIR / EPHEM_FILE

    # Self-heal a partial download rather than failing on it every run afterwards.
    if path.exists() and path.stat().st_size < MIN_EPHEM_BYTES:
        size_mb = path.stat().st_size / 1_048_576
        print(
            f"\n  {EPHEM_FILE} is only {size_mb:.1f} MB - an interrupted download.\n"
            f"  Removing it and fetching again.\n",
            file=sys.stderr,
        )
        try:
            path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Found a truncated ephemeris at {path} ({size_mb:.1f} MB, expected ~31 MB) "
                f"and could not delete it: {exc}. Delete the file manually and re-run."
            ) from exc

    if not path.exists():
        print(
            f"\n  First run: downloading the NASA JPL {EPHEM_FILE} ephemeris (~32 MB).\n"
            f"  This happens once. Everything afterwards runs offline.\n",
            file=sys.stderr,
        )
        return Loader(str(EPHEM_DIR), verbose=True)(EPHEM_FILE)

    try:
        return _loader()(EPHEM_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the ephemeris at {path}. It is most likely corrupt or "
            f"incomplete. Delete it and re-run `python scripts/refresh_data.py --all` "
            f"to fetch a fresh copy. Underlying error: {exc}"
        ) from exc


@functools.lru_cache(maxsize=1)
def _timescale():
    return _loader().timescale()


def _to_utc_index(dates: pd.DatetimeIndex, local_time: str, tz: str) -> pd.DatetimeIndex:
    """Stamp naive dates with a local clock time and convert to UTC."""
    hh, mm = (int(x) for x in local_time.split(":"))
    zone = pytz.timezone(tz)
    naive = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    stamped = [zone.localize(datetime.combine(d.date(), dtime(hh, mm))) for d in naive]
    return pd.DatetimeIndex(stamped).tz_convert("UTC")


def _skyfield_time(utc_index: pd.DatetimeIndex):
    ts = _timescale()
    return ts.from_datetimes(list(utc_index.to_pydatetime()))


def mean_node_longitude(jd_tt: np.ndarray) -> np.ndarray:
    """Mean longitude of the Moon's ascending node (Rahu), tropical, degrees.

    Meeus, *Astronomical Algorithms*, ch. 47.
    """
    T = (np.asarray(jd_tt, dtype=float) - 2451545.0) / 36525.0
    omega = (
        125.0445479
        - 1934.1362891 * T
        + 0.0020754 * T**2
        + T**3 / 467441.0
        - T**4 / 60616000.0
    )
    return np.mod(omega, 360.0)


def _true_node_longitude(t) -> np.ndarray:
    """Osculating (true) ascending node from the Moon's geocentric state vector."""
    kernel = _kernel()
    earth, moon = kernel["earth"], kernel["moon"]
    rel = (moon - earth).at(t)
    r = rel.position.au.T           # (n, 3) equatorial
    v = rel.velocity.au_per_d.T
    h = np.cross(r, v)              # orbital angular momentum
    # Rotate equatorial -> ecliptic of J2000 (obliquity is enough for a node angle).
    eps = np.radians(23.4392911)
    hy = h[:, 1] * np.cos(eps) + h[:, 2] * np.sin(eps)
    hx = h[:, 0]
    return np.mod(np.degrees(np.arctan2(hx, -hy)), 360.0)


def _ecliptic_lonlat(t, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apparent geocentric ecliptic (lon, lat, distance) of date, in degrees / AU."""
    kernel = _kernel()
    earth = kernel["earth"]
    body = kernel[target]
    astrometric = earth.at(t).observe(body).apparent()
    lat, lon, dist = astrometric.frame_latlon(ecliptic_frame)
    return lon.degrees, lat.degrees, dist.au


def _wrapped_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a - b wrapped to (-180, 180], so speed is well behaved across the 0/360 seam."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _cache_key(dates: pd.DatetimeIndex, local_time: str, tz: str, true_node: bool) -> str:
    payload = f"{dates[0]}|{dates[-1]}|{len(dates)}|{local_time}|{tz}|{true_node}"
    digest = hashlib.md5(payload.encode()).hexdigest()[:16]
    return f"ephem_{digest}.parquet"


def graha_positions(
    dates,
    local_time: str = "09:15",
    tz: str = "Asia/Kolkata",
    true_node: bool = False,
    include_outers: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Sidereal positions of the navagraha for each date.

    Returns a DataFrame indexed by the input dates with, per graha:
        ``{g}_lon``    sidereal ecliptic longitude, degrees [0, 360)
        ``{g}_trop``   tropical ecliptic longitude, degrees
        ``{g}_lat``    ecliptic latitude, degrees
        ``{g}_dist``   distance from Earth, AU
        ``{g}_speed``  daily motion in longitude, degrees/day (negative = vakri)
    plus ``ayanamsa`` and ``jd_tt``.

    Every value is a deterministic function of the timestamp, so these columns are
    leak-free by construction and computable arbitrarily far into the future.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    cache_path = CACHE_DIR / _cache_key(dates, local_time, tz, true_node)
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(dates):
            return cached

    utc = _to_utc_index(dates, local_time, tz)
    t = _skyfield_time(utc)
    # +/- half a day for central-difference speeds.
    t_minus = _skyfield_time(utc - pd.Timedelta(hours=12))
    t_plus = _skyfield_time(utc + pd.Timedelta(hours=12))

    jd_tt = np.asarray(t.tt, dtype=float)
    ayan = np.asarray(ay.lahiri_ayanamsa(jd_tt), dtype=float)

    out: dict[str, np.ndarray] = {"jd_tt": jd_tt, "ayanamsa": ayan}

    targets = dict(GRAHA_TARGETS)
    if include_outers:
        targets.update(OUTER_TARGETS)

    for graha, target in targets.items():
        lon, lat, dist = _ecliptic_lonlat(t, target)
        lon_m, _, _ = _ecliptic_lonlat(t_minus, target)
        lon_p, _, _ = _ecliptic_lonlat(t_plus, target)
        speed = _wrapped_delta(lon_p, lon_m)  # over a full day
        out[f"{graha}_trop"] = lon
        out[f"{graha}_lon"] = np.mod(lon - ayan, 360.0)
        out[f"{graha}_lat"] = lat
        out[f"{graha}_dist"] = dist
        out[f"{graha}_speed"] = speed

    # --- Rahu / Ketu ---------------------------------------------------------------
    if true_node:
        rahu_trop = _true_node_longitude(t)
        rahu_m = _true_node_longitude(t_minus)
        rahu_p = _true_node_longitude(t_plus)
    else:
        rahu_trop = mean_node_longitude(jd_tt)
        rahu_m = mean_node_longitude(np.asarray(t_minus.tt, dtype=float))
        rahu_p = mean_node_longitude(np.asarray(t_plus.tt, dtype=float))

    rahu_speed = _wrapped_delta(rahu_p, rahu_m)
    for name, offset in (("Rahu", 0.0), ("Ketu", 180.0)):
        trop = np.mod(rahu_trop + offset, 360.0)
        out[f"{name}_trop"] = trop
        out[f"{name}_lon"] = np.mod(trop - ayan, 360.0)
        out[f"{name}_lat"] = np.zeros_like(trop)
        out[f"{name}_dist"] = np.full_like(trop, np.nan)
        out[f"{name}_speed"] = rahu_speed

    df = pd.DataFrame(out, index=dates)
    df.index.name = "date"
    if use_cache:
        df.to_parquet(cache_path)
    return df


def position_at(when, local_time: str = "09:15", tz: str = "Asia/Kolkata", **kw) -> pd.Series:
    """Convenience: graha positions for a single moment, as a Series."""
    df = graha_positions([pd.Timestamp(when)], local_time=local_time, tz=tz, use_cache=False, **kw)
    return df.iloc[0]


def luminaries_at_utc(utc_timestamps) -> pd.DataFrame:
    """Sun and Moon longitude / latitude at arbitrary UTC instants.

    Used for sub-daily refinement (eclipse geometry needs the Moon's latitude at the
    exact syzygy, not at the following market open).
    """
    utc = pd.DatetimeIndex(pd.to_datetime(utc_timestamps, utc=True))
    if len(utc) == 0:
        return pd.DataFrame(
            columns=["sun_trop", "moon_trop", "moon_lat", "elongation"], index=utc
        )
    t = _skyfield_time(utc)
    sun_lon, _, _ = _ecliptic_lonlat(t, "sun")
    moon_lon, moon_lat, _ = _ecliptic_lonlat(t, "moon")
    return pd.DataFrame(
        {
            "sun_trop": sun_lon,
            "moon_trop": moon_lon,
            "moon_lat": moon_lat,
            "elongation": np.mod(moon_lon - sun_lon, 360.0),
        },
        index=utc,
    )
