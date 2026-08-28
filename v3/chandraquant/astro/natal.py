"""Mundane natal charts - the feature that makes astrology ticker-specific.

THIS MODULE IS THE POINT OF THE WHOLE PROJECT.

ChandraQuant v1 fed the model ten astro columns that were functions of calendar date
alone: Moon longitude, Sun longitude, Tithi, Nakshatra and their harmonics. On any
given day NIFTY, BANKNIFTY and SENSEX therefore received *byte-identical* astro
vectors while carrying three different labels. Pooled into one training set, the astro
block was mathematically incapable of discriminating between the indices - it could
only add variance. That is why v1 measured a NEGATIVE lift (dAUC -0.018): the hybrid
model was worse than the technical model alone.

Mundane (foundation) astrology fixes this at the root. An index is born the day it
begins publishing; cast a chart for that moment and place, and every subsequent
celestial position can be expressed as a transit RELATIVE TO THAT CHART. NIFTY's natal
Moon sits in a different nakshatra from BANKNIFTY's, so on the same calendar day the
same transiting Moon is in Sampat tara for one index and Vipat for the other.

That single change unlocks the entire classical apparatus v1 never touched:
  Gochara       transits counted from the natal Moon and natal Lagna
  Tarabala      the 9-fold auspiciousness cycle from the natal Moon's nakshatra
  Chandrabala   the transit Moon's rashi relation to the natal Moon
  Sade Sati     Shani's 7.5-year passage over the 12th, 1st and 2nd from natal Moon
  Bhava         mundane house activation - 5th speculation, 8th crisis, 11th gains
  Vimshottari   the dasha chain, which needs the natal Moon's nakshatra (see dasha.py)
"""

from __future__ import annotations

import functools
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz

from ..config import graha_cfg, get_ticker, nakshatra_cfg
from . import ayanamsa as ay
from .ephemeris import NAVAGRAHA, graha_positions

OBLIQUITY_DEG = 23.4392911

BENEFIC_GRAHAS = ["Guru", "Shukra", "Chandra"]
MALEFIC_GRAHAS = ["Shani", "Mangala", "Rahu", "Ketu", "Surya"]

# Transit grahas worth tracking against the natal chart. Chandra changes daily and
# carries the short-cycle signal; Guru / Shani / Rahu define the macro bands.
GOCHARA_GRAHAS = ["Chandra", "Surya", "Mangala", "Budha", "Guru", "Shukra", "Shani", "Rahu", "Ketu"]

# Classical Gochara: houses from the natal Moon in which each graha gives good results.
# (Brihat Parashara Hora Shastra, Gochara Phala chapter.)
GOCHARA_FAVOURABLE = {
    "Surya": [3, 6, 10, 11],
    "Chandra": [1, 3, 6, 7, 10, 11],
    "Mangala": [3, 6, 11],
    "Budha": [2, 4, 6, 8, 10, 11],
    "Guru": [2, 5, 7, 9, 11],
    "Shukra": [1, 2, 3, 4, 5, 8, 9, 11, 12],
    "Shani": [3, 6, 11],
    "Rahu": [3, 6, 10, 11],
    "Ketu": [3, 6, 10, 11],
}


def gmst_degrees(jd_ut: np.ndarray | float) -> np.ndarray | float:
    """Greenwich Mean Sidereal Time in degrees (IAU 1982 series)."""
    jd = np.asarray(jd_ut, dtype=float)
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T**2
        - T**3 / 38710000.0
    )
    return np.mod(gmst, 360.0)


def ascendant_tropical(jd_ut, latitude_deg: float, longitude_deg: float):
    """Tropical ecliptic longitude of the rising point (Lagna), degrees.

    Sanity check that this is right: at local sunrise the Sun sits on the eastern
    horizon, so the ascendant must equal the Sun's longitude. tests/test_natal.py
    asserts exactly that.
    """
    lst = np.mod(gmst_degrees(jd_ut) + longitude_deg, 360.0)
    theta = np.radians(lst)
    eps = np.radians(OBLIQUITY_DEG)
    phi = np.radians(latitude_deg)
    asc = np.arctan2(
        np.cos(theta),
        -(np.sin(theta) * np.cos(eps) + np.tan(phi) * np.sin(eps)),
    )
    return np.mod(np.degrees(asc), 360.0)


def _natal_datetime_utc(entry: dict) -> pd.Timestamp:
    source = entry.get("natal_source", "launch")
    date_str = entry["natal"][source]
    hh, mm = (int(x) for x in entry["natal_time"].split(":"))
    zone = pytz.timezone(entry["timezone"])
    local = zone.localize(
        datetime.combine(pd.Timestamp(date_str).date(), dtime(hh, mm))
    )
    return pd.Timestamp(local).tz_convert("UTC")


@functools.lru_cache(maxsize=16)
def natal_chart(ticker_key: str) -> dict:
    """Cast the mundane natal chart for an index.

    Returns sidereal longitudes for the Lagna and all nine grahas, plus the derived
    quantities the transit engine and the dasha engine need.
    """
    entry = get_ticker(ticker_key)
    utc = _natal_datetime_utc(entry)

    # Ephemeris at the natal instant.
    pos = graha_positions(
        [utc.tz_convert(entry["timezone"]).normalize()],
        local_time=entry["natal_time"],
        tz=entry["timezone"],
        use_cache=False,
    ).iloc[0]

    jd_ut = float(pos["jd_tt"]) - 69.2 / 86400.0  # TT -> UT1, adequate at this precision
    ayan = float(pos["ayanamsa"])
    asc_trop = float(ascendant_tropical(jd_ut, entry["latitude"], entry["longitude"]))
    lagna = float(np.mod(asc_trop - ayan, 360.0))

    grahas = {g: float(pos[f"{g}_lon"]) for g in NAVAGRAHA}
    moon = grahas["Chandra"]
    nak_table = nakshatra_cfg()["nakshatras"]
    moon_nak = int(ay.nakshatra_index(moon))

    return {
        "key": entry["key"],
        "name": entry["name"],
        "display": entry["display"],
        "natal_utc": utc,
        "natal_local": utc.tz_convert(entry["timezone"]),
        "natal_source": entry.get("natal_source", "launch"),
        "latitude": entry["latitude"],
        "longitude": entry["longitude"],
        "ayanamsa": ayan,
        "lagna": lagna,
        "lagna_rashi": int(ay.rashi_index(lagna)),
        "lagna_nakshatra": int(ay.nakshatra_index(lagna)),
        "grahas": grahas,
        "moon": moon,
        "moon_rashi": int(ay.rashi_index(moon)),
        "moon_nakshatra": moon_nak,
        "moon_nakshatra_name": nak_table[moon_nak]["display"],
        "moon_nakshatra_lord": nak_table[moon_nak]["lord"],
        "moon_nak_fraction": float(ay.nakshatra_fraction(moon)),
        "sun_rashi": int(ay.rashi_index(grahas["Surya"])),
    }


def compute(positions: pd.DataFrame, ticker_key: str) -> pd.DataFrame:
    """Transit-relative-to-natal features for one index.

    Every column here differs between tickers on the same calendar day. That is the
    whole point.
    """
    chart = natal_chart(ticker_key)
    gcfg = graha_cfg()
    ncfg = nakshatra_cfg()
    idx = positions.index
    n = len(idx)
    cols: dict[str, np.ndarray] = {}

    natal_moon = chart["moon"]
    natal_lagna = chart["lagna"]
    natal_moon_nak = chart["moon_nakshatra"]

    # --- Gochara: transits counted from the natal Moon and natal Lagna ---------------
    house_from_moon: dict[str, np.ndarray] = {}
    for g in GOCHARA_GRAHAS:
        lon = positions[f"{g}_lon"].to_numpy()
        h_moon = ay.house_from(natal_moon, lon)
        h_lagna = ay.house_from(natal_lagna, lon)
        house_from_moon[g] = h_moon
        cols[f"nat_{g}_house_moon"] = h_moon
        cols[f"nat_{g}_house_lagna"] = h_lagna
        # Classical Gochara Phala: is this graha in a house where it gives results?
        cols[f"nat_{g}_gochara_good"] = np.isin(h_moon, GOCHARA_FAVOURABLE[g]).astype(int)
        for name, arr in ay.cyclic_encode(
            h_lagna - 1, 12.0, harmonics=1, prefix=f"nat_{g}_lagna_"
        ).items():
            cols[name] = arr

    # --- Bhava activation: mundane house meanings weighted by occupant nature --------
    bhava = gcfg["bhava_market_meaning"]
    activation = np.zeros(n)
    for house in range(1, 13):
        occ_benefic = np.zeros(n)
        occ_malefic = np.zeros(n)
        for g in GOCHARA_GRAHAS:
            hit = (cols[f"nat_{g}_house_lagna"] == house).astype(float)
            if g in BENEFIC_GRAHAS:
                occ_benefic += hit
            elif g in MALEFIC_GRAHAS:
                occ_malefic += hit
        cols[f"nat_bhava{house}_benefic"] = occ_benefic
        cols[f"nat_bhava{house}_malefic"] = occ_malefic
        net = occ_benefic - occ_malefic
        cols[f"nat_bhava{house}_net"] = net
        activation += float(bhava[house]["bias"]) * net
    cols["nat_bhava_activation"] = activation

    # The two houses that matter most for a market chart.
    cols["nat_speculation_house"] = cols["nat_bhava5_net"]   # Poorva Punya - speculation
    cols["nat_gains_house"] = cols["nat_bhava11_net"]        # Labha - realised gains
    cols["nat_crisis_house"] = cols["nat_bhava8_net"]        # Randhra - sudden loss
    cols["nat_loss_house"] = cols["nat_bhava12_net"]         # Vyaya - outflows

    # --- Tarabala: the 9-fold cycle from the natal Moon's nakshatra -------------------
    transit_nak = ay.nakshatra_index(positions["Chandra_lon"].to_numpy())
    count_from_natal = ((transit_nak - natal_moon_nak) % 27) + 1
    tara = ((count_from_natal - 1) % 9) + 1
    cols["nat_tara_count"] = count_from_natal
    cols["nat_tarabala"] = tara
    tara_quality = {t["index"]: t["quality"] for t in ncfg["tarabala"]}
    cols["nat_tarabala_quality"] = np.array([tara_quality[int(t)] for t in tara])
    cols["nat_tara_favourable"] = (cols["nat_tarabala_quality"] > 0).astype(int)
    for name, arr in ay.cyclic_encode(tara - 1, 9.0, harmonics=2, prefix="nat_tara_").items():
        cols[name] = arr
    # The 27-step position in the natal-relative lunar cycle.
    for name, arr in ay.cyclic_encode(
        count_from_natal - 1, 27.0, harmonics=2, prefix="nat_nakcycle_"
    ).items():
        cols[name] = arr

    # --- Chandrabala and Chandrashtama -------------------------------------------------
    moon_house = house_from_moon["Chandra"]
    cols["nat_chandra_house"] = moon_house
    cols["nat_chandrabala"] = np.isin(
        moon_house, gcfg["chandrabala_favourable"]
    ).astype(int)
    cols["nat_chandrashtama"] = (moon_house == 8).astype(int)

    # --- Sade Sati and the other Shani afflictions ---------------------------------------
    shani_house = house_from_moon["Shani"]
    ss_cfg = gcfg["sade_sati"]
    cols["nat_shani_house_moon"] = shani_house
    cols["nat_sade_sati"] = np.isin(shani_house, ss_cfg["houses"]).astype(int)
    # 0 = rising (12th), 1 = peak (1st), 2 = setting (2nd), NaN outside.
    phase = np.full(n, np.nan)
    for i, h in enumerate(ss_cfg["houses"]):
        phase[shani_house == h] = i
    cols["nat_sade_sati_phase"] = phase
    cols["nat_kantaka_shani"] = np.isin(
        shani_house, gcfg["kantaka_shani_houses"]
    ).astype(int)
    cols["nat_ashtama_shani"] = (shani_house == int(gcfg["ashtama_shani_house"])).astype(int)

    # Guru's transit is the classical counterweight to Shani's.
    guru_house = house_from_moon["Guru"]
    cols["nat_guru_house_moon"] = guru_house
    cols["nat_guru_favourable"] = np.isin(guru_house, [2, 5, 7, 9, 11]).astype(int)
    cols["nat_guru_kendra_moon"] = np.isin(guru_house, [1, 4, 7, 10]).astype(int)

    # --- Transit hits on natal points ------------------------------------------------------
    # A transiting graha conjunct a natal graha within orb is the classical trigger.
    orb = 5.0
    total_hits = np.zeros(n)
    benefic_hits = np.zeros(n)
    malefic_hits = np.zeros(n)
    for t_graha in GOCHARA_GRAHAS:
        t_lon = positions[f"{t_graha}_lon"].to_numpy()
        graha_hits = np.zeros(n)
        for n_graha, n_lon in chart["grahas"].items():
            sep = ay.angular_separation(t_lon, n_lon)
            strength = np.clip(1.0 - sep / orb, 0.0, 1.0)
            graha_hits += strength
        cols[f"nat_{t_graha}_hits"] = graha_hits
        total_hits += graha_hits
        if t_graha in BENEFIC_GRAHAS:
            benefic_hits += graha_hits
        elif t_graha in MALEFIC_GRAHAS:
            malefic_hits += graha_hits
    cols["nat_total_hits"] = total_hits
    cols["nat_benefic_hits"] = benefic_hits
    cols["nat_malefic_hits"] = malefic_hits
    cols["nat_net_hits"] = benefic_hits - malefic_hits

    # Conjunctions with the two most sensitive natal points.
    for label, point in (("lagna", natal_lagna), ("moon", natal_moon)):
        ben = np.zeros(n)
        mal = np.zeros(n)
        for g in GOCHARA_GRAHAS:
            sep = ay.angular_separation(positions[f"{g}_lon"].to_numpy(), point)
            s = np.clip(1.0 - sep / 8.0, 0.0, 1.0)
            if g in BENEFIC_GRAHAS:
                ben += s
            elif g in MALEFIC_GRAHAS:
                mal += s
        cols[f"nat_{label}_benefic_transit"] = ben
        cols[f"nat_{label}_malefic_transit"] = mal
        cols[f"nat_{label}_net_transit"] = ben - mal

    # --- Composite natal-relative favourability ---------------------------------------------
    cols["nat_gochara_score"] = sum(
        cols[f"nat_{g}_gochara_good"] for g in GOCHARA_GRAHAS
    ) / len(GOCHARA_GRAHAS)

    return pd.DataFrame(cols, index=idx)
