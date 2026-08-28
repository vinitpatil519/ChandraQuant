"""Astronomical correctness - checked against external ground truth, not self-consistency.

These are the tests that matter. A Jyotisha engine that is internally consistent but
astronomically wrong would sail through any amount of unit testing while producing
nonsense, so every assertion here is anchored to something published outside this repo:
the NASA eclipse catalogue, Drik Panchang, the classical Ashtakavarga totals, or a
physical identity (the ascendant equals the Sun's longitude at sunrise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chandraquant.astro import ashtakavarga, ayanamsa as ay, dasha, ephemeris as ep
from chandraquant.astro import events, grahas, lunar, natal, panchanga, solar
from chandraquant.astro.natal import ascendant_tropical, natal_chart
from chandraquant.config import TICKER_KEYS, ashtakavarga_cfg, nakshatra_cfg


@pytest.fixture(scope="module")
def positions():
    dates = pd.date_range("2018-01-01", "2026-12-31", freq="D")
    return ep.graha_positions(dates)


# --------------------------------------------------------------------------------
# Ayanamsa
# --------------------------------------------------------------------------------
def test_lahiri_anchor():
    """Lahiri is defined as 23 deg 15' 00.65 at 1956-01-01 (JD 2435553.5)."""
    assert ay.lahiri_ayanamsa(2435553.5) == pytest.approx(23.250182, abs=1e-4)


def test_ayanamsa_current_epoch():
    """~24 deg 14' in 2026 - the value every Indian panchanga prints."""
    jd_2026 = 2461280.0  # 2026-08-28
    assert ay.lahiri_ayanamsa(jd_2026) == pytest.approx(24.234, abs=0.01)


def test_ayanamsa_increases_monotonically():
    jd = np.linspace(2415020.0, 2488070.0, 500)  # 1900 to 2100
    assert np.all(np.diff(ay.lahiri_ayanamsa(jd)) > 0)


# --------------------------------------------------------------------------------
# Solar ingress
# --------------------------------------------------------------------------------
def test_makar_sankranti_2025():
    """Makar Sankranti 2025 fell on 14 January: Surya enters sidereal Makara (270 deg)."""
    pos = ep.graha_positions(pd.to_datetime(["2025-01-14"]), use_cache=False)
    sun = float(pos["Surya_lon"].iloc[0])
    assert sun == pytest.approx(270.0, abs=0.2), f"Surya at {sun}, expected ~270"
    assert ay.rashi_index(sun) == 9  # Makara


def test_solar_year_completes():
    """The Sun returns to its own longitude after a sidereal year."""
    a = ep.graha_positions(pd.to_datetime(["2020-03-01"]), use_cache=False)["Surya_lon"].iloc[0]
    b = ep.graha_positions(pd.to_datetime(["2021-03-01"]), use_cache=False)["Surya_lon"].iloc[0]
    assert abs(((a - b + 180) % 360) - 180) < 1.2


# --------------------------------------------------------------------------------
# Panchanga vs Drik Panchang
# --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "date,tithi_idx,nak_idx,label",
    [
        # Diwali 2024: Krishna Chaturdashi, Chitra nakshatra at the 09:15 IST reading.
        ("2024-10-31", 28, 13, "Diwali 2024"),
        # Diwali 2025: Krishna Chaturdashi, Hasta nakshatra.
        ("2025-10-20", 28, 12, "Diwali 2025"),
    ],
)
def test_panchanga_known_dates(date, tithi_idx, nak_idx, label):
    pos = ep.graha_positions(pd.to_datetime([date]), use_cache=False)
    p = panchanga.compute(pos)
    assert int(p["tithi_index"].iloc[0]) == tithi_idx, label
    assert int(p["nakshatra_index"].iloc[0]) == nak_idx, label


def test_panchanga_frequencies(positions):
    """Structural frequencies must match the geometry that defines them."""
    p = panchanga.compute(positions)
    # Rikta tithis are 6 of 30.
    assert p["tithi_is_rikta"].mean() == pytest.approx(0.20, abs=0.03)
    # Nine of the 27 nitya yogas are malefic.
    assert p["yoga_is_malefic"].mean() == pytest.approx(1 / 3, abs=0.03)
    # Vishti occupies 7 of every 60 karana slots; daily sampling skews it slightly.
    assert 0.09 < p["karana_is_vishti"].mean() < 0.16
    # Every state must actually occur over nine years.
    assert p["nakshatra_index"].nunique() == 27
    assert p["tithi_index"].nunique() == 30
    assert p["karana_index"].nunique() == 11


def test_tithi_elongation_consistency(positions):
    p = panchanga.compute(positions)
    elong = np.mod(positions["Chandra_lon"] - positions["Surya_lon"], 360.0)
    assert np.all(p["tithi_index"].to_numpy() == (elong // 12).astype(int))


# --------------------------------------------------------------------------------
# Eclipses vs the NASA catalogue
# --------------------------------------------------------------------------------
NASA_ECLIPSES_2025_26 = [
    "2025-03-14", "2025-03-29", "2025-09-07", "2025-09-21",
    "2026-02-17", "2026-03-03", "2026-08-12", "2026-08-28",
]


def test_eclipses_match_nasa_catalogue(positions):
    """Every catalogued 2025-26 eclipse detected, on the exact date, with no extras."""
    detected = events.detect_eclipses(positions.index, positions)
    got = {str(d.date()) for d in detected.loc["2025":"2026"].index}
    expected = set(NASA_ECLIPSES_2025_26)
    assert got == expected, f"missing {expected - got}, spurious {got - expected}"


def test_eclipse_rate_is_physical(positions):
    """Between 4 and 7 eclipses a year is the astronomical range."""
    detected = events.detect_eclipses(positions.index, positions)
    per_year = len(detected) / 9.0
    assert 4.0 <= per_year <= 7.0, f"{per_year:.2f} eclipses per year"


# --------------------------------------------------------------------------------
# Graha motion
# --------------------------------------------------------------------------------
def test_mercury_retrograde_2025(positions):
    """Budha retrogrades three times in 2025, near 15 Mar, 18 Jul and 9 Nov."""
    g = grahas.compute(positions)
    vakri = g.loc["2025", "Budha_vakri"]
    starts = vakri[vakri.diff() == 1].index
    assert len(starts) == 3
    for actual, expected in zip(starts, ["2025-03-15", "2025-07-18", "2025-11-09"]):
        assert abs((actual - pd.Timestamp(expected)).days) <= 2


def test_saturn_retrograde_duration(positions):
    """Shani is retrograde ~140 days a year."""
    g = grahas.compute(positions)
    assert 125 <= int(g.loc["2025", "Shani_vakri"].sum()) <= 150


def test_great_conjunction_2020(positions):
    """Guru and Shani conjoined on 2020-12-21, the closest in 400 years."""
    sep = np.abs(
        ((positions["Guru_lon"] - positions["Shani_lon"] + 180) % 360) - 180
    )
    when = sep.loc["2020"].idxmin()
    assert abs((when - pd.Timestamp("2020-12-21")).days) <= 2
    assert sep.min() < 0.5


def test_nodes_are_always_retrograde(positions):
    assert np.all(positions["Rahu_speed"] < 0)
    assert np.all(positions["Ketu_speed"] < 0)


def test_ketu_opposes_rahu(positions):
    diff = np.mod(positions["Ketu_lon"] - positions["Rahu_lon"], 360.0)
    assert np.allclose(diff, 180.0, atol=1e-6)


def test_moon_speed_range(positions):
    """The Moon runs 11.8-15.4 deg/day; anything outside is a bug."""
    s = positions["Chandra_speed"]
    assert s.min() > 11.0 and s.max() < 16.0


# --------------------------------------------------------------------------------
# Ascendant
# --------------------------------------------------------------------------------
def test_ascendant_equals_sun_at_sunrise():
    """A physical identity: at sunrise the Sun sits on the eastern horizon.

    The residual is the ~0.85 deg refraction-plus-semidiameter offset that defines
    sunrise, so agreement to about a degree is exactly right.
    """
    from skyfield import almanac
    from skyfield.api import wgs84

    from chandraquant.astro.ephemeris import _kernel, _timescale

    ts, eph = _timescale(), _kernel()
    loc = wgs84.latlon(18.9388, 72.8354)
    t0, t1 = ts.utc(2025, 6, 15), ts.utc(2025, 6, 16)
    times, kinds = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, loc))
    sunrise = times[list(kinds).index(1)]

    lum = ep.luminaries_at_utc([pd.Timestamp(sunrise.utc_iso())])
    asc = float(ascendant_tropical(float(sunrise.ut1), 18.9388, 72.8354))
    diff = abs(((float(lum["sun_trop"].iloc[0]) - asc + 180) % 360) - 180)
    assert diff < 1.5, f"ascendant off by {diff:.3f} deg at sunrise"


def test_ascendant_completes_a_circuit():
    """The Lagna advances through all twelve rashis in a day."""
    jds = 2460000.0 + np.linspace(0, 1, 200)
    asc = ascendant_tropical(jds, 18.9388, 72.8354)
    assert len(np.unique((asc // 30).astype(int))) == 12


# --------------------------------------------------------------------------------
# Ashtakavarga
# --------------------------------------------------------------------------------
def test_bhinnashtakavarga_totals():
    """The classical per-graha bindu totals: 48/49/39/54/56/52/39."""
    cfg = ashtakavarga_cfg()
    for graha, expected in cfg["expected_totals"].items():
        total = sum(len(v) for v in cfg["tables"][graha].values())
        assert total == expected, f"{graha} has {total} bindus, expected {expected}"


def test_sarvashtakavarga_total_is_337():
    cfg = ashtakavarga_cfg()
    total = sum(
        len(v) for g in cfg["expected_totals"] for v in cfg["tables"][g].values()
    )
    assert total == 337


@pytest.mark.parametrize("ticker", TICKER_KEYS)
def test_sav_per_chart_sums_to_337(ticker):
    charts = ashtakavarga.compute_charts(ticker)
    assert int(charts["SAV"].sum()) == 337
    assert charts["SAV"].shape == (12,)


# --------------------------------------------------------------------------------
# Vimshottari
# --------------------------------------------------------------------------------
def test_vimshottari_totals_120_years():
    cfg = nakshatra_cfg()["vimshottari"]
    assert sum(cfg["years"].values()) == 120
    assert len(cfg["order"]) == 9


def test_nakshatra_lords_follow_vimshottari_cycle():
    cfg = nakshatra_cfg()
    lords = [n["lord"] for n in cfg["nakshatras"]]
    assert lords == list(cfg["vimshottari"]["order"]) * 3


@pytest.mark.parametrize("ticker", TICKER_KEYS)
def test_dasha_chain_is_continuous(ticker):
    chain = dasha.build_chain(ticker)
    gaps = (chain["start"].shift(-1) - chain["end"]).dropna()
    assert gaps.abs().max() < pd.Timedelta(seconds=2)


@pytest.mark.parametrize("ticker", TICKER_KEYS)
def test_dasha_covers_today(ticker):
    d = dasha.describe(ticker, "2026-08-28")
    assert d and d["md"] and d["ad"]


# --------------------------------------------------------------------------------
# Natal charts and ticker specificity - the v1 bug regression test
# --------------------------------------------------------------------------------
def test_natal_charts_differ_between_tickers():
    charts = {k: natal_chart(k) for k in TICKER_KEYS}
    moons = {k: c["moon_nakshatra"] for k, c in charts.items()}
    assert len(set(moons.values())) == len(TICKER_KEYS), f"natal Moons collide: {moons}"


def test_astro_features_are_ticker_specific():
    """THE regression test for v1's fatal flaw.

    In v1 every astro feature was a function of calendar date alone, so all three
    indices received byte-identical astro vectors and the astro block could not
    possibly discriminate between them. If this test ever fails, that bug is back.
    """
    from chandraquant.astro import engine as astro_engine

    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    mats = {k: astro_engine.numeric_matrix(astro_engine.build(dates, k)) for k in TICKER_KEYS}
    common = sorted(set.intersection(*(set(m.columns) for m in mats.values())))
    a, b = mats["NIFTY"][common], mats["BANKNIFTY"][common]
    differing = int(((a - b).abs().mean() > 1e-9).sum())
    assert differing > 100, f"only {differing} of {len(common)} astro features differ"


# --------------------------------------------------------------------------------
# Determinism and leak-freedom
# --------------------------------------------------------------------------------
def test_astro_is_deterministic():
    """The same date must always produce the same sky."""
    d = pd.to_datetime(["2023-06-15"])
    a = ep.graha_positions(d, use_cache=False)
    b = ep.graha_positions(d, use_cache=False)
    pd.testing.assert_frame_equal(a, b)


def test_astro_is_computable_for_the_future():
    """Astro features are functions of time alone, so the forward calendar is real."""
    future = pd.date_range("2030-01-01", "2030-03-01", freq="D")
    from chandraquant.astro import engine as astro_engine

    m = astro_engine.build(future, "NIFTY")
    assert len(m) == len(future)
    assert m["nakshatra_index"].notna().all()


def test_astro_windowing_does_not_change_values():
    """A feature at date t must not depend on which window it was computed in."""
    from chandraquant.astro import engine as astro_engine

    wide = astro_engine.build(pd.date_range("2020-01-01", "2024-12-31", freq="D"), "NIFTY")
    narrow = astro_engine.build(pd.date_range("2021-01-01", "2023-12-31", freq="D"), "NIFTY")
    probe = "2022-06-15"
    for col in ("nakshatra_index", "tithi_index", "nat_tarabala", "Guru_lon", "dasha_md_idx"):
        assert wide.loc[probe, col] == narrow.loc[probe, col], col
