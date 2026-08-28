"""The five branded composite indices - ChandraQuant's signature astro readouts.

Everything upstream produces raw Jyotisha quantities. This module collapses them into
five named indices that a human can actually read off a screen, and that the gating
model consumes directly. They are the numbers on the dashboard and the lines in the
Pine indicator.

  CBI  Chandra Bala      short-cycle sentiment - the Moon's condition relative to the
                         index's own natal chart. Moves daily.
  GSI  Graha Shakti      structural strength - Shadbala-weighted benefics minus
                         malefics. Moves over weeks.
  VRI  Vriddhi           expansion pressure - Guru and Shukra, the 5th and 11th bhavas,
                         and the dasha lord's character. Moves over months.
  BHY  Bhaya             panic risk - Shani, Rahu and Mangala stress, eclipse
                         proximity, gandanta, Vishti, malefic yogas. Spikes.
  KTW  Kala Taranga      the cosmic tide - a Bradley-siderograph-style weighted sum of
                         every aspect term, standardised. The overlay line.

All five are z-scored against an expanding window so that they are (a) comparable to
each other, (b) interpretable as standard deviations, and (c) FREE OF LOOKAHEAD - an
expanding window only ever uses the past. A full-sample z-score would leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PERIODS = 250


def _expanding_z(s: pd.Series, min_periods: int = MIN_PERIODS) -> pd.Series:
    """Z-score against an expanding window - causal, so it cannot leak the future."""
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    # Before the window fills, fall back to the raw centred value rather than NaN.
    return z.fillna(s - s.expanding(min_periods=1).mean()).fillna(0.0)


def _squash(z: pd.Series, scale: float = 2.0) -> pd.Series:
    """Map a z-score onto [-1, 1] smoothly so composites stay bounded."""
    return np.tanh(z / scale)


def compute(
    panchanga_df: pd.DataFrame,
    lunar_df: pd.DataFrame,
    solar_df: pd.DataFrame,
    grahas_df: pd.DataFrame,
    aspects_df: pd.DataFrame,
    events_df: pd.DataFrame,
    natal_df: pd.DataFrame,
    dasha_df: pd.DataFrame,
    av_df: pd.DataFrame,
    shadbala_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the five composite indices from the full astro feature set."""
    idx = panchanga_df.index
    out = pd.DataFrame(index=idx)

    # --- CBI: Chandra Bala Index -------------------------------------------------------
    # The Moon's condition, judged against this index's own natal chart. Tarabala and
    # Chandrabala are the natal-relative terms and carry the most weight - they are the
    # reason this index differs between NIFTY, BANKNIFTY and CNXIT.
    cbi_raw = (
        0.28 * natal_df["nat_tarabala_quality"]
        + 0.16 * natal_df["nat_chandrabala"]
        - 0.16 * natal_df["nat_chandrashtama"]
        + 0.14 * panchanga_df["tithi_five_fold_quality"]
        + 0.10 * panchanga_df["gana_score"]
        + 0.08 * panchanga_df["paksha_bias"]
        + 0.08 * (av_df["av_chandra_sav"] - 28.0) / 8.0
        - 0.20 * lunar_df["lunar_instability"]
        + 0.06 * lunar_df["moon_speed_norm"].clip(-2, 2)
    )
    out["CBI_raw"] = cbi_raw
    out["CBI"] = _squash(_expanding_z(cbi_raw))

    # --- GSI: Graha Shakti Index --------------------------------------------------------
    # Structural strength of the benefics against the malefics, from Shadbala.
    gsi_raw = (
        0.55 * shadbala_df["graha_shakti_raw"]
        + 0.20 * grahas_df["graha_balance"]
        + 0.15 * (av_df["av_net"] / 8.0)
        + 0.10 * shadbala_df["sb_n_sufficient"]
    )
    out["GSI_raw"] = gsi_raw
    out["GSI"] = _squash(_expanding_z(gsi_raw))

    # --- VRI: Vriddhi (expansion) Index --------------------------------------------------
    # Guru and Shukra strength, the speculation and gains bhavas, and the dasha era.
    vri_raw = (
        0.24 * shadbala_df["sb_Guru_ratio"]
        + 0.14 * shadbala_df["sb_Shukra_ratio"]
        + 0.16 * dasha_df["dasha_combined_bias"]
        + 0.12 * natal_df["nat_speculation_house"]
        + 0.12 * natal_df["nat_gains_house"]
        + 0.08 * natal_df["nat_guru_favourable"]
        + 0.08 * aspects_df["yoga_benefic_pressure"]
        + 0.06 * (av_df["av_guru_sav"] - 28.0) / 8.0
    )
    out["VRI_raw"] = vri_raw
    out["VRI"] = _squash(_expanding_z(vri_raw))

    # --- BHY: Bhaya (panic) Index ---------------------------------------------------------
    # The fear side. Deliberately spiky - it is a risk gate, not a trend line.
    bhy_raw = (
        0.22 * aspects_df["yoga_malefic_pressure"]
        + 0.18 * events_df["grahana_intensity"]
        + 0.12 * panchanga_df["karana_is_vishti"]
        + 0.10 * panchanga_df["yoga_is_malefic"]
        + 0.10 * lunar_df["gandanta_intensity"]
        + 0.08 * natal_df["nat_sade_sati"]
        + 0.08 * natal_df["nat_ashtama_shani"]
        + 0.06 * (-natal_df["nat_crisis_house"]).clip(lower=0)
        + 0.06 * grahas_df["n_debilitated"] / 3.0
        + 0.06 * events_df["Budha_station_intensity"]
        - 0.06 * shadbala_df["sb_Guru_ratio"]
    )
    out["BHY_raw"] = bhy_raw
    out["BHY"] = ((_squash(_expanding_z(bhy_raw)) + 1.0) / 2.0)  # 0..1, a risk level

    # --- KTW: Kala Taranga (the cosmic tide) -----------------------------------------------
    ktw_raw = (
        0.70 * aspects_df["kala_taranga"]
        + 0.20 * aspects_df["yoga_net_pressure"]
        + 0.10 * solar_df["surya_declination"] / 24.0
    )
    out["KTW_raw"] = ktw_raw
    out["KTW"] = _squash(_expanding_z(ktw_raw), scale=1.6)

    # --- Derived readouts used by the gate and the narrative --------------------------------
    # A single headline astro score: expansion minus fear, tempered by structure.
    out["ASTRO_SCORE"] = (
        0.34 * out["VRI"] + 0.26 * out["CBI"] + 0.22 * out["GSI"] - 0.18 * (out["BHY"] * 2 - 1)
    )
    out["ASTRO_MOMENTUM"] = out["ASTRO_SCORE"].diff(5).fillna(0.0)

    # Hard-gate conditions: the classical "do not transact" set.
    out["GATE_vishti"] = panchanga_df["karana_is_vishti"].astype(int)
    out["GATE_eclipse"] = events_df["eclipse_window_3d"].astype(int)
    out["GATE_chandrashtama"] = natal_df["nat_chandrashtama"].astype(int)
    out["GATE_gandanta"] = lunar_df["moon_in_gandanta"].astype(int)
    out["GATE_rikta"] = panchanga_df["tithi_is_rikta"].astype(int)
    out["GATE_any"] = (
        out[["GATE_vishti", "GATE_eclipse", "GATE_chandrashtama", "GATE_gandanta"]]
        .max(axis=1)
        .astype(int)
    )
    return out
