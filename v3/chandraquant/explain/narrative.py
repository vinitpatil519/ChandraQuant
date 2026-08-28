"""The "why" engine - turns celestial state into plain English with Sanskrit terms.

Template-and-rule based, deliberately. It must run offline, instantly, and identically
every time: the same sky on the same date for the same index always produces the same
paragraph. An LLM here would be slower, non-deterministic and impossible to test.

Construction: gather every astro condition that is currently *notable* (a state well
away from its neutral value), score each by how far it deviates and how much classical
weight it carries, then render the top few as sentences in descending order. Supporting
and opposing conditions are separated so the paragraph reads as a genuine argument
rather than a list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import graha_cfg, nakshatra_cfg, tithi_cfg
from ..labels.regime import REGIME_MEANING, REGIME_NAMES


@dataclass
class Driver:
    """One notable celestial condition."""

    text: str
    weight: float           # how strongly it argues, in [0, 1]
    direction: int          # +1 supportive, -1 opposing, 0 contextual
    term: str = ""          # the Sanskrit term to gloss

    @property
    def signed(self) -> float:
        return self.weight * self.direction


def _tara_name(idx: int) -> tuple[str, float]:
    for t in nakshatra_cfg()["tarabala"]:
        if int(t["index"]) == int(idx):
            return t["display"], float(t["quality"])
    return "", 0.0


def collect_drivers(row: pd.Series, ticker_display: str) -> list[Driver]:
    """Every notable condition in today's sky, scored and signed."""
    d: list[Driver] = []
    g = lambda k, default=0.0: float(row[k]) if k in row and pd.notna(row[k]) else default
    s = lambda k, default="": str(row[k]) if k in row and pd.notna(row[k]) else default

    # --- Nakshatra: the headline lunar position ---------------------------------
    nak = s("nakshatra_name")
    arche = s("nakshatra_archetype")
    if nak:
        gana = s("nakshatra_gana")
        pada = int(g("pada", 1))
        d.append(
            Driver(
                f"Chandra transits **{nak}** (pada {pada}) - {arche}",
                0.85,
                1 if g("nakshatra_bias") > 0 else -1,
                "Nakshatra",
            )
        )
        if gana == "Rakshasa":
            d.append(
                Driver(
                    f"{nak} is a **Rakshasa gana** nakshatra, classically obstructive to "
                    f"smooth commerce",
                    0.45,
                    -1,
                    "Nakshatra",
                )
            )

    # --- Tarabala: the natal-relative term, and the most ticker-specific --------
    tara_idx = int(g("nat_tarabala", 0))
    if tara_idx:
        tname, tq = _tara_name(tara_idx)
        verdict = "favourable" if tq > 0.2 else ("adverse" if tq < -0.2 else "neutral")
        d.append(
            Driver(
                f"the Moon sits in **{tname} tara** counted from {ticker_display}'s natal "
                f"Chandra - the {verdict} position in the nine-fold cycle",
                0.75 * min(1.0, abs(tq) + 0.3),
                1 if tq > 0 else (-1 if tq < 0 else 0),
                tname.replace("-", "-"),
            )
        )

    # --- Tithi and paksha --------------------------------------------------------
    tithi = s("tithi_name")
    if tithi:
        waxing = g("paksha_shukla") > 0.5
        phrase = "waxing bias of **Shukla Paksha**" if waxing else "waning drag of **Krishna Paksha**"
        d.append(Driver(f"**{tithi}** carries the {phrase}", 0.5, 1 if waxing else -1, "Paksha"))
    if g("tithi_is_rikta") > 0.5:
        d.append(
            Driver(
                "today is a **Rikta tithi** - the 4th, 9th or 14th, which muhurta texts "
                "call barren for new commercial undertakings",
                0.6,
                -1,
                "Rikta",
            )
        )

    # --- Hard gates: the abstention rules ------------------------------------------
    if g("karana_is_vishti") > 0.5:
        d.append(
            Driver(
                "**Vishti karana (Bhadra)** is running - the classical no-transaction "
                "window, recurring roughly every 3.7 days",
                0.9,
                -1,
                "Vishti",
            )
        )
    if g("eclipse_window_3d") > 0.5:
        kind = "solar" if g("is_solar_eclipse") > 0.5 else "lunar"
        d.append(
            Driver(
                f"a **Grahana** ({kind} eclipse) falls within three days - classically a "
                f"window for abstention rather than action",
                0.85,
                -1,
                "Grahana",
            )
        )
    if g("nat_chandrashtama") > 0.5:
        d.append(
            Driver(
                f"**Chandrashtama** - the Moon occupies the 8th from {ticker_display}'s "
                f"natal Chandra, an exposed position",
                0.7,
                -1,
                "Chandrashtama",
            )
        )
    if g("moon_in_gandanta") > 0.5:
        d.append(
            Driver(
                "the Moon is in **Gandanta**, the karmic knot at a water-fire junction - "
                "a marker of structural instability",
                0.65,
                -1,
                "Gandanta",
            )
        )

    # --- Retrogression -------------------------------------------------------------
    for graha, label in (
        ("Budha", "commerce and settlement"),
        ("Shukra", "valuation and risk appetite"),
        ("Mangala", "momentum and aggression"),
        ("Guru", "credit and expansion"),
        ("Shani", "structure and discipline"),
    ):
        if g(f"{graha}_vakri") > 0.5:
            weight = 0.55 if graha in ("Budha", "Guru") else 0.4
            d.append(
                Driver(
                    f"**{graha} is vakri** (retrograde), which classically turns {label} "
                    f"inward - review rather than initiation",
                    weight,
                    -1,
                    "Vakri",
                )
            )
        if g(f"{graha}_stambhana") > 0.5:
            d.append(
                Driver(
                    f"**{graha} is at Stambhana** - stationary, the point of maximum "
                    f"classical effect and the strongest turning-point marker",
                    0.7,
                    0,
                    "Stambhana",
                )
            )
        if g(f"{graha}_asta") > 0.5 and g(f"{graha}_asta_depth") > 0.4:
            d.append(
                Driver(
                    f"**{graha} is asta** (combust) - swallowed by Surya's glare and "
                    f"unable to deliver",
                    0.4,
                    -1,
                    "Asta",
                )
            )

    # --- Sade Sati and the Shani afflictions ------------------------------------------
    if g("nat_sade_sati") > 0.5:
        phase = g("nat_sade_sati_phase", -1)
        names = {0: "rising", 1: "peak", 2: "setting"}
        pname = names.get(int(phase), "")
        d.append(
            Driver(
                f"{ticker_display} is under **Sade Sati** ({pname} phase) - Shani's "
                f"seven-and-a-half year passage over the natal Chandra",
                0.6,
                -1,
                "Sade Sati",
            )
        )
    if g("nat_guru_kendra_moon") > 0.5:
        d.append(
            Driver(
                "**Guru occupies a kendra** from the natal Chandra - the Gajakesari "
                "configuration, classically a sustained-expansion structure",
                0.6,
                1,
                "Guru",
            )
        )

    # --- Named yogas ------------------------------------------------------------------
    yoga_specs = {y["name"]: y for y in graha_cfg()["named_yogas"]}
    for name, spec in yoga_specs.items():
        col = f"yoga_{name}"
        if g(col) > 0.5:
            polarity = float(spec["polarity"])
            d.append(
                Driver(
                    f"**{spec['label']}** is active - {spec['reading']}",
                    min(0.8, abs(polarity)),
                    1 if polarity > 0 else -1,
                    name,
                )
            )

    # --- Nitya yoga -------------------------------------------------------------------
    if g("yoga_is_malefic") > 0.5:
        d.append(
            Driver(
                f"the nitya yoga is **{s('yoga_name')}**, one of the nine classically "
                f"inauspicious yogas",
                0.5,
                -1,
                "Yoga",
            )
        )

    # --- Bhava activation --------------------------------------------------------------
    spec5 = g("nat_speculation_house")
    if abs(spec5) >= 1:
        d.append(
            Driver(
                f"the **5th bhava** (Poorva Punya - speculation) is "
                f"{'strongly supported' if spec5 > 0 else 'under malefic pressure'}",
                0.45,
                1 if spec5 > 0 else -1,
                "Bhava",
            )
        )
    gains = g("nat_gains_house")
    if abs(gains) >= 1:
        d.append(
            Driver(
                f"the **11th bhava** (Labha - gains) is "
                f"{'occupied by benefics' if gains > 0 else 'afflicted'}",
                0.45,
                1 if gains > 0 else -1,
                "Bhava",
            )
        )

    return d


def dasha_sentence(dasha: dict, ticker_display: str) -> str:
    if not dasha:
        return ""
    md, ad = dasha.get("md"), dasha.get("ad")
    md_char = dasha.get("md_character", "")
    md_read = dasha.get("md_reading", "")
    years = dasha.get("md_years_remaining", 0)
    return (
        f"Structurally, {ticker_display} is running **{md} Mahadasha / {ad} Antardasha** - "
        f"an era of {md_char}, {md_read}. The Mahadasha holds for another "
        f"{years:.1f} years."
    )


def build(
    row: pd.Series,
    ticker_display: str,
    regime: int,
    probability: float,
    dasha: dict | None = None,
    max_drivers: int = 5,
) -> dict:
    """The full explanation block for one date and one index."""
    drivers = collect_drivers(row, ticker_display)
    drivers.sort(key=lambda x: -x.weight)

    supporting = [d for d in drivers if d.direction > 0][:3]
    opposing = [d for d in drivers if d.direction < 0][:3]
    contextual = [d for d in drivers if d.direction == 0][:1]

    regime_name = REGIME_NAMES.get(int(regime), "Sthira")
    # The detected regime and the forward probability answer different questions and are
    # phrased separately - calling P(Vriddhi) "confidence in Sthira" would be nonsense.
    lead = f"**{regime_name.upper()}** - {REGIME_MEANING.get(int(regime), '')}."
    if np.isfinite(probability):
        lead += (
            f" Forward view: the engine puts **P(Vriddhi) at {probability * 100:.0f}%** "
            f"over the next five sessions."
        )

    parts = [lead]
    if supporting:
        joined = "; ".join(x.text for x in supporting)
        parts.append(f"Supporting it: {joined}.")
    if opposing:
        joined = "; ".join(x.text for x in opposing)
        parts.append(f"Working against it: {joined}.")
    if contextual:
        parts.append(contextual[0].text.capitalize() + ".")
    if dasha:
        parts.append(dasha_sentence(dasha, ticker_display))

    net = sum(x.signed for x in drivers)
    return {
        "regime": regime_name,
        "probability": probability,
        "text": " ".join(parts),
        "sentences": parts,
        "drivers": drivers[:max_drivers],
        "supporting": supporting,
        "opposing": opposing,
        "net_astro_argument": float(np.tanh(net / 3.0)),
    }


def plain(text: str) -> str:
    """Strip the markdown emphasis markers for plain-text surfaces."""
    return text.replace("**", "")
