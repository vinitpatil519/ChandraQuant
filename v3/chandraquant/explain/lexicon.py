"""Sanskrit glossary - every term the interface shows, with a one-line plain meaning.

A reader who has never opened a Jyotisha text should be able to follow every sentence
the narrative engine produces. Any term that appears on screen gets an entry here, and
the UI shows the gloss on expand.
"""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # --- regimes -------------------------------------------------------------
    "Vriddhi": "growth - the expansion regime, trend favours the long side",
    "Sthira": "stability - consolidation, the market coils rather than trends",
    "Kshaya": "decay - the decline regime, distribution is in control",
    "Kshobha": "agitation - turbulence, direction unreliable and risk elevated",
    # --- panchanga -----------------------------------------------------------
    "Panchanga": "the five limbs - tithi, vara, nakshatra, yoga and karana",
    "Tithi": "lunar day - 12 degrees of angular separation between Moon and Sun",
    "Paksha": "fortnight - Shukla is waxing toward full, Krishna waning toward dark",
    "Shukla Paksha": "the waxing fortnight - building light, growing conviction",
    "Krishna Paksha": "the waning fortnight - fading light, distribution and fear",
    "Nakshatra": "lunar mansion - one of 27 sectors of 13 deg 20' the Moon transits",
    "Pada": "quarter - each nakshatra divides into four padas of 3 deg 20'",
    "Yoga": "one of 27 states from the summed longitudes of Sun and Moon",
    "Karana": "half a tithi - sixty of them make a lunar month",
    "Vara": "weekday, and the graha that rules it",
    "Vishti": "Bhadra - the classical no-transaction window, recurring every ~3.7 days",
    "Rikta": "empty - the 4th, 9th and 14th tithis, barren for commerce",
    "Purnima": "full moon - peak illumination, a classical reversal window",
    "Amavasya": "new moon - the dark moon, a void and a turning point",
    # --- grahas --------------------------------------------------------------
    "Graha": "a celestial body that seizes or influences - the nine of Jyotisha",
    "Surya": "the Sun - sovereign authority, policy, the structural year",
    "Chandra": "the Moon - mind and crowd sentiment, the fastest-moving signal",
    "Mangala": "Mars - energy, conflict, volatility expansion",
    "Budha": "Mercury - commerce, communication, the trade itself",
    "Guru": "Jupiter - expansion, credit, the great benefic",
    "Shukra": "Venus - valuation, risk appetite, what buyers will pay up for",
    "Shani": "Saturn - contraction, discipline, time, the great malefic",
    "Rahu": "the north lunar node - leverage, speculation, amplification",
    "Ketu": "the south lunar node - severance, sudden dislocation",
    "Vakri": "retrograde - apparent motion reversed against the zodiac",
    "Stambhana": "station - the instant motion halts and reverses, maximum effect",
    "Asta": "combustion - swallowed by the Sun's glare and unable to act",
    "Graha Yuddha": "planetary war - two grahas within one degree, one is defeated",
    "Uccha": "exaltation - a graha at its point of greatest dignity",
    "Neecha": "debilitation - a graha at its weakest point",
    # --- natal / transit ------------------------------------------------------
    "Lagna": "the ascendant - the zodiac degree rising on the eastern horizon",
    "Rashi": "zodiac sign - one of the twelve 30-degree divisions",
    "Bhava": "house - a sector of the chart carrying a domain of meaning",
    "Gochara": "transit - a graha's current position read against the natal chart",
    "Tarabala": "the 9-fold cycle counted from the natal Moon's nakshatra",
    "Janma": "the birth star itself - self-referential and exposed",
    "Sampat": "wealth - the second tara, giving prosperity",
    "Vipat": "danger - the third tara, giving loss",
    "Kshema": "well-being - the fourth tara, giving safety",
    "Pratyari": "obstruction - the fifth tara, active opposition",
    "Sadhaka": "accomplishment - the sixth tara, goals realised",
    "Vadha": "destruction - the seventh and most adverse tara",
    "Mitra": "friend - the eighth tara, supportive",
    "Ati-Mitra": "great friend - the ninth tara, strongest support",
    "Chandrabala": "lunar strength - the transit Moon's relation to the natal Moon",
    "Chandrashtama": "the Moon in the 8th from its natal position - an exposed window",
    "Sade Sati": "Saturn's 7.5-year passage over the 12th, 1st and 2nd from natal Moon",
    "Gandanta": "the karmic knot at a water-fire sign junction - structural instability",
    "Sandhi": "junction - the boundary degrees between two signs",
    # --- dasha ----------------------------------------------------------------
    "Dasha": "planetary period - an era ruled by one graha",
    "Vimshottari": "the 120-year dasha system seeded by the natal Moon's nakshatra",
    "Mahadasha": "the major period, 6 to 20 years, setting the structural era",
    "Antardasha": "the sub-period within a Mahadasha, modulating its character",
    "Pratyantardasha": "the sub-sub-period, colouring the immediate weeks",
    # --- strength / measurement ------------------------------------------------
    "Shadbala": "six-fold strength - the classical measure of what a graha can deliver",
    "Rupa": "the unit of Shadbala - 60 virupas make one rupa",
    "Ashtakavarga": "the bindu system scoring each sign's capacity to support transits",
    "Bindu": "a benefic point donated to a sign in the Ashtakavarga scheme",
    "Sarvashtakavarga": "the summed bindu score per sign, 0 to 56, totalling 337",
    "Muhurta": "electional timing - the branch concerned with when to act",
    "Hora": "planetary hour - each day divides into 24 ruled in Chaldean order",
    "Rahu Kaal": "the inauspicious eighth-part of daylight, different each weekday",
    "Abhijit": "the auspicious muhurta straddling local noon",
    "Grahana": "eclipse - the luminaries obscured at a node",
    "Sankranti": "the Sun's ingress into a new sign",
    "Uttarayana": "the northern solar course - the auspicious half of the year",
    "Kala Chakra": "the wheel of time - the cyclic conception of temporal structure",
    "Kala Taranga": "time-wave - ChandraQuant's composite planetary tide",
    "Jyotisha": "the science of light - Vedic astronomy and astrology",
    # --- composites -----------------------------------------------------------
    "CBI": "Chandra Bala Index - short-cycle lunar sentiment, natal-relative",
    "GSI": "Graha Shakti Index - structural benefic minus malefic strength",
    "VRI": "Vriddhi Index - expansion pressure from Guru, Shukra and the dasha lord",
    "BHY": "Bhaya Index - panic risk from Shani, Rahu, Mangala and eclipse proximity",
    "KTW": "Kala Taranga - the normalised cosmic tide overlaid on price",
}

# Aliases so lookups survive spelling variants and diacritic-stripped forms.
ALIASES = {
    "shukla": "Shukla Paksha",
    "krishna": "Krishna Paksha",
    "bhadra": "Vishti",
    "vishti (bhadra)": "Vishti",
    "atimitra": "Ati-Mitra",
    "sadesati": "Sade Sati",
    "rahukaal": "Rahu Kaal",
}


def gloss(term: str) -> str | None:
    """Plain-language meaning for a term, or None if it is not in the glossary."""
    if term in GLOSSARY:
        return GLOSSARY[term]
    key = term.strip().lower().replace("-", "").replace(" ", "")
    for name, meaning in GLOSSARY.items():
        if name.lower().replace("-", "").replace(" ", "") == key:
            return meaning
    alias = ALIASES.get(term.strip().lower())
    return GLOSSARY.get(alias) if alias else None


def annotate(text: str) -> list[tuple[str, str]]:
    """Every glossary term appearing in `text`, with its meaning."""
    found = []
    lowered = text.lower()
    for term, meaning in GLOSSARY.items():
        if term.lower() in lowered:
            found.append((term, meaning))
    return sorted(set(found), key=lambda x: -len(x[0]))
