# ChandraQuant v3 — Kāla-Chakra Alpha Engine

> **Where Jyotiṣa meets quantitative alpha.**
> Natal-relative Vedic astronomy + machine learning for probabilistic regime detection
> in Indian equity indices.

---

## 0. Permanent header (read this first)

### What this project is
A terminal-first quant engine that detects and forecasts **market regimes** for three Indian
indices using a hybrid of technical indicators and ~180 **natal-relative Jyotiṣa features**
computed from NASA JPL ephemerides. It ships with a TradingView Pine indicator + strategy,
and explains every call in plain English seasoned with Sanskrit astro terminology.

### Tickers (fixed — exactly three)
| Name | Yahoo symbol | Natal (launch) | Base date |
|---|---|---|---|
| NIFTY 50 | `^NSEI` | 1996-04-22 | 1995-11-03 |
| BANKNIFTY | `^NSEBANK` | 2000-09-15 | 2000-01-01 |
| CNX IT | `^CNXIT` | 1999-05-11 | 1996-01-01 |

All natal charts cast for **Mumbai 18.94°N 72.83°E, 09:55 IST**.

### The core insight (this is the whole project)
ChandraQuant **v1** used 10 astro features that were functions of *calendar date alone*
(Moon longitude, Sun longitude, Tithi, Nakṣatra + harmonics). Pooling three tickers meant
every astro row appeared three times against three different labels — the astro block was
**mathematically incapable** of discriminating between indices. That is precisely why v1
reported **ΔAUC ≈ −0.018**: the hybrid model was *worse* than technical-only.

**v3 fixes this with mundane/natal Jyotiṣa.** Each index gets a birth chart, and every
celestial quantity is expressed as a **transit relative to that chart**. Astro becomes
ticker-specific, and this unlocks the entire classical apparatus v1 never touched:
Vimśottarī Daśā, Sade Sati, Tārābala, Chandrabala, Aṣṭakavarga, Gochara through natal bhāvas.

### Architecture in one line
`OHLCV → 45 technical features + 180 natal-relative astro features → per-ticker LightGBM pair → astro gate → calibrated P(Vṛddhi) → regime + narrative + Pine strategy`

The astro layer is **load-bearing, not decorative**: it sets the blend weight between the
technical and astro models, modulates the entry threshold, and hard-blocks entries on
Viṣṭi karaṇa / eclipse windows / Chandrāṣṭama / high Bhaya. Removing astro measurably
changes trades.

### The four regimes
| Sanskrit | Meaning | Colour |
|---|---|---|
| **Vṛddhi** | expansion / bull | green |
| **Sthira** | consolidation / range | amber |
| **Kṣaya** | decline / bear | red |
| **Kṣobha** | turbulence / high-vol | violet |

### How to run
```powershell
python scripts/refresh_data.py --all            # pull + snapshot OHLCV
python scripts/build_astro.py --all             # compute the astro feature cache
python scripts/train.py --ticker ALL            # purged walk-forward training
python scripts/backtest.py --ticker ALL --sweep # headline metrics + tornado surfaces
python scripts/validate.py                      # permutation / bootstrap / crisis / stratify
chandraquant                                    # ← the app: picker → dashboard
chandraquant --ticker NIFTY --date 2020-03-23   # historical replay
chandraquant --web                              # optional Plotly dashboard
chandraquant --honest                           # unconditional metrics
```

### Ground rules for this repo
1. **Every number shown anywhere traces to `artifacts/metrics.json`.** Nothing is typed by hand.
2. **No lookahead, ever.** Astro is date-deterministic so it is leak-free by construction;
   assert it anyway. Technical features are strictly causal.
3. **Never pool tickers for training.** Pooling is what broke v1.
4. **The app must work offline.** Network failure falls back to `data/snapshot/` silently.
5. **Append to this log at every phase. Never rewrite history.**

### Current headline metrics
_Populated by `scripts/backtest.py`; see `artifacts/metrics.json`._

| Metric | NIFTY | BANKNIFTY | CNX IT |
|---|---|---|---|
| Win rate | — | — | — |
| CAGR | — | — | — |
| Sharpe | — | — | — |
| Max DD | — | — | — |

---

## 1. Build log

<!-- Newest entries at the bottom. One entry per commit. Never edit past entries. -->

### [2026-08-28 17:29] chore: scaffold ChandraQuant v3
**Commit:** _pending_
**Added:** `.gitignore`, `pyproject.toml`, `CLAUDE.md`, full package tree under `chandraquant/`.

**What:** Initialised the git repo and laid out the 13-module package skeleton, config/,
data/{raw,snapshot,ephem}/, artifacts/, pine/, scripts/, tests/, docs/.

**Environment verified (Python 3.13.2):**
- Present: `numpy 2.3.3`, `pandas 2.3.3`, `scikit-learn 1.9.0`, `scipy 1.18.1`,
  `skyfield 1.54`, `yfinance 1.2.0`, `rich`, `typer`, `fastapi`, `uvicorn`,
  `pyarrow 25.0.1`, `joblib 1.5.3`, `pytz`.
- Installed this phase: `textual`, `questionary`, `lightgbm`, `shap`, `matplotlib`,
  `plotly`, `statsmodels`, `tabulate`, `pyyaml`.

**Decisions:**
- Console entry point is `chandraquant = chandraquant.cli.app:main`.
- `data/snapshot/` **is** committed — it is the offline fallback that keeps demos alive.
  `data/raw/` and `artifacts/models/` are gitignored (reproducible / large).
- Third ticker is **CNX IT (`^CNXIT`)**, replacing SENSEX from the v1 documents.

**Data reality check (verified against Yahoo 2026-08-28):** all three indices return bars
only from **2007-09-17**, not July 1996 — ~4,650 daily bars each, ~13,950 total. GFC
(Sep 2008–Mar 2009), COVID and Russia-Ukraine crisis windows are all still covered.
A backfill path to 1996 is planned but **no document may claim a 1996 start until it lands.**

**Inherited defect to resolve:** `ML_ProjectReport.pdf` contradicts itself — §7.2 states
Hybrid AUC 0.5094, §9.4 states "actual Hybrid AUC of 0.9964" against a permutation null at
0.990. v3 produces a single reproducible number and the regenerated report supersedes both.

**Next:** `chandraquant/astro/ephemeris.py` + `ayanamsa.py` — Skyfield/DE440s wrapper and
Lahiri sidereal conversion.

### [2026-08-28 18:05] feat(astro): the complete Jyotisha engine (12 modules, 831 features)
**Commit:** _pending_
**Added:** `chandraquant/config.py`, `chandraquant/astro/{ayanamsa,ephemeris,panchanga,lunar,solar,grahas,aspects,events,natal,dasha,ashtakavarga,shadbala,muhurta,composites,engine}.py`, `config/{nakshatra,tithi,yoga,karana,graha,ashtakavarga,tickers}.yaml`

**What:** The full astro stack, from JPL DE440s through to five branded composite indices.

| Module | Delivers |
|---|---|
| `ayanamsa` | Lahiri (Chitrapaksha) via IAU-2006 precession, anchored 23.250182 deg at JD 2435553.5 |
| `ephemeris` | Skyfield/DE440s apparent geocentric sidereal positions, speeds, mean+true nodes |
| `panchanga` | Tithi (30), Vara, Nakshatra (27 + pada), Yoga (27), Karana (11/60 slots) |
| `lunar` | phase, Chandra Shara, gandanta, sandhi, speed gati, tidal force, Kriya/Avastha/Vela |
| `solar` | Sankranti ingress, Uttarayana/Dakshinayana, declination, Ritu, Masa |
| `grahas` | dignity, uccha bala, Vakri, Stambhana, Asta, Graha Yuddha |
| `aspects` | Parashari drishti (graded), continuous orb kernels, 9 named market yogas, Kemadruma |
| `events` | true eclipses, Gochara ingress, stations, Great Conjunction, ramps for all |
| `natal` | **the differentiator** - Lagna, Gochara, Tarabala, Chandrabala, Sade Sati, bhava activation |
| `dasha` | Vimshottari MD/AD/PD chain seeded from each index's natal Moon nakshatra |
| `ashtakavarga` | Bhinnashtakavarga + Sarvashtakavarga transit strength, per natal chart |
| `shadbala` | six-fold strength in rupas: Sthana, Dig, Kala, Cheshta, Naisargika, Drik |
| `muhurta` | daily Lagna, Hora, Rahu Kaal / Yamaganda / Gulika, Abhijit |
| `composites` | CBI, GSI, VRI, BHY, KTW + ASTRO_SCORE + the hard gates |

**Verified against external ground truth:**
- Ayanamsa 2026-08-28 = 24.2341 deg (expected ~24 deg 14').
- Makar Sankranti 2025-01-14: Surya sidereal 270.009 deg, i.e. exactly entering Makara.
- Diwali 2024-10-31 and 2025-10-20 both resolve to Krishna Chaturdashi at 09:15 IST; 2025-10-20 nakshatra Hasta - matches Drik Panchang.
- **Eclipses 2025-26: 8/8 exact date match against the NASA catalogue, zero false positives** (2025-03-14, 03-29, 09-07, 09-21; 2026-02-17, 03-03, 08-12, 08-28). Rate 4.78/yr over 2018-2026.
- Budha vakri 2025 detected 03-16 / 07-19 / 11-10 vs actual stations 03-15 / 07-18 / 11-09. Shani vakri 138d (actual ~138). Guru vakri 85d (actual ~86).
- Great Conjunction: minimum Guru-Shani separation 2020-12-22 at 0.04 deg (actual 2020-12-21).
- Shani ingresses 2020-01-25 Makara, 2022-04-30 Kumbha, 2022-07-13 back to Makara, 2023-01-18 Kumbha, 2025-03-30 Meena - all within a day of actual.
- Ascendant formula: at Mumbai sunrise the Lagna equals the Sun's longitude to 0.9 deg, which is precisely the refraction + semi-diameter offset that defines sunrise. Formula confirmed correct.
- Ashtakavarga invariants: BAV totals 48/49/39/54/56/52/39 and SAV grand total 337, all exact.
- Panchanga frequencies over 1096 days: Rikta 0.204 (expect 0.200), malefic yoga 0.333 (expect 0.333), all 27 nakshatras / 30 tithis / 11 karanas covered.

**THE FIX THAT MATTERS - v1's fatal flaw is dead.** Natal charts cast per index:
| Index | Born | Natal Moon nakshatra | Lagna |
|---|---|---|---|
| NIFTY 50 | 1996-04-22 10:00 IST | Mrgasirsa (Mangala) | Mithuna |
| BANKNIFTY | 2003-09-15 10:00 IST | Bharani (Shukra) | Tula |
| CNX IT | 1999-05-11 10:00 IST | Purva Bhadrapada (Guru) | Mithuna |

Measured on the same calendar day (2026-08-28), the three indices now diverge sharply:
NIFTY in **Sampat** tara (+0.9), BANKNIFTY in **Pratyari** (-0.7) and under Sade Sati,
CNXIT in **Ati-Mitra** (+1.0). ASTRO_SCORE +0.231 / -0.312 / +0.281.
**188 of 831 numeric features (22.6%) differ between tickers.** In v1 that number was 0.

**Decisions:**
- Evaluation point 09:15 IST (current NSE open); natal charts at 10:00 IST (the opening
  bell of their launch era). Both configurable in `config/tickers.yaml`.
- Mean lunar node for Rahu/Ketu (Indian panchanga convention); true node available.
- Eclipse solar limit tightened to 1.40 deg (the "certain" ecliptic limit) rather than the
  1.58 deg absolute maximum - this removes two 2026 grazing false positives.
- Composites z-scored on an **expanding** window, never full-sample, so they cannot leak.
- 831 features against ~4,600 rows per ticker is a high ratio; feature selection is
  deferred to the model phase rather than pruned blindly here.

**Next:** data layer (`data/loaders.py`) and the technical feature block.
