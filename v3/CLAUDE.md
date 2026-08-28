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

### [2026-08-28 18:35] feat(data+features): loaders, 68 technical features, 4-state labels
**Commit:** _pending_
**Added:** `chandraquant/data/loaders.py`, `chandraquant/features/{technical,dataset}.py`, `chandraquant/labels/regime.py`, `data/snapshot/*.parquet`

**What:**
- `loaders.py` - yfinance with an 8s hard timeout and silent fallback to the committed
  snapshot. Returns a `DataStatus` so the UI reports `live` / `cached` honestly.
- `technical.py` - 68 causal features (v1's 31 plus realised-vol ratios, Donchian
  position, ADX, drawdown state, gap/candle shape, OBV slope, streaks).
- `regime.py` - four-state Vriddhi / Sthira / Kshaya / Kshobha labels, vol-scaled mu.
- `dataset.py` - the join, with an explicit astro/technical column split for ablation.

**Verified:**
- `technical.assert_causal` passes: recomputing on truncated history reproduces the
  tail exactly, so no feature peeks forward. Run as part of dataset build.
- Live pull succeeded for all three tickers; snapshots written for offline demo.
- Rows: NIFTY 4649, BANKNIFTY 4664, CNXIT 4664, all 2007-09-17 to 2026-08-28.
- Feature counts: 899-900 raw (831 astro + 68 technical); 582 after `prune_features`
  drops empty, constant and >0.995-correlated columns (520 astro + 62 technical).

**Base rates after the Kshobha fix** (H=5): Vriddhi 0.354 / Sthira 0.289 / Kshaya 0.252
/ Kshobha 0.104 on NIFTY, similar on the others.

**Decision - Kshobha threshold is ROLLING, not expanding.** With an expanding quantile
the 2008 GFC dominates the threshold permanently and Kshobha collapses to 3.8% of rows,
concentrated in 2008. A 500-day rolling quantile restores the intended ~11% and makes
the label mean "turbulent relative to the current regime", which is the useful reading.
Still strictly causal.

**Open item:** 582 features against ~4,600 rows is a high ratio. Handled at the model
layer by per-block selection rather than by blind pruning here.

**Next:** the gated hybrid model - technical LightGBM, astro LightGBM, astro gate,
isotonic calibration, purged walk-forward CV.

### [2026-08-28 19:20] feat(models+backtest): gated hybrid, learned edges, vol-targeted strategy
**Commit:** `b8193db`
**Added:** `models/{splits,hybrid,astro_edge}.py`, `backtest/{engine,metrics,strategy}.py`, `scripts/optimize.py`

**What:** Purged walk-forward CV (6 folds, 5-bar purge + 5-bar embargo); technical and
astro LightGBM blocks under an astro gate; event-driven and vectorised backtesters.

**THE CENTRAL NEGATIVE RESULT.** The hand-assigned `market_bias` values I wrote into the
config YAMLs - faithful to the classical texts, Pushya +0.60, Mula -0.55 - are not merely
uninformative, they are **harmful**. Filtering NIFTY exposure on `ASTRO_SCORE > 0` built
from those priors cut CAGR from **9.2% to 0.1%**. Reading a Sanskrit adjective and guessing
a sign does not produce alpha.

**The fix:** `astro_edge.py` measures each celestial state's conditional edge from training
data with empirical-Bayes shrinkage (k=60 pseudo-observations) instead of asserting it.
Astro block AUC improved 0.4912 -> 0.5025. On a 60/40 holdout `ASTRO_EDGE` alone reached
**0.5274, beating the technical block's 0.5128**, with high-edge days at 37.9% Vriddhi vs
32.4% low-edge. **But across six walk-forward folds the sign flips** - the conditional
edges are not stationary. Recorded rather than buried.

**Threshold bug found and fixed:** entry thresholds were absolute probabilities, but after
isotonic calibration to a ~35% base rate a 0.55 cut fires on 0.1% of days. Switched to
causal expanding percentiles - scale-free and survives recalibration.

**Strategy design, arrived at empirically:** long-only daily index systems cannot beat
buy-and-hold on return. The lever that works is **volatility targeting**. The classical
abstention gates (Vishti, eclipse, Chandrashtama) cost Sharpe 0.60 -> 0.47, so per the
user's decision astro became a display layer and position sizing is trend + vol targeting.

**Bug caught:** `strategy.trend_signal` used a binary gate while `optimize.py` tuned a
continuous one, so CNXIT's tuned parameters were applied to a different formula (Sharpe
0.49 vs the 0.70 the optimiser found). Aligned; they now share one definition.

---

### [2026-08-28 19:50] feat(cli+tui+explain): the `chandraquant` app
**Commit:** `0c50225`, `7d1e4e0`
**Added:** `cli/{app,picker,splash}.py`, `tui/{dashboard,widgets}.py`, `explain/{narrative,lexicon}.py`, `inference.py`

**What:** Animated Kala-Chakra splash, arrow-key ticker picker with live status, and a Rich
dashboard: verdict, panchanga, natal+dasha, navagraha table, composite meters, sparklines,
narrative, forward calendar, backtest card. Deterministic template narrative (no LLM - must
run offline and identically every time) over a 90-term Sanskrit glossary.

**Design decision:** `inference.py` keeps **DETECTED regime** (nowcast from realised price,
no model) strictly separate from **FORECAST P(Vriddhi)** (model output). Conflating them
would be the easiest way to overstate what the system knows. An early narrative bug did
exactly that - it printed "STHIRA, 37% confidence" where 37% was P(Vriddhi). Fixed.

---

### [2026-08-28 20:30] feat(pine+tests): Meeus astronomy in Pine, 60-test suite
**Commit:** `8ab4def`
**Added:** `pine/ChandraQuant_{KalaChakra,Strategy}.pine`, `chandraquant/pine/emit.py`, `tests/test_{astro,pipeline}.py`

**What:** Pine v6 scripts computing Sun and Moon ecliptic longitudes natively from Meeus
periodic series (27 lunar terms, 12 latitude terms), Lahiri sidereal. No ephemeris file,
no expiry, any bar any date.

**Verified against Skyfield/DE440s, 1,253 samples 2007-2030:**
| | max error | classification agreement |
|---|---|---|
| Surya | 0.0095 deg | nakshatra / tithi / rashi **100%** |
| Chandra | 0.0236 deg | **100%** |

Slow grahas use an embedded table. **Quarterly nodes gave 3.2 deg error on Guru** because
its retrograde loops are cut straight across; switched to monthly nodes - now under 0.4 deg.

**Pine strategy honesty note.** High-win-rate configurations were tested and rejected:
a 0.8-ATR target gives **84.7% win rate at profit factor 0.98 and negative CAGR**. That is
exactly the shape of v1's advertised 92.6% win rate (738 wins at +0.46%, 57 losses at
-5.33% - profit factor 1.12). TradingView prints Net Profit beside Win Rate, so that
screenshot collapses on sight. The default "Trend" mode ships instead: ~28-33% win rate,
**profit factor 1.8-2.1, max drawdown -16.5% vs buy-and-hold -38.4%**. Swing mode is
included but labelled.

**60 tests, all passing.** Every astro assertion anchored to external ground truth:
NASA eclipse catalogue (8/8 exact, zero spurious), Drik Panchang dates, Ashtakavarga
totals, ascendant-equals-Sun-at-sunrise, retrograde stations, Great Conjunction. Plus a
named regression test for the v1 ticker-identity bug and causality proofs.

---

### [2026-08-28 21:00] docs+web: README, methodology, browser dashboard
**Commit:** _pending_
**Added:** `README.md`, `docs/METHODOLOGY.md`, `chandraquant/web/server.py`, `scripts/{refresh_data,train,validate}.py`

**Final measured results** (2010-01-04 to 2026-08-28, 5bps/side, next-bar execution):

| | NIFTY | BANKNIFTY | CNX IT |
|---|---|---|---|
| CAGR | 8.35% (B&H 9.63%) | 8.49% (11.70%) | **15.06%** (10.57%) |
| Sharpe | 0.66 (0.66) | 0.66 (0.61) | 0.68 (0.59) |
| Max DD | **-21.6%** (-38.4%) | **-22.3%** (-47.9%) | -40.2% (-44.0%) |
| **Calmar** | **0.387** (0.251) | **0.380** (0.245) | **0.375** (0.240) |

**Headline: Calmar +54% to +56% across all three indices, drawdown nearly halved on two.**

**Validation battery** (`artifacts/validation.json`): astro ablation deltas +0.0014 /
-0.0019 / -0.0032; permutation p = 0.500 / 0.125 / 0.075 - none significant. **Recorded as
found.** The one place astro shows up is the COVID window, where the astro-only block beat
technical on BANKNIFTY (AUC 0.719 vs 0.611) and CNXIT (0.647 vs 0.489) - the original
paper's "crisis alpha" hypothesis, on 80-100 observations, flagged as a hypothesis not a
result.

**Corrections to the v1 documents, both recorded in METHODOLOGY.md:**
1. Yahoo serves these tickers only from 2007-09-17, not 1996. The claimed "15,507
   observations spanning 1996-2026" is not obtainable from the stated source.
2. `ML_ProjectReport.pdf` contradicts itself: SS7.2 reports Hybrid AUC 0.5094, SS9.4 reports
   0.9964. v3 produces one reproducible number (0.5143 on NIFTY).

**Status: complete.** All 12 planned phases delivered.

---

### [2026-08-28 23:45] release: GitHub publication, SETUP.md, preparation.md
**Commits:** `3ad7bd1`, `2e9d4fb`, `594c8a1` (pushed as `d4e3152..594c8a1`)
**Added:** `SETUP.md`, `preparation.md`; trimmed `pyproject.toml`; shipped `artifacts/models/`

**Published to** https://github.com/vinitpatil519/ChandraQuant **under `v3/`**, alongside the
existing v1 research at the repository root. All 9 build commits preserved via a
`filter-branch` path rewrite rather than squashed, so the history reads as the project
being built. Root README gained a banner pointing at v3.

**Dependency audit.** `textual`, `typer`, `scipy`, `shap`, `matplotlib`, `plotly`,
`statsmodels`, `tabulate`, `fastapi` and `uvicorn` were all declared and never imported -
the browser dashboard runs on stdlib `http.server` with inline SVG. Removed. Install is
now 12 packages instead of 22.

**Four defects found by installing the published repo from scratch and running it as a
stranger would.** None were visible from inside the development environment:

1. **Fresh clones trained models on first launch.** `artifacts/models/` was gitignored as
   "large" - they are 2 MB each. Now committed; first render drops from minutes to seconds.
2. **The root `.gitignore` silently swallowed them anyway.** v1's ignore file carries broad
   `models/` and `data/` rules that also match `v3/artifacts/models/` and
   `v3/data/snapshot/`. Git will not descend into an excluded directory, so a nested
   `.gitignore` cannot re-include them - the negations had to go at the root. The first
   push landed without the models and I only caught it by counting files afterwards.
3. **The 32 MB ephemeris downloaded in total silence** (`Loader(verbose=False)`), so the
   very first run looked hung. Now announced, with a progress bar.
4. **A truncated ephemeris bricked the install permanently.** The verification run's
   download was interrupted at 15.5 MB; Skyfield finalised the partial file and reused it
   forever, failing with jplephem's opaque `buffer is too small for requested array`.
   `_kernel()` now validates size before loading, deletes anything materially under ~31 MB
   as a partial download and re-fetches, and wraps the load so residual failures explain
   themselves.

**Forward-compatibility verified.** A clean virtualenv resolves **pandas 3.0.5 / numpy
2.5.2** - both newer than the 2.3.3 / 2.3.x developed against - and all 60 tests pass, as
does the dashboard. Worth knowing, since anyone installing today gets pandas 3.

**Final end-to-end check:** fresh `git clone` from GitHub, `pip install -e .`,
`chandraquant --ticker NIFTY --no-refresh` renders the full dashboard in **9 seconds**,
offline, with the header correctly reporting `o snapshot` rather than claiming live data.

**Note for future work.** The local checkout at `C:\ChandraQuant` keeps files at its root
while GitHub has them under `v3/`, so a direct push from it would collide with v1. The
canonical working copy going forward is a clone of the repository, working in
`ChandraQuant/v3/` - which is exactly what SETUP.md instructs.
