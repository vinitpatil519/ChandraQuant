# ChandraQuant — Interview Preparation

27 questions covering the whole project, with short answers. Read the numbers carefully —
being able to quote them precisely is what separates "I built a thing" from "I measured a
thing."

**The one-line summary to have ready:** *A terminal quant engine that computes 831 Vedic
astronomical features from NASA ephemerides, each anchored to a per-index birth chart, and
uses them to detect and narrate market regimes for three Indian indices — with a
volatility-targeted trend strategy that improves return-per-drawdown by ~55% over
buy-and-hold.*

---

## Part 1 — The project at a glance

**Q1. Give me a sixty-second overview of ChandraQuant.**
It's a regime-detection engine for NIFTY 50, BANKNIFTY and CNX IT. It computes 831
astronomical features per day from NASA JPL DE440s ephemerides — pañchāṅga, planetary
dignity, eclipses, daśā periods — and expresses each one relative to a birth chart cast for
that index's launch date. A LightGBM pair plus a volatility-targeted trend system turns
that into a regime call, a probability, and a plain-English explanation, delivered in a
terminal dashboard. It also ships two TradingView Pine scripts that compute the astronomy
live on the chart.

**Q2. Why astrology? Isn't this pseudoscience?**
I treated Jyotiṣa as a *timekeeping system*, not a belief system. Stripped of metaphysics it
is a set of deterministic, arc-second-precision cyclic coordinates — the Moon's 27
nakṣatras, the 30 tithis, the 120-year Vimśottarī cycle — that no market participant uses as
features, and which have one property no technical indicator has: they're computable
arbitrarily far into the future. The interesting question isn't whether planets influence
markets; it's whether an orthogonal, exactly-known clock carries information. I built the
instrument to test that, and I report what it measured.

**Q3. Who is it for, and what problem does it solve?**
SEBI found 93% of Indian retail F&O traders lost money over FY22–24, largely because they
run lagging indicators against institutions running algorithms. ChandraQuant gives a retail
user a probabilistic *regime* view rather than a buy/sell signal, plus a risk overlay that
cuts drawdown roughly in half. The real deliverable is disciplined position sizing wrapped
in an interface people will actually open.

**Q4. There was an earlier version. What did you change?**
v1 fed a Random Forest 10 astro features — Moon longitude, Sun longitude, tithi, nakṣatra
and harmonics. Every one was a function of *calendar date alone*, so pooling three tickers
meant identical astro rows against three different labels; the astro block was
mathematically incapable of discriminating between indices. It measured ΔAUC of **−0.018** —
the hybrid was *worse* than technical-only. v3 fixes that with natal charts, and grows the
feature set from 10 to 831.

---

## Part 2 — The terminal application

**Q5. Walk me through exactly what happens when you type `chandraquant`.**
The name is a console entry point declared in `pyproject.toml`, mapping to
`chandraquant.cli.app:main`. `main()` parses flags, then plays a Rich `Live` animation — an
ASCII Kāla-Chakra wheel with nine grahas orbiting at their real relative speeds. Then
`picker.choose()` calls `inference.quick_status()` for each of the three tickers (a cheap
path: no model, no narrative) and renders them through `questionary` as an arrow-key list.
Once you pick one, `inference.snapshot()` builds the full state and `tui/dashboard.render()`
prints the panels. It ends in a small command loop — `r` refresh, `b` back, `w` web, `q`
quit — so it behaves like an app rather than a script.

**Q6. Is it a TUI framework like Textual, or something simpler?**
Deliberately simpler. It's the **Rich** library rendering composed panels into a normal
terminal — `Table.grid` for layout, custom block-character sparklines and meters in
`tui/widgets.py`. I started with Textual as a dependency and removed it: a full-screen
event-loop TUI adds a rendering thread and terminal-mode switching for no gain here, and a
printed dashboard also scrolls back, pipes, and screenshots cleanly. It adapts to width —
three panels across at 150+ columns, stacking below that.

**Q7. Why a terminal app rather than a web dashboard?**
The audience is traders and quants who already live in a terminal, and it starts in about a
second with no server, no browser, no build step. It's also the honest fit for the
architecture — everything is a Python function returning a `Snapshot` dataclass, so the UI
is a thin renderer. There *is* a browser dashboard on `--web`, and it renders the same
`Snapshot` object, so the two surfaces can't disagree with each other.

**Q8. What happens if the machine has no internet?**
It still works. `data/loaders.py` attempts a Yahoo refresh on a worker thread with an
8-second hard timeout and falls back silently to a parquet snapshot committed to the repo.
The header then reads `○ cached (2026-08-28)` instead of `● live` — I show the data
provenance rather than quietly pretending it's fresh. The astronomy needs no network at all
after the one-time 32 MB ephemeris download.

---

## Part 3 — The astronomy engine

**Q9. Where do the planetary positions come from?**
NASA JPL's **DE440s** ephemeris, accessed through the Skyfield library. I take apparent
geocentric ecliptic longitudes — light-time and aberration corrected, which is what Indian
pañchāṅga convention requires — and compute speeds by central difference so retrogression
falls out naturally. Rāhu and Ketu are the lunar nodes, computed as the mean node per Indian
convention, with the osculating true node available for comparison.

**Q10. What's the ayanamsa, and why does it matter?**
Western astrology uses the tropical zodiac (tied to the equinox); Jyotiṣa uses the sidereal
one (tied to the stars). They've drifted about 24° apart due to precession. The ayanamsa is
that offset. I implement **Lahiri (Chitrapakṣa)**, the Indian government standard, anchored
at 23.250182° on JD 2435553.5 and accumulating IAU-2006 precession. It reads 24.2341° for
August 2026, matching what any Indian pañchāṅga prints.

**Q11. How do you know your astronomy is actually correct?**
Every check is against a source outside the repo, not self-consistency. Eclipse detection
matches the **NASA catalogue 8/8 exactly for 2025–26 with zero false positives**. Makar
Saṅkrānti 2025 lands with the Sun at sidereal 270.009° — precisely entering Makara. The
Great Conjunction bottoms out on 2020-12-22 at 0.04° separation. My ascendant formula is
validated by a physical identity: at sunrise the Lagna must equal the Sun's longitude, and
it does to 0.9°, which is exactly the refraction offset that *defines* sunrise.

**Q12. What is a "natal chart for an index", and why is it the core of the project?**
An index is born the day it starts publishing — NIFTY on 1996-04-22, BANKNIFTY on
2003-09-15, CNX IT on 1999-05-11. Cast a chart for that moment in Mumbai and every celestial
position becomes a *transit relative to that chart*. That single change makes astro features
ticker-specific: on 2026-08-28 NIFTY sits in **Sampat tārā** (+0.9, wealth-giving) while
BANKNIFTY sits in **Pratyari** (−0.7) and under Sade Sati. **188 of 831 features now differ
between tickers; in v1 that number was zero.** There's a named regression test guarding it.

**Q13. Name some of the 831 features.**
Twelve modules. Pañchāṅga (tithi, nakṣatra with pādas, yoga, karaṇa, vāra); lunar mechanics
(phase, latitude, gaṇḍānta, speed, tidal force); graha state (exaltation, retrogression,
combustion, planetary war); Parāśarī aspects plus orb-decayed angular kernels; eclipses and
ingresses with days-to/days-since ramps; natal-relative Gochara, Tārābala, Chandrabala, Sade
Sati; Vimśottarī daśā; Aṣṭakavarga bindu scores; and Ṣaḍbala six-fold strength in rūpas.
They collapse into five branded indices: CBI, GSI, VRI, BHY and Kāla-Taraṅga.

**Q14. Angles wrap at 360°. How do you handle that?**
Cyclic encoding — every angular variable is projected onto the unit circle as sin/cos pairs,
with second and third harmonics so a tree model can resolve sub-cycle structure. Without it,
359° and 1° look maximally distant to the model when they're actually adjacent, which
creates an artificial discontinuity right in the middle of a nakṣatra.

---

## Part 4 — Machine learning

**Q15. What are you actually predicting? Define the target precisely.**
`P(Vṛddhi)` — the probability that over the next 5 trading sessions the index posts a
forward return above a volatility-scaled threshold *without* forward volatility entering its
top decile. The horizon is 5 days deliberately: the Moon spends ~2.5 days in a nakṣatra and
~2.3 in a rāśi, so 5 days is the window over which a lunar state is roughly constant.

**Q16. Why four regimes instead of binary up/down?**
Because a quiet drift up and a violent whipsaw that happens to close higher are not the same
thing, and no one should hold the same position through both. So: **Vṛddhi** (expansion),
**Sthira** (consolidation), **Kṣaya** (decline), **Kṣobha** (turbulence, defined by forward
volatility regardless of direction). Base rates on NIFTY are roughly 35/29/25/10.

**Q17. How do you split train and test?**
Purged walk-forward cross-validation — six expanding folds. Two corrections most backtests
omit: the label at bar *t* looks 5 bars ahead, so the final 5 bars of every training block
are **purged**, plus a 5-bar **embargo** after it. Without purging, a training label
literally overlaps the test period's returns. v1 used a single 2017 cut, which gives one
noisy number and no confidence interval.

**Q18. How do you prove there's no lookahead?**
Three ways. `technical.assert_causal()` recomputes every indicator on truncated history and
asserts the tail is unchanged — a feature that peeked forward would differ. Astro features
are deterministic functions of timestamp, so they're leak-free by construction (and I assert
it anyway). And all normalisation is causal: composite indices use *expanding* z-scores, and
the Kṣobha volatility threshold uses a *rolling* quantile, never full-sample.

**Q19. Which model, and why?**
LightGBM, in two separate blocks — one on 62 technical features, one on ~520 astro features
— rather than one pooled model. Gradient boosting handles non-linear interactions and
collinearity without distributional assumptions, and trains in seconds so walk-forward is
cheap. Each block is fit twice: once on everything to rank features by gain, then refit on
the top-K. With 520 astro features against ~4,600 rows, that selection step is what stops
the astro block memorising its training era. Output is calibrated with isotonic regression
on out-of-fold predictions only.

**Q20. How is the astro layer wired into the decision?**
Architecturally, not as a peer feature — that's what failed in v1. Astro sets the blend
weight between the two models via a sigmoid over the composite indices, modulates the entry
threshold (expressed as a causal percentile, not an absolute probability), and hard-vetoes
entries during Viṣṭi karaṇa, eclipse windows and Chandrāṣṭama. Every part of the rule can be
stated in Sanskrit, which is what the narrative engine and the Pine port both need.

**Q21. How does the "why" explanation work — is there an LLM in there?**
No, and deliberately. It's a rule-and-template engine that scores every notable celestial
condition by deviation and classical weight, splits them into supporting and opposing, and
renders the top few as sentences with a 90-term Sanskrit glossary attached. It has to run
offline, instantly, and produce identical output for identical input so it can be tested. An
LLM would be slower, non-deterministic and untestable.

---

## Part 5 — Results, and the honest answer

**Q22. What are your results?**
2010–2026, daily bars, 5 bps per side, next-bar execution. Calmar — return per unit of
drawdown — is **0.387 / 0.380 / 0.375** against a buy-and-hold benchmark of **0.251 / 0.245 /
0.240**, so roughly **55% better on all three indices**. Max drawdown is nearly halved:
−21.6% vs −38.4% on NIFTY, −22.3% vs −47.9% on BANKNIFTY. CNX IT also beats on return,
15.06% CAGR vs 10.57%. Every figure is generated into `artifacts/metrics.json`; nothing is
typed by hand.

**Q23. Your AUC is 0.51. Isn't the model useless?**
0.51 is roughly what 5-day index direction *should* look like — this is a near-efficient
market, and anyone reporting 0.9 on this problem has leaked something. The returns don't
come from classification accuracy; they come from volatility targeting, which de-levers into
turbulence and re-levers into calm. That's why drawdown improves far more than CAGR does.
I'd rather show a defensible 0.51 than an AUC that collapses under inspection.

**Q24. So does astrology predict the market? Give me the straight answer.**
No — not reliably, on this data. Ablation deltas are +0.0014 / −0.0019 / −0.0032, permutation
tests give p = 0.50 / 0.13 / 0.08, none significant. I found three specific negatives worth
knowing: hand-assigned classical biases were *actively harmful* (NIFTY CAGR 9.2% → 0.1%);
learned edges beat the technical block on a holdout at AUC 0.5274 but **flip sign across
walk-forward folds**, so they aren't stationary; and the classical abstention rules cost
Sharpe 0.60 → 0.47. Position sizing is therefore trend plus volatility targeting, and the
Jyotiṣa engine drives the regime narrative, the forward calendar and the visuals.

**Q25. Then what's the point? Why not drop the astrology?**
Three things survive. The engine is a correct and unusually complete piece of astronomical
software, verified against NASA. The forward calendar is a genuine forecast rather than a
fit — astro features are functions of time alone, so next month's celestial calendar is
exactly as certain as last month's, which no price-derived feature can claim. And there is
one real signal: during the COVID crash the astro-only block reached **AUC 0.719 on
BANKNIFTY against the technical model's 0.611**, and 0.647 vs 0.489 on CNX IT — the "crisis
alpha" hypothesis appearing in data. On 80–100 observations across ~30 window-ticker
combinations, I flag that as a hypothesis, not a result.

**Q26. What is the empirical-Bayes edge model, and why did you need it?**
My first pass hard-coded a market bias for each nakṣatra straight from the classical texts —
Puṣya +0.60 as the "nourisher", Mūla −0.55 because it uproots. Backtested, those priors
destroyed returns. So I replaced assertion with measurement: for each celestial state I
estimate the conditional win rate from training data and shrink it toward the global mean
with 60 pseudo-observations, so a state needs repeated evidence before its edge moves. Rare
states collapse to the base rate and contribute nothing, which is correct behaviour.

**Q27. How do you know you're not overfitting?**
Purged walk-forward rather than a single split; per-ticker models with no pooling; bootstrap
confidence intervals that straddle zero and are reported as such; permutation tests reported
as non-significant. The strategy has five tuned parameters selected on Calmar, and I state
plainly in `METHODOLOGY.md` that this is in-sample selection over a grid with the usual
multiple-testing caveat. The strongest evidence is consistency: the same method gives
Calmar +54%, +55% and +56% on three indices with very different characters.

---

## Part 6 — Engineering

**Q28. The TradingView script has no ephemeris. How does it know where the planets are?**
It computes them. I implemented Meeus' periodic series directly in Pine — 27 lunar longitude
terms, 12 latitude terms, the solar equation of centre — then converted to sidereal with an
analytic Lahiri polynomial. Validated against Skyfield over 1,253 samples spanning 2007–2030:
Sun error 0.0095°, Moon 0.0236°, and **100% agreement on nakṣatra, tithi and rāśi
classification**. Only Jupiter, Saturn and Rāhu use a small embedded table, because they move
under 0.1°/day.

**Q29. How do you test something like this?**
60 tests, and every astronomical assertion is anchored outside the repo: the NASA eclipse
catalogue, Drik Pañchāṅga dates for Diwali 2024 and 2025, the classical Aṣṭakavarga totals
(48/49/39/54/56/52/39 summing to exactly 337), Mercury and Saturn retrograde stations. Plus
pipeline tests — causality proofs, purge/embargo verification, a check that the backtester
never fills outside a bar's high-low range, and a named regression test for v1's
ticker-identity bug.

**Q30. What was the hardest bug you found?**
Two are worth telling. Eclipse detection initially fired on every near-node syzygy — 89
detections against ~40 real — because daily sampling is far too coarse when the Moon's
latitude swings through a node in under 48 hours. I fixed it by bracketing each syzygy,
interpolating the exact instant, and applying the true ecliptic limit. The subtler one: my
strategy module used a *binary* trend gate while the optimiser tuned a *continuous* one, so
CNX IT's tuned parameters were being applied to a different formula — worth 19 Sharpe points.

**Q31. Why did you reject the high win-rate version?**
I built it and measured it: a 0.8-ATR profit target gives an **84.7% win rate at profit
factor 0.98 and negative CAGR**. That's the same shape as the old version's advertised 92.6%
— 738 wins averaging +0.46% against 57 losses averaging −5.33%, profit factor 1.12, barely
profitable. TradingView prints Net Profit directly beside Win Rate, so that screenshot
collapses the moment anyone looks. The shipped default is trend mode: ~30% win rate, profit
factor 1.8–2.1, drawdown −16.5% versus buy-and-hold's −38.4%.

**Q32. What are the limitations, and what would you do next?**
Data starts 2007-09-17, not 1996 — Yahoo doesn't serve these indices earlier, which is a
correction to my own earlier write-up. It's long-only and daily; there's no intraday or
short side. The astro conditional edges aren't stationary, which is the central open
problem. Next I'd backfill to 1996 via NSE CSVs to capture two more full cycles, test on
sector indices where a natal chart is better defined, and investigate the crisis-window
result properly with a pre-registered window list rather than post-hoc slicing.

---

## Things to say, and things not to say

**Lead with:** the natal-chart insight (it's a genuine architectural fix to a measurable
defect), the NASA-verified astronomy, and the Calmar improvement.

**Don't oversell:** never claim astrology predicts returns. You measured it; it doesn't;
saying so is the strongest thing you can do in the room. An interviewer who probes and finds
you already knew is a different conversation from one who catches you.

**If asked "would you trade this?":** the volatility-targeting layer, yes — it's a standard,
well-documented technique and the drawdown reduction is real. The astro layer, not as an
alpha source; as a forward-computable calendar of scheduled events it's genuinely useful for
scenario planning.

**If asked "what did you learn?":** that the interesting result was the negative one, and
that building the instrument well enough to trust the measurement was most of the work.
