# Methodology and Limitations

This document reports what ChandraQuant measured, including — especially including — the
results that do not support the project's own premise. It exists because a system that only
reports its wins is decoration rather than evidence, and because the most interesting
finding here is a negative one.

---

## 1. What the astro layer does not do

**Celestial state does not reliably predict Indian equity index returns at a 5-day horizon.**

Out-of-fold results from purged walk-forward cross-validation (6 folds, 5-day purge,
5-day embargo), target `P(Vṛddhi)`:

| Ticker | Technical AUC | Astro-only AUC | Hybrid AUC | Δ hybrid − technical | Permutation *p* |
|---|---|---|---|---|---|
| NIFTY 50 | 0.5128 | 0.5025 | 0.5143 | **+0.0014** | 0.500 |
| BANKNIFTY | 0.5168 | 0.5012 | 0.5149 | **−0.0019** | 0.125 |
| CNX IT | 0.5347 | 0.5084 | 0.5316 | **−0.0032** | 0.075 |

Read plainly: the astro block sits within a rounding error of chance, the hybrid's edge over
technical-only is indistinguishable from zero, and no permutation test clears p < 0.05.

Bootstrap confidence intervals on Δ(hybrid − technical) straddle zero in all three cases
(NIFTY: +0.0017, CI95 [−0.0091, +0.0149], P(Δ>0) = 0.59).

### Three specific negative findings worth recording

**1. Hand-assigned classical biases were actively harmful.** The first version of
`config/*.yaml` carried a `market_bias` for every nakṣatra, tithi, yoga and karaṇa, assigned
faithfully from the classical descriptions — Puṣya +0.60 as the "nourisher", Mūla −0.55
because it tears out roots. Filtering NIFTY exposure on `ASTRO_SCORE > 0` built from those
priors cut CAGR from **9.2% to 0.1%**. Reading a Sanskrit adjective and guessing a sign does
not produce alpha. They were replaced with empirical-Bayes edges estimated from training
data (`models/astro_edge.py`).

**2. Learned astro edges are not stationary.** With edges *measured* rather than asserted,
a 60/40 holdout gave `ASTRO_EDGE` an AUC of **0.5274** — better than the technical block —
with high-edge days at 37.9% Vṛddhi versus 32.4% on low-edge days. But across six
walk-forward folds the sign of that edge **flips**. A conditional relationship that reverses
between regimes is not a tradeable effect, and reporting only the holdout number would have
been the single most misleading thing this project could do.

**3. The classical abstention rules cost money in this sample.** Gating exposure on Viṣṭi
karaṇa, eclipse windows and Chandrāṣṭama reduced Sharpe from **0.60 to 0.47**. They remove
roughly 27% of trading days in a way that is uncorrelated with returns, so they subtract
exposure to a positively-drifting asset without subtracting risk in proportion.

---

## 2. What the astro layer does do

**It is a genuinely correct, high-precision astronomical engine**, verified against external
ground truth rather than against itself:

- 8/8 exact eclipse dates versus the NASA catalogue for 2025–26, zero false positives
- Makar Saṅkrānti 2025 to 0.009°; Great Conjunction to within one day at 0.04° separation
- Mercury and Saturn retrograde stations within 1–2 days of actual
- Aṣṭakavarga invariants exact (SAV = 337)
- Ascendant validated by a physical identity (Lagna = Sun at sunrise, to 0.9°)

**It is ticker-specific**, which v1's was not. 188 of 831 features differ between indices
on the same calendar day. This is a real architectural fix to a real defect, independent of
whether the resulting features predict anything.

**It is forward-computable.** Every astro feature is a deterministic function of timestamp.
The 30-day forward calendar in the dashboard is a genuine forecast, not a backfit — an
unusual property that no price-derived feature can have.

**It showed crisis-period signal.** During the COVID window the astro-only block reached
AUC 0.719 on BANKNIFTY (technical: 0.611) and 0.647 on CNX IT (technical: 0.489). This is
the original paper's "crisis alpha" hypothesis appearing in the data. **Caveat that matters:**
these windows contain 80–100 observations each, five crisis windows were examined across
three tickers, and results are mixed elsewhere (the Iran-2025 window shows astro
*under*performing on two of three tickers). With ~30 window-ticker combinations examined,
some will look strong by chance. This is a hypothesis worth further work, not a result.

---

## 3. Where the returns come from

Position sizing is **trend + volatility targeting**, with no astro input:

```
position = trend_strength × (target_vol / realised_vol)
trend_strength = distance above EMA, in volatility units, floored and capped
```

Volatility targeting is the load-bearing component. It de-levers into turbulence and
re-levers into calm, which is why maximum drawdown falls by more than CAGR does:

| | Strategy | Buy & Hold | Improvement |
|---|---|---|---|
| NIFTY Calmar | 0.387 | 0.251 | **+54%** |
| BANKNIFTY Calmar | 0.380 | 0.245 | **+55%** |
| CNX IT Calmar | 0.375 | 0.240 | **+56%** |

The consistency across three indices with different characters (broad market, high-beta
financials, USD-sensitive exporters) is the reason to believe it is a property of the method
rather than of a fitted parameter set. Parameters were selected per ticker by
`scripts/optimize.py` maximising Calmar subject to beating buy-and-hold on drawdown — this
is an in-sample selection over a 4×4×3×3×3 grid and carries the usual multiple-testing
caveat.

**Sharpe is not meaningfully improved on NIFTY** (0.660 vs 0.656). The gain is concentrated
in drawdown, not in return per unit of volatility. On BANKNIFTY and CNX IT, Sharpe does
improve (0.66 vs 0.61; 0.68 vs 0.59).

---

## 4. Data, and a correction to the v1 documents

Yahoo Finance serves `^NSEI`, `^NSEBANK` and `^CNXIT` only from **2007-09-17** — verified
2026-08-28. The v1 report and paper both claim a dataset "spanning 1996 to 2026" with
"15,507 observations". That span is not obtainable from the stated source. This project
uses 2007-09-17 onward (~4,650 bars per ticker) and backtests from 2010-01-04 to allow
200-day feature warmup. GFC, COVID and Russia-Ukraine windows are all still covered.

**A second correction.** `ML_ProjectReport.pdf` contradicts itself on its central number:
§7.2 reports Hybrid AUC **0.5094**, while §9.4 reports "the actual Hybrid AUC of **0.9964**"
against a permutation null centred at 0.990. These cannot both be true, and an AUC of 0.9964
on 5-day forward index direction would be a once-in-a-generation result rather than a
footnote. v3 produces a single reproducible number (0.5143 on NIFTY) from code anyone can
run.

---

## 5. Methodological choices

**Purged walk-forward CV.** Six expanding folds. Labels look forward H = 5 bars, so the final
5 bars of every training block are dropped (purge) plus a further 5-bar embargo, preventing
a training label from overlapping its own test period. v1 used a single 2017 cut, which
yields one noisy number and no confidence interval.

**No cross-ticker pooling.** Models are fitted per index. Pooling is what destroyed v1's
astro block.

**Causality proved, not assumed.** `technical.assert_causal` recomputes features on
truncated history and asserts the tail is unchanged; a feature that peeked forward would
differ. Astro features are date-deterministic and leak-free by construction.

**Causal normalisation.** Composite indices are z-scored on *expanding* windows. The Kṣobha
volatility threshold is a *rolling* quantile — with an expanding one the 2008 crisis
dominates permanently and the class collapses to 3.8% of rows instead of the intended ~11%.

**Costs.** 5 bps per side, next-bar-open execution, turnover charged on every rebalance.
Stop fills assume the pessimistic ordering when a bar touches both stop and target.

---

## 6. Honest summary

ChandraQuant is a correct and unusually complete computational implementation of Jyotiṣa —
831 features spanning pañchāṅga, natal charts, daśā, aṣṭakavarga and ṣaḍbala, validated
against NASA data — attached to a competent volatility-targeted trend system that improves
return-per-drawdown by roughly 55% against buy-and-hold across three indices.

The astronomy is real, the engineering is sound, the backtest is clean, and the astrology
does not predict returns. Three of those four are worth putting your name to, and the
fourth is worth knowing.

> *"Grahāṇāṁ ca yathā karma, tathā phalaṁ prāpyate nṛṇām."*
> As the planets execute their motion, so do the fruits of men's endeavours manifest.
> — Bṛhat Parāśara Horā Śāstra, ch. 2

The engine computes the motion exactly. The fruits remain, as ever, harder to pin down.
