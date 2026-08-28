
# ChandraQuant Alpha Engine

**Where Jyotiṣa Meets Quantitative Alpha**

> A hybrid quantitative trading framework that combines technical analysis, machine learning, and astronomical cyclic embeddings to probabilistically detect bullish market regimes in Indian equity indices.

---

# 1. Project Overview

ChandraQuant Alpha Engine is an interdisciplinary quantitative finance project built around one central question:

**Can deterministic astronomical cycles provide measurable exogenous information for market regime prediction when combined with modern technical indicators and machine learning?**

Rather than using astrology as a mystical forecasting tool, the project converts **Vedic astronomical concepts (Jyotiṣa)** into mathematically continuous ML features and evaluates them using rigorous statistical validation and financial backtesting.

The system predicts the **probability of a bullish regime** over the next few trading sessions for:

- NIFTY 50
- BANKNIFTY
- SENSEX

using approximately **30 years (1996–2026)** of historical market data together with live planetary ephemeris data from **NASA JPL DE421**.

---

# 2. The Core Idea

Traditional retail trading relies on **lagging indicators**:

- MACD
- RSI
- EMA
- ATR
- Breakouts

These indicators react **after** price has already begun moving.

Institutional traders, however, often trade using:

- Liquidity
- Order Flow
- Fair Value Gaps
- Volatility regimes
- Market structure

ChandraQuant introduces a third layer:

> **Astronomical temporal state variables**.

Instead of predicting price directly, the model attempts to estimate **when the market is statistically more likely to transition into a bullish momentum regime.**

---

# 3. System Architecture

```text
Historical OHLC Data
          │
          ▼
 Technical Indicators
(MACD • RSI • ATR • EMA)
          │
          ├───────────────┐
          │               │
          ▼               ▼
NASA JPL Ephemeris    Market Features
(Moon • Sun • Jupiter)
          │
          ▼
Astro Feature Engineering
(Tithi • Nakshatra • Moon Phase)
          │
          ▼
Cyclic Harmonic Encoding
(sin/cos Fourier Embeddings)
          │
          ▼
Hybrid Feature Matrix
          │
          ▼
Random Forest Classifier
          │
          ▼
P(Bullish Regime)
          │
          ▼
Trading Signals + Backtesting
```

---

# 4. Where Astrology Comes Into Play

This project uses **Jyotiṣa as an astronomical timing system**, not as horoscope prediction.

The astronomical engine computes planetary positions using **NASA's Jet Propulsion Laboratory (JPL DE421)** dataset and converts them into Vedic constructs.

## 4.1 Chandra (Moon)

The Moon represents the fastest-moving celestial body and forms the primary cyclic variable.

Feature extracted:

- Geocentric Moon Longitude

This becomes the foundation for lunar state modeling.

---

## 4.2 Tithi (तिथि)

Tithi represents the angular separation between the Moon and the Sun.

Mathematically:

```text
Tithi = (Moon Longitude − Sun Longitude) mod 360°
```

Instead of treating Tithi as a categorical variable, it is encoded continuously:

```text
sin(Tithi)
cos(Tithi)
```

This preserves cyclic continuity and allows machine learning to understand periodic behavior.

---

## 4.3 Nakṣatra (नक्षत्र)

The Moon's orbit is divided into **27 equal sectors**.

Each sector spans:

```text
360° / 27 = 13°20'
```

Nakshatra index:

```text
Nak = floor(Moon Longitude / 13°20')
```

Rather than using integers 0–26, ChandraQuant embeds Nakshatra into harmonic space:

```text
sin(2πn/27)
cos(2πn/27)
```

This converts ancient Jyotiṣa into a continuous numerical representation.

---

## 4.4 Chandra-Kalā Harmonics

Moon phase itself is represented using Fourier-style cyclic embeddings.

The ML model therefore never sees labels like:

- Full Moon
- New Moon

Instead it learns smooth periodic geometry.

---

# 5. Technical Indicators Used

The endogenous market structure consists of:

| Indicator | Purpose |
|-----------|----------|
| MACD | Momentum acceleration |
| RSI | Trend strength |
| ATR | Volatility expansion |
| EMA | Market trend alignment |
| Breakout Filter | Momentum persistence |

These features describe **what the market is doing**.

The astronomical features describe **when cyclical temporal conditions occur**.

---

# 6. Machine Learning Model

The final feature matrix contains both technical and astro-cyclic variables.

## Inputs

Technical:

- MACD
- RSI
- ATR
- EMA
- Histogram
- Momentum

Astronomical:

- Moon longitude
- Sun longitude
- Tithi
- Nakshatra
- Moon phase
- Harmonic embeddings

## Model

- Random Forest Classifier
- Time-series chronological split
- No data shuffling
- Probabilistic output

The target variable is:

> **Will the index enter a bullish momentum regime within the next H trading sessions?**

Output:

```text
P(Bullish Regime)
```

instead of Buy/Sell directly.

---

# 7. Bullish Regime Logic

A bullish regime is defined using momentum confirmation rather than raw price prediction.

The model learns relationships between:

- Positive MACD acceleration
- RSI strength
- Breakout persistence
- Volatility expansion
- Astro-cyclic timing

The prediction is therefore probabilistic:

```text
0.82 → High bullish confidence
0.31 → Low bullish confidence
```

This probability later becomes the TradingView trading signal.

---

# 8. Backtesting Pipeline

After training, the model is deployed as a **TradingView Pine Script strategy**.

The strategy generates algorithmic LONG entries using:

- EMA trend filter
- MACD confirmation
- RSI momentum
- Volatility breakout
- Astro timing layer

Performance is evaluated using:

- Win Rate
- CAGR
- Sharpe Ratio
- Maximum Drawdown
- Equity Curve

This validates whether the probabilistic model produces economically meaningful signals.

---

# 9. Validation Framework

ChandraQuant does not rely solely on ROC-AUC.

The project includes institutional-grade validation.

## 9.1 Historical Backtesting

Evaluates live strategy performance on NIFTY using TradingView.

Metrics:

- Win rate
- Drawdown
- CAGR
- Sharpe

---

## 9.2 Bootstrap Testing

Repeated resampling estimates confidence intervals for model stability.

Purpose:

- Robustness
- Statistical reliability

---

## 9.3 Permutation Testing

Astro features are randomly shuffled.

If predictive performance collapses, it indicates that astronomical embeddings contain genuine informational structure rather than random noise.

---

## 9.4 Monte Carlo Simulation

Thousands of randomized equity paths are generated to estimate:

- Probability of profit
- Tail risk
- Capital dispersion
- Equity uncertainty

---

## 9.5 Crisis-Regime Validation

The hybrid model is tested separately during major market shocks.

Evaluated regimes include:

- 2008 Global Financial Crisis
- 2020 COVID Crash
- Russia–Ukraine volatility regime
- Iran–US geopolitical escalation

The objective is to determine whether astro-cyclic features become more useful during periods dominated by fear and uncertainty.

---

# 10. Why This Project Is Different

Unlike conventional trading systems:

- It does **not** attempt deterministic price prediction.
- It models **market regimes** instead of candles.
- It combines **endogenous** (technical) and **exogenous** (astronomical) variables.
- It converts Sanskrit Jyotiṣa concepts into machine-learning compatible harmonic embeddings.
- It validates performance using both statistical and financial methodologies.

The project represents an intersection of:

- Quantitative Finance
- Machine Learning
- Financial Time-Series Analysis
- Astronomical Computation
- Vedic Cyclic Mathematics
- Algorithmic Trading

---

# 11. Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Astronomy | Skyfield + NASA JPL DE421 |
| Visualization | Plotly, Matplotlib |
| Backtesting | TradingView Pine Script |
| Research | Monte Carlo, Bootstrap, Permutation Testing |

---

# 12. Project Summary

**ChandraQuant Alpha Engine** demonstrates how ancient cyclic astronomical systems can be translated into rigorous computational representations and evaluated objectively within modern quantitative finance.

Its primary contribution is not proving that astrology predicts markets, but establishing a reproducible methodology for testing whether **astronomical temporal cycles contain weak yet measurable exogenous information capable of improving probabilistic market-regime detection.**
