"""Single entry point for "what does ChandraQuant say about X today".

Everything the TUI, the web dashboard and the Pine emitter need comes from here, so
there is exactly one code path producing a verdict and no chance of two surfaces
disagreeing.

Two distinct outputs, deliberately kept apart because they answer different questions:

  DETECTED regime   what regime the market is in *now*, computed from realised price
                    over the trailing window using the same four-state definition the
                    labels use. This is a nowcast and involves no model.
  FORECAST          P(Vriddhi) over the next H sessions from the calibrated hybrid
                    model. This is a prediction and carries all the model's
                    uncertainty.

Conflating them would be the easiest way to overstate what the system knows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .astro import ashtakavarga, dasha
from .astro import engine as astro_engine
from .astro.natal import natal_chart
from .backtest import strategy as st
from .config import ARTIFACT_DIR, MODEL_DIR, get_ticker
from .data.loaders import DataStatus
from .explain import narrative as narrative_engine
from .features import dataset as ds
from .labels.regime import (
    KSHAYA,
    KSHOBHA,
    REGIME_COLOUR,
    REGIME_MEANING,
    REGIME_NAMES,
    STHIRA,
    VRIDDHI,
    LabelConfig,
)
from .models.hybrid import HybridModel

MODEL_VERSION = "v3"


@dataclass
class Snapshot:
    """Everything known about one index on one date."""

    ticker: str
    display: str
    date: pd.Timestamp
    status: DataStatus

    detected_regime: int
    detected_name: str
    detected_meaning: str
    detected_colour: str

    probability: float
    astro_probability: float
    tech_probability: float
    astro_weight: float

    position: float
    trend: float
    vol_scalar: float

    close: float
    change_1d: float
    change_20d: float

    astro_row: pd.Series
    chart: dict
    dasha: dict
    sav: pd.DataFrame
    narrative: dict

    prices: pd.DataFrame = field(repr=False, default=None)
    astro_history: pd.DataFrame = field(repr=False, default=None)
    forward: pd.DataFrame = field(repr=False, default=None)
    backtest: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------
# Regime detection (nowcast)
# --------------------------------------------------------------------------------
def detect_regime(prices: pd.DataFrame, when: pd.Timestamp, cfg: LabelConfig | None = None) -> int:
    """Which of the four regimes the market is in *now*, from realised price only."""
    cfg = cfg or LabelConfig()
    close = prices["Close"]
    upto = close.loc[:when]
    if len(upto) < 30:
        return STHIRA
    H = cfg.horizon
    ret = upto.pct_change()

    realised_ret = upto.iloc[-1] / upto.iloc[-(H + 1)] - 1.0
    realised_vol = ret.iloc[-H:].std() * np.sqrt(252)
    trailing_vol = ret.iloc[-20:].std()
    mu = cfg.mu_k * trailing_vol * np.sqrt(H)

    vol_hist = (ret.rolling(H).std() * np.sqrt(252)).dropna()
    vol_thresh = vol_hist.tail(cfg.vol_window).quantile(cfg.kshobha_quantile) if len(vol_hist) else np.inf

    if np.isfinite(vol_thresh) and realised_vol > vol_thresh:
        return KSHOBHA
    if realised_ret > mu:
        return VRIDDHI
    if realised_ret < -mu:
        return KSHAYA
    return STHIRA


# --------------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------------
def model_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker}_{MODEL_VERSION}.joblib"


def load_or_train(ticker: str, data: ds.Dataset, force: bool = False) -> HybridModel:
    """Load a cached fitted model, or train and cache one."""
    path = model_path(ticker)
    if path.exists() and not force:
        try:
            return joblib.load(path)
        except Exception:
            pass
    model = HybridModel()
    model.fit_walk_forward(data)
    model.fit_full(data)
    joblib.dump(model, path)
    return model


def load_backtest_metrics() -> dict:
    path = ARTIFACT_DIR / "metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# --------------------------------------------------------------------------------
# Forward astro calendar
# --------------------------------------------------------------------------------
EVENT_COLUMNS = [
    ("karana_is_vishti", "Vishti (Bhadra) karana - abstention window"),
    ("eclipse_window_3d", "Grahana window - eclipse within three days"),
    ("moon_in_gandanta", "Moon in Gandanta - water-fire junction"),
    ("nat_chandrashtama", "Chandrashtama - Moon 8th from natal Chandra"),
    ("tithi_is_rikta", "Rikta tithi - barren for commerce"),
    ("is_purnima", "Purnima - full moon"),
    ("is_amavasya", "Amavasya - new moon"),
    ("Budha_stambhana", "Budha at Stambhana - Mercury stationary"),
    ("sankranti_window", "Sankranti - solar ingress"),
]


def forward_calendar(ticker: str, start: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    """The next N days of celestial events - genuinely forward, not backfitted.

    This is the one thing in the system that is a real forecast rather than a fit:
    every astro quantity is a deterministic function of the date, so the calendar for
    next month is exactly as certain as the calendar for last month.
    """
    idx = pd.date_range(start, start + pd.Timedelta(days=days), freq="D")
    astro = astro_engine.build(idx, ticker)
    rows = []
    for date, row in astro.iterrows():
        events = []
        for col, label in EVENT_COLUMNS:
            if col in row and pd.notna(row[col]) and float(row[col]) > 0.5:
                events.append(label)
        rows.append(
            {
                "date": date,
                "nakshatra": row.get("nakshatra_name", ""),
                "tithi": row.get("tithi_name", ""),
                "tara": int(row.get("nat_tarabala", 0) or 0),
                "CBI": float(row.get("CBI", 0.0)),
                "VRI": float(row.get("VRI", 0.0)),
                "BHY": float(row.get("BHY", 0.5)),
                "KTW": float(row.get("KTW", 0.0)),
                "events": events,
            }
        )
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------------
# The main entry point
# --------------------------------------------------------------------------------
def snapshot(
    ticker: str,
    when: str | pd.Timestamp | None = None,
    refresh: bool = True,
    retrain: bool = False,
    with_forward: bool = True,
) -> Snapshot:
    """Build the complete state for one index on one date."""
    entry = get_ticker(ticker)
    key = entry["key"]

    data = ds.build(key, refresh=refresh)
    prices = data.prices

    ts = pd.Timestamp(when).normalize() if when is not None else prices.index[-1]
    if ts not in prices.index:
        eligible = prices.index[prices.index <= ts]
        if len(eligible) == 0:
            raise ValueError(f"No market data on or before {ts.date()} for {key}")
        ts = eligible[-1]

    model = load_or_train(key, data, force=retrain)

    # Score the row. predict() needs the same columns the model was fitted on.
    X = data.features.loc[[ts]]
    try:
        pred = model.predict(X).iloc[0]
        p_hat = float(pred["p_calibrated"])
        p_astro = float(pred["p_astro"])
        p_tech = float(pred["p_tech"])
        a_weight = float(pred["astro_weight"])
    except Exception:
        p_hat = p_astro = p_tech = float("nan")
        a_weight = 0.0

    # Strategy position for the day.
    astro_full = astro_engine.build(prices.index, key)
    params = _strategy_params(key)
    comp = st.build_position(prices, astro_full, params)
    pos_row = comp.loc[ts] if ts in comp.index else comp.iloc[-1]

    detected = detect_regime(prices, ts)
    astro_row = astro_full.loc[ts]
    chart = natal_chart(key)
    dsh = dasha.describe(key, ts)

    narr = narrative_engine.build(
        astro_row,
        entry["display"],
        detected,
        p_hat if np.isfinite(p_hat) else 0.5,
        dsh,
    )

    close = float(prices["Close"].loc[ts])
    hist = prices["Close"].loc[:ts]
    chg1 = float(hist.pct_change().iloc[-1]) if len(hist) > 1 else 0.0
    chg20 = float(hist.iloc[-1] / hist.iloc[-21] - 1.0) if len(hist) > 21 else 0.0

    return Snapshot(
        ticker=key,
        display=entry["display"],
        date=ts,
        status=data.status,
        detected_regime=detected,
        detected_name=REGIME_NAMES[detected],
        detected_meaning=REGIME_MEANING[detected],
        detected_colour=REGIME_COLOUR[detected],
        probability=p_hat,
        astro_probability=p_astro,
        tech_probability=p_tech,
        astro_weight=a_weight,
        position=float(pos_row["position"]),
        trend=float(pos_row["trend"]),
        vol_scalar=float(pos_row["vol_scalar"]),
        close=close,
        change_1d=chg1,
        change_20d=chg20,
        astro_row=astro_row,
        chart=chart,
        dasha=dsh,
        sav=ashtakavarga.describe(key),
        narrative=narr,
        prices=prices,
        astro_history=astro_full,
        forward=forward_calendar(key, ts, 30) if with_forward else None,
        backtest=load_backtest_metrics().get(key, {}),
    )


def _strategy_params(ticker: str) -> st.StrategyParams:
    """Per-ticker tuned parameters from config/strategy.yaml, if present."""
    import yaml

    from .config import CONFIG_DIR

    path = CONFIG_DIR / "strategy.yaml"
    p = st.StrategyParams(use_astro=False)
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        tuned = (cfg.get("tickers") or {}).get(ticker)
        if tuned:
            p.target_vol = float(tuned.get("target_vol", p.target_vol))
            p.leverage_cap = float(tuned.get("leverage_cap", p.leverage_cap))
            p.trend_span = int(tuned.get("trend_span", p.trend_span))
            p.vol_window = int(tuned.get("vol_window", p.vol_window))
            p.smooth_days = int(tuned.get("smooth_days", p.smooth_days))
        p.cost_bps = float(cfg.get("cost_bps", p.cost_bps))
    return p


def quick_status(ticker: str, refresh: bool = False) -> dict:
    """Cheap one-line status for the ticker picker - no model, no narrative."""
    entry = get_ticker(ticker)
    key = entry["key"]
    data = ds.build(key, refresh=refresh)
    ts = data.prices.index[-1]
    regime = detect_regime(data.prices, ts)
    astro = astro_engine.build(data.prices.index[-40:], key)
    row = astro.loc[ts]
    return {
        "key": key,
        "display": entry["display"],
        "yahoo": entry["yahoo"],
        "date": ts,
        "regime": regime,
        "regime_name": REGIME_NAMES[regime],
        "colour": REGIME_COLOUR[regime],
        "close": float(data.prices["Close"].loc[ts]),
        "change_1d": float(data.prices["Close"].pct_change().loc[ts]),
        "nakshatra": str(row.get("nakshatra_name", "")),
        "tithi": str(row.get("tithi_name", "")),
        "CBI": float(row.get("CBI", 0.0)),
        "BHY": float(row.get("BHY", 0.5)),
        "source": data.status.source,
    }
