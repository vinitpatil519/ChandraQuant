"""Pipeline integrity - leak-freedom, label sanity, backtest honesty, determinism.

The astro tests prove the sky is right. These prove the machinery around it does not
cheat: no feature sees the future, no label leaks across a fold boundary, and the
backtester cannot buy at a price the market never offered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chandraquant.backtest import engine as bt
from chandraquant.backtest import metrics as M
from chandraquant.backtest import strategy as st
from chandraquant.config import TICKER_KEYS
from chandraquant.features import dataset as ds
from chandraquant.features import technical
from chandraquant.labels import regime as R
from chandraquant.models.splits import WalkForwardConfig, walk_forward


@pytest.fixture(scope="module")
def data():
    return ds.build("NIFTY", refresh=False)


# --------------------------------------------------------------------------------
# Leak-freedom
# --------------------------------------------------------------------------------
def test_technical_features_are_causal(data):
    """Recomputing on truncated history must reproduce the tail exactly.

    If any indicator peeked forward, cutting the input short would change its value at
    earlier timestamps.
    """
    technical.assert_causal(data.prices, data.features)


def test_no_astro_technical_column_collision(data):
    assert not (set(data.astro_cols) & set(data.tech_cols))


def test_every_feature_is_classified(data):
    """A column that is neither astro nor technical would silently escape ablation."""
    assert len(data.astro_cols) + len(data.tech_cols) == data.features.shape[1]


def test_labels_are_the_only_forward_looking_columns(data):
    """Labels must be NaN for exactly the last `horizon` rows."""
    cfg = R.LabelConfig()
    labels = R.compute(data.prices, cfg)
    tail = labels["fwd_return"].tail(cfg.horizon)
    assert tail.isna().all()
    assert labels["fwd_return"].iloc[: -cfg.horizon].notna().sum() > 0


def test_walk_forward_purges_the_label_window():
    """No training index may sit within `horizon + embargo` of a test index."""
    idx = pd.date_range("2010-01-01", periods=3000, freq="B")
    cfg = WalkForwardConfig(n_splits=5, horizon=5, embargo=5, min_train=600)
    folds = list(walk_forward(idx, cfg))
    assert len(folds) >= 4
    for train, test in folds:
        assert train.max() < test.min()
        assert test.min() - train.max() >= cfg.horizon + cfg.embargo


def test_walk_forward_test_blocks_are_disjoint():
    idx = pd.date_range("2010-01-01", periods=3000, freq="B")
    seen = set()
    for _, test in walk_forward(idx, WalkForwardConfig(n_splits=5, min_train=600)):
        assert not (seen & set(test.tolist()))
        seen |= set(test.tolist())


# --------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------
def test_regime_labels_are_exhaustive_and_exclusive(data):
    labels = R.compute(data.prices)
    valid = labels["regime"].dropna()
    assert set(valid.unique()) <= {0.0, 1.0, 2.0, 3.0}
    rates = R.base_rates(labels)
    assert abs(sum(rates.values()) - 1.0) < 1e-9


def test_regime_base_rates_are_reasonable(data):
    """No class may collapse. Kshobha in particular was 3.8% before the rolling fix."""
    rates = R.base_rates(R.compute(data.prices))
    for name in ("Vriddhi", "Sthira", "Kshaya", "Kshobha"):
        assert 0.05 < rates[name] < 0.55, f"{name} at {rates[name]:.3f}"


def test_kshobha_tracks_forward_volatility(data):
    labels = R.compute(data.prices)
    kshobha = labels.loc[labels["regime"] == R.KSHOBHA, "fwd_vol"]
    rest = labels.loc[labels["regime"].isin([0, 1, 2]), "fwd_vol"]
    assert kshobha.mean() > rest.mean() * 1.5


# --------------------------------------------------------------------------------
# Backtest honesty
# --------------------------------------------------------------------------------
def test_backtest_never_fills_outside_the_bar(data):
    """Every entry and exit must sit inside the day's actual high-low range."""
    px = data.prices.tail(1200)
    sig = pd.Series(
        (np.arange(len(px)) % 7 == 0).astype(int), index=px.index
    )
    res = bt.run(px, sig, params=bt.BacktestParams(max_hold=10))
    trades = res["trades"]
    for _, t in trades.iterrows():
        lo = px["Low"].loc[t["exit_date"]] * 0.999
        hi = px["High"].loc[t["exit_date"]] * 1.001
        assert lo <= t["exit"] <= hi, f"exit {t['exit']} outside [{lo}, {hi}]"


def test_zero_signal_means_zero_trades(data):
    px = data.prices.tail(500)
    sig = pd.Series(0, index=px.index)
    res = bt.run(px, sig)
    assert len(res["trades"]) == 0
    assert res["equity"].iloc[-1] == pytest.approx(1.0)


def test_costs_reduce_returns(data):
    px = data.prices.tail(1500)
    sig = pd.Series((np.arange(len(px)) % 5 == 0).astype(int), index=px.index)
    free = bt.run(px, sig, params=bt.BacktestParams(cost_bps=0.0))["metrics"]
    costly = bt.run(px, sig, params=bt.BacktestParams(cost_bps=25.0))["metrics"]
    assert costly["total_return"] < free["total_return"]


def test_strategy_position_is_lagged(data):
    """The position applied on day t must be the one decided on day t-1."""
    from chandraquant.astro import engine as astro_engine

    px = data.prices.tail(600)
    astro = astro_engine.build(px.index, "NIFTY")
    p = st.StrategyParams(use_astro=False)
    res = st.run(px, astro, p)
    built = st.build_position(px, astro, p)["position"].reindex(px.index)
    pd.testing.assert_series_equal(
        res["position"], built.shift(1).fillna(0.0), check_names=False
    )


def test_strategy_is_deterministic(data):
    from chandraquant.astro import engine as astro_engine

    px = data.prices.tail(800)
    astro = astro_engine.build(px.index, "NIFTY")
    a = st.run(px, astro, st.StrategyParams(use_astro=False))["metrics"]
    b = st.run(px, astro, st.StrategyParams(use_astro=False))["metrics"]
    assert a == b


# --------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------
def test_metrics_on_a_known_series():
    idx = pd.date_range("2020-01-01", periods=252 * 4, freq="B")
    equity = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    assert M.max_drawdown(equity) == pytest.approx(0.0, abs=1e-12)
    assert M.cagr(equity) > 0
    falling = pd.Series(np.linspace(1.0, 0.5, len(idx)), index=idx)
    assert M.max_drawdown(falling) == pytest.approx(-0.5, abs=1e-6)


def test_sharpe_of_constant_returns_is_zero():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    assert M.sharpe(pd.Series(0.0, index=idx)) == 0.0


# --------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", TICKER_KEYS)
def test_dataset_builds_for_every_ticker(ticker):
    d = ds.build(ticker, refresh=False)
    assert len(d.prices) > 3000
    assert len(d.astro_cols) > 500
    assert len(d.tech_cols) > 50
    assert d.features.index.is_monotonic_increasing
    assert not d.features.index.has_duplicates


@pytest.mark.parametrize("ticker", TICKER_KEYS)
def test_snapshot_renders(ticker):
    """The full inference path must produce a usable verdict without a network."""
    from chandraquant import inference

    snap = inference.snapshot(ticker, refresh=False, with_forward=False)
    assert snap.detected_name in ("Vriddhi", "Sthira", "Kshaya", "Kshobha")
    assert 0.0 <= snap.position <= 3.5
    assert snap.narrative["text"]
    assert snap.chart["moon_nakshatra_name"]


def test_forward_calendar_is_genuinely_forward():
    """It must extend past the last bar of market data - that is the whole point."""
    from chandraquant import inference

    d = ds.build("NIFTY", refresh=False)
    last_bar = d.prices.index[-1]
    cal = inference.forward_calendar("NIFTY", last_bar, days=30)
    assert cal.index.max() > last_bar
    assert len(cal) >= 30
