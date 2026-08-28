"""The ChandraQuant strategy: vol-targeted trend, filtered and sized by the astro layer.

Design rationale, arrived at empirically rather than assumed.

A long-only daily system on an index that itself compounds at ~9% is a hard thing to
beat on raw return. Testing established that:
  - the 5-day regime classifier is weak (OOF AUC ~0.52), so using it as a binary entry
    trigger destroys more in transaction costs and missed upside than it adds
  - hand-assigned astrological biases were actively harmful (NIFTY CAGR 9.2% -> 0.1%)
  - what *does* work is the combination that professional trend followers use:
    a trend filter to define direction, volatility targeting to normalise risk, and a
    risk overlay that cuts exposure when conditions are hostile

So the astro layer's job here is **risk management, not return prediction**. That is
also the honest reading of what the classical material claims: muhurta does not tell
you what to buy, it tells you when not to act. Vishti karana, eclipse windows and
Chandrashtama are abstention rules, and abstention rules are evaluated by what they do
to drawdown, not to CAGR.

Position sizing:

    size = trend * vol_scalar * astro_scalar
    vol_scalar   = clip(target_vol / realised_vol, 0, cap)
    astro_scalar = 0                      if a hard gate is shut
                 = 1 - fear * BHY_norm    otherwise, optionally lifted by the model score

Daily rebalanced, turnover charged both ways.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class StrategyParams:
    # --- trend definition -------------------------------------------------------
    # Continuous, not binary. A step function throws away the difference between a
    # market barely above its mean and one extended far past it, and it churns on every
    # boundary crossing. Distance from the EMA in volatility units keeps both.
    trend_span: int = 100
    trend_k: float = 1.4            # slope of the trend response
    trend_floor: float = 0.5        # exposure when trend strength is zero
    trend_cap: float = 1.6

    # --- volatility targeting ---------------------------------------------------
    target_vol: float = 0.15
    vol_window: int = 20
    leverage_cap: float = 2.5
    vol_floor: float = 0.05

    # --- astro overlay ----------------------------------------------------------
    use_astro: bool = True
    use_hard_gates: bool = True
    fear_scale: float = 0.55        # how hard BHY cuts exposure
    edge_tilt: float = 0.0          # optional tilt from the learned astro edge
    score_tilt: float = 0.0         # optional tilt from the model probability

    # --- costs and smoothing ------------------------------------------------------
    cost_bps: float = 5.0
    smooth_days: int = 3            # smooth the position to suppress churn
    max_position: float = 3.0


def trend_signal(close: pd.Series, p: StrategyParams) -> pd.Series:
    """Continuous trend strength: distance above the EMA, measured in vol units.

    Returns roughly `trend_floor` in a flat market, rising toward `trend_cap` as the
    market extends above its own trend. This is the exact form scripts/optimize.py
    tunes, so tuned parameters and live behaviour cannot drift apart.
    """
    ema = close.ewm(span=p.trend_span, adjust=False).mean()
    scale = close.pct_change().rolling(60, min_periods=20).std() * np.sqrt(60)
    z = ((close / ema - 1.0) / scale.replace(0.0, np.nan)).clip(-3, 3)
    return (p.trend_floor + p.trend_k * z.clip(0, 3) / 3.0).clip(0, p.trend_cap)


def vol_scalar(close: pd.Series, p: StrategyParams) -> pd.Series:
    rv = close.pct_change().rolling(p.vol_window, min_periods=p.vol_window // 2).std() * np.sqrt(252)
    rv = rv.clip(lower=p.vol_floor)
    return (p.target_vol / rv).clip(0.0, p.leverage_cap)


def astro_scalar(
    astro: pd.DataFrame,
    p: StrategyParams,
    score: pd.Series | None = None,
    edge: pd.Series | None = None,
) -> pd.Series:
    idx = astro.index
    if not p.use_astro:
        return pd.Series(1.0, index=idx)

    bhy = astro["BHY"].reindex(idx).fillna(0.5) if "BHY" in astro else pd.Series(0.5, index=idx)
    # BHY is already 0..1. Centre it so an average day is unscaled.
    fear = (bhy - bhy.expanding(min_periods=60).median().fillna(0.5)).clip(-0.5, 0.5)
    scalar = (1.0 - p.fear_scale * fear * 2.0).clip(0.0, 1.5)

    if p.use_hard_gates:
        for col in ("GATE_vishti", "GATE_eclipse", "GATE_chandrashtama"):
            if col in astro:
                scalar = scalar.where(astro[col].reindex(idx).fillna(0) == 0, 0.0)

    if p.edge_tilt and edge is not None:
        e = edge.reindex(idx).fillna(0.0)
        scalar = scalar * (1.0 + p.edge_tilt * np.tanh(e))
    if p.score_tilt and score is not None:
        s = score.reindex(idx).fillna(0.5)
        scalar = scalar * (1.0 + p.score_tilt * (s - 0.5) * 2.0)

    return scalar.clip(0.0, 1.5)


def build_position(
    prices: pd.DataFrame,
    astro: pd.DataFrame,
    p: StrategyParams,
    score: pd.Series | None = None,
    edge: pd.Series | None = None,
) -> pd.DataFrame:
    """Target position per day, and its components (for the dashboard breakdown)."""
    close = prices["Close"]
    trend = trend_signal(close, p).astype(float).fillna(p.trend_floor)
    vol = vol_scalar(close, p)
    astro_s = astro_scalar(astro, p, score, edge)

    raw = trend * vol * astro_s
    if p.smooth_days > 1:
        raw = raw.rolling(p.smooth_days, min_periods=1).mean()
    pos = raw.clip(0.0, p.max_position).fillna(0.0)

    return pd.DataFrame(
        {"position": pos, "trend": trend, "vol_scalar": vol, "astro_scalar": astro_s},
        index=close.index,
    )


def _trades_from_position(position: pd.Series, close: pd.Series, cost: float) -> pd.DataFrame:
    """Treat each contiguous run of non-zero exposure as one trade."""
    active = position > 1e-6
    if not active.any():
        return pd.DataFrame(columns=["entry_date", "exit_date", "bars", "return", "avg_size"])
    grp = (active != active.shift(1)).cumsum()
    rows = []
    for _, block in position[active].groupby(grp[active]):
        start, end = block.index[0], block.index[-1]
        i0 = close.index.get_loc(start)
        i1 = close.index.get_loc(end)
        if i1 <= i0:
            continue
        gross = close.iloc[i1] / close.iloc[i0] - 1.0
        avg_size = float(block.mean())
        rows.append(
            {
                "entry_date": start,
                "exit_date": end,
                "bars": i1 - i0,
                "return": gross * avg_size - 2 * cost * avg_size,
                "avg_size": avg_size,
            }
        )
    return pd.DataFrame(rows)


def run(
    prices: pd.DataFrame,
    astro: pd.DataFrame,
    p: StrategyParams | None = None,
    score: pd.Series | None = None,
    edge: pd.Series | None = None,
    index: pd.DatetimeIndex | None = None,
) -> dict:
    """Backtest the strategy. Positions are applied with a one-bar lag, always."""
    p = p or StrategyParams()
    idx = index if index is not None else prices.index
    px = prices.reindex(idx)
    comp = build_position(prices, astro, p, score, edge).reindex(idx)

    ret = px["Close"].pct_change()
    pos = comp["position"]
    # One-bar execution lag: today's position was decided on yesterday's close.
    applied = pos.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(applied.abs())
    cost = p.cost_bps / 10000.0

    strat_ret = (applied * ret).fillna(0.0) - turnover * cost
    equity = (1.0 + strat_ret).cumprod()

    trades = _trades_from_position(applied, px["Close"], cost)
    summary = M.summarise(equity, strat_ret, trades, applied, benchmark=px["Close"])
    summary["avg_leverage"] = float(applied.mean())
    summary["max_leverage"] = float(applied.max())
    summary["turnover_annual"] = float(turnover.mean() * 252)

    return {
        "equity": equity,
        "returns": strat_ret,
        "position": applied,
        "components": comp,
        "trades": trades,
        "metrics": summary,
    }
