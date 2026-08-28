"""Event-driven long-only backtest with next-bar execution.

Rules of the house, all enforced in code rather than promised in a docstring:
  - a signal computed on bar t is executed at the OPEN of bar t+1, never at t's close
  - stops and targets are evaluated against bar highs/lows, with the pessimistic
    assumption that if a bar touches both the stop and the target, the STOP fills first
  - costs are charged on both legs
  - the equity curve is daily and mark-to-market, so Sharpe is computed on real daily
    returns rather than on trade returns

Exits, in priority order: hard stop, trailing stop, profit target, astro gate close,
regime flip, maximum holding period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class BacktestParams:
    cost_bps: float = 5.0          # per side, in basis points
    atr_period: int = 14
    stop_atr: float = 2.2          # hard stop distance
    trail_atr: float = 2.6         # trailing stop distance once in profit
    target_atr: float = 0.0        # 0 disables the fixed profit target
    max_hold: int = 12             # bars
    min_hold: int = 1
    exit_on_gate_close: bool = True
    exit_on_signal_off: bool = False
    allow_reentry_same_bar: bool = False
    risk_per_trade: float = 1.0    # 1.0 = full notional; <1 scales position size


def _atr(prices: pd.DataFrame, period: int) -> pd.Series:
    prev_close = prices["Close"].shift(1)
    tr = pd.concat(
        [
            prices["High"] - prices["Low"],
            (prices["High"] - prev_close).abs(),
            (prices["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def run(
    prices: pd.DataFrame,
    signal: pd.Series,
    gate_open: pd.Series | None = None,
    params: BacktestParams | None = None,
) -> dict:
    """Run the strategy. `signal` is 1 to want exposure on the NEXT bar, else 0."""
    p = params or BacktestParams()
    df = prices.loc[signal.index].copy()
    atr = _atr(prices, p.atr_period).reindex(df.index)

    open_, high, low, close = (df["Open"].to_numpy(), df["High"].to_numpy(),
                               df["Low"].to_numpy(), df["Close"].to_numpy())
    sig = signal.reindex(df.index).fillna(0).to_numpy().astype(int)
    gate = (
        gate_open.reindex(df.index).fillna(1).to_numpy().astype(int)
        if gate_open is not None
        else np.ones(len(df), dtype=int)
    )
    atr_v = atr.to_numpy()
    n = len(df)
    cost = p.cost_bps / 10000.0

    position = np.zeros(n)
    equity = np.ones(n)
    trades = []

    in_trade = False
    entry_price = 0.0
    entry_i = -1
    stop = 0.0
    peak = 0.0
    cash = 1.0
    units = 0.0

    for i in range(n):
        # Mark to market before acting on this bar.
        if in_trade:
            equity[i] = cash + units * close[i]
            position[i] = p.risk_per_trade
        else:
            equity[i] = cash
            position[i] = 0.0

        if in_trade:
            bars_held = i - entry_i
            exit_price = None
            reason = None

            # 1. Stop, checked against the bar's low. Pessimistic ordering.
            if low[i] <= stop:
                exit_price = min(stop, open_[i])
                reason = "stop"
            # 2. Profit target.
            elif p.target_atr > 0 and high[i] >= entry_price + p.target_atr * atr_v[entry_i]:
                exit_price = max(entry_price + p.target_atr * atr_v[entry_i], open_[i])
                reason = "target"

            if exit_price is None and bars_held >= p.min_hold:
                # 3. Astro gate slammed shut.
                if p.exit_on_gate_close and gate[i] == 0:
                    exit_price, reason = close[i], "gate"
                # 4. Signal withdrawn.
                elif p.exit_on_signal_off and sig[i] == 0:
                    exit_price, reason = close[i], "signal_off"
                # 5. Time stop.
                elif bars_held >= p.max_hold:
                    exit_price, reason = close[i], "time"

            if exit_price is not None:
                proceeds = units * exit_price * (1.0 - cost)
                cash += proceeds
                gross = exit_price / entry_price - 1.0
                net = gross - 2 * cost
                trades.append(
                    {
                        "entry_date": df.index[entry_i],
                        "exit_date": df.index[i],
                        "entry": entry_price,
                        "exit": exit_price,
                        "bars": bars_held,
                        "return": net,
                        "reason": reason,
                    }
                )
                in_trade = False
                units = 0.0
                equity[i] = cash
                position[i] = 0.0
            else:
                # Trail the stop up behind the running peak.
                peak = max(peak, high[i])
                trail = peak - p.trail_atr * atr_v[entry_i]
                stop = max(stop, trail)

        # Entry decision for the NEXT bar, using only information available at bar i.
        if not in_trade and i + 1 < n and sig[i] == 1 and gate[i] == 1:
            entry_price = open_[i + 1]
            if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(atr_v[i]):
                units = (cash * p.risk_per_trade) / (entry_price * (1.0 + cost))
                cash -= units * entry_price * (1.0 + cost)
                in_trade = True
                entry_i = i + 1
                peak = entry_price
                stop = entry_price - p.stop_atr * atr_v[i]

    # Close any open position at the final close.
    if in_trade:
        cash += units * close[-1] * (1.0 - cost)
        trades.append(
            {
                "entry_date": df.index[entry_i],
                "exit_date": df.index[-1],
                "entry": entry_price,
                "exit": close[-1],
                "bars": n - 1 - entry_i,
                "return": close[-1] / entry_price - 1.0 - 2 * cost,
                "reason": "eod",
            }
        )
        equity[-1] = cash

    equity_s = pd.Series(equity, index=df.index).replace(0.0, np.nan).ffill().fillna(1.0)
    daily_ret = equity_s.pct_change()
    position_s = pd.Series(position, index=df.index)
    trades_df = pd.DataFrame(trades)

    summary = M.summarise(equity_s, daily_ret, trades_df, position_s, benchmark=df["Close"])
    return {
        "equity": equity_s,
        "returns": daily_ret,
        "position": position_s,
        "trades": trades_df,
        "metrics": summary,
    }


def buy_and_hold(prices: pd.DataFrame) -> dict:
    close = prices["Close"]
    equity = close / close.iloc[0]
    ret = equity.pct_change()
    trades = pd.DataFrame(
        [
            {
                "entry_date": close.index[0],
                "exit_date": close.index[-1],
                "entry": close.iloc[0],
                "exit": close.iloc[-1],
                "bars": len(close),
                "return": float(close.iloc[-1] / close.iloc[0] - 1.0),
                "reason": "hold",
            }
        ]
    )
    return {
        "equity": equity,
        "returns": ret,
        "position": pd.Series(1.0, index=close.index),
        "trades": trades,
        "metrics": M.summarise(equity, ret, trades, pd.Series(1.0, index=close.index)),
    }
