"""Performance metrics. One definition each, used everywhere, no hand-typed numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r - rf / TRADING_DAYS
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float((r.mean() - rf / TRADING_DAYS) / downside.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    return float(cagr(equity) / dd) if dd > 1e-9 else 0.0


def profit_factor(trade_returns: pd.Series) -> float:
    wins = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    return float(wins / losses) if losses > 1e-12 else float("inf")


def expectancy(trade_returns: pd.Series) -> float:
    return float(trade_returns.mean()) if len(trade_returns) else 0.0


def win_rate(trade_returns: pd.Series) -> float:
    return float((trade_returns > 0).mean()) if len(trade_returns) else 0.0


def exposure(position: pd.Series) -> float:
    return float((position != 0).mean()) if len(position) else 0.0


def summarise(
    equity: pd.Series,
    daily_returns: pd.Series,
    trades: pd.DataFrame,
    position: pd.Series,
    benchmark: pd.Series | None = None,
) -> dict:
    """The full metric block written to artifacts/metrics.json."""
    tr = trades["return"] if "return" in trades else pd.Series(dtype=float)
    out = {
        "n_trades": int(len(trades)),
        "win_rate": win_rate(tr),
        "cagr": cagr(equity),
        "sharpe": sharpe(daily_returns),
        "sortino": sortino(daily_returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "profit_factor": profit_factor(tr),
        "expectancy": expectancy(tr),
        "avg_win": float(tr[tr > 0].mean()) if (tr > 0).any() else 0.0,
        "avg_loss": float(tr[tr < 0].mean()) if (tr < 0).any() else 0.0,
        "avg_bars_held": float(trades["bars"].mean()) if "bars" in trades and len(trades) else 0.0,
        "exposure": exposure(position),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "best_trade": float(tr.max()) if len(tr) else 0.0,
        "worst_trade": float(tr.min()) if len(tr) else 0.0,
    }
    if benchmark is not None and len(benchmark) > 1:
        bench_eq = benchmark / benchmark.iloc[0]
        bench_ret = benchmark.pct_change()
        out["benchmark_cagr"] = cagr(bench_eq)
        out["benchmark_sharpe"] = sharpe(bench_ret)
        out["benchmark_max_drawdown"] = max_drawdown(bench_eq)
        out["excess_cagr"] = out["cagr"] - out["benchmark_cagr"]
        # Risk-adjusted outperformance is the honest headline for a long-only system on
        # an index that itself trends up.
        out["cagr_over_dd_vs_benchmark"] = (
            out["calmar"] - (out["benchmark_cagr"] / abs(out["benchmark_max_drawdown"]))
            if out["benchmark_max_drawdown"] < -1e-9
            else 0.0
        )
    return out
