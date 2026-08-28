"""Sweep the core strategy per ticker and write the winning parameters to config.

The search is over the trend/volatility core only. Astro is a display layer and does
not size positions (see CLAUDE.md, 2026-08-28 decision), so it is deliberately absent
from this grid - putting it in would only let the optimiser fit noise.

Selection rule: maximise Calmar (CAGR / |max drawdown|) subject to beating buy-and-hold
on drawdown. Calmar rather than CAGR because a long-only system on an index that
compounds at ~9% can always buy more return with more leverage; what has to be earned
is the drawdown reduction.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from chandraquant.backtest import metrics as M
from chandraquant.backtest import strategy as st
from chandraquant.config import ARTIFACT_DIR, CONFIG_DIR, TICKER_KEYS
from chandraquant.features import dataset as ds

GRID = {
    "target_vol": [0.15, 0.20, 0.25, 0.30],
    "leverage_cap": [2.0, 2.5, 3.0, 3.5],
    "trend_span": [50, 100, 200],
    "vol_window": [15, 20, 40],
    "smooth_days": [1, 3, 5],
}

START = "2010-01-01"


def continuous_trend(close: pd.Series, span: int, k: float = 1.4) -> pd.Series:
    """Continuous trend strength: how far above its own EMA, in volatility units."""
    ema = close.ewm(span=span, adjust=False).mean()
    scale = close.pct_change().rolling(60, min_periods=20).std() * np.sqrt(60)
    z = ((close / ema - 1.0) / scale.replace(0.0, np.nan)).clip(-3, 3)
    return (0.5 + k * z.clip(0, 3) / 3.0).clip(0, 1.6)


def evaluate(close: pd.Series, params: dict, start: str = START, cost_bps: float = 5.0) -> dict:
    rv = (
        close.pct_change().rolling(params["vol_window"], min_periods=10).std() * np.sqrt(252)
    ).clip(lower=0.05)
    pos = continuous_trend(close, params["trend_span"]) * (
        params["target_vol"] / rv
    ).clip(0, params["leverage_cap"])
    if params["smooth_days"] > 1:
        pos = pos.rolling(params["smooth_days"], min_periods=1).mean()

    idx = close.index[close.index >= start]
    r = close.reindex(idx).pct_change()
    applied = pos.reindex(idx).fillna(0.0).clip(0, 3.5).shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(0.0)
    sr = (applied * r).fillna(0.0) - turnover * cost_bps / 10000.0
    eq = (1.0 + sr).cumprod()
    return {
        "cagr": M.cagr(eq),
        "sharpe": M.sharpe(sr),
        "max_drawdown": M.max_drawdown(eq),
        "calmar": M.calmar(eq),
        "avg_leverage": float(applied.mean()),
        "turnover_annual": float(turnover.mean() * 252),
    }


def benchmark(close: pd.Series, start: str = START) -> dict:
    idx = close.index[close.index >= start]
    c = close.reindex(idx)
    eq = c / c.iloc[0]
    r = c.pct_change()
    return {
        "cagr": M.cagr(eq),
        "sharpe": M.sharpe(r),
        "max_drawdown": M.max_drawdown(eq),
        "calmar": M.calmar(eq),
    }


def sweep(close: pd.Series, start: str = START) -> pd.DataFrame:
    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        rec = evaluate(close, params, start)
        rec.update(params)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    chosen: dict[str, dict] = {}
    report: dict[str, dict] = {}

    for key in TICKER_KEYS:
        d = ds.build(key, refresh=args.refresh)
        close = d.prices["Close"]
        bench = benchmark(close, args.start)
        table = sweep(close, args.start)

        # Must beat buy-and-hold on drawdown, then take the best Calmar.
        eligible = table[table["max_drawdown"] > bench["max_drawdown"]]
        pool = eligible if len(eligible) else table
        best = pool.sort_values("calmar", ascending=False).iloc[0]

        params = {k: (int(best[k]) if k != "target_vol" else float(best[k])) for k in GRID}
        params["target_vol"] = float(best["target_vol"])
        params["leverage_cap"] = float(best["leverage_cap"])
        chosen[key] = params
        report[key] = {
            "benchmark": bench,
            "strategy": {
                m: float(best[m])
                for m in ("cagr", "sharpe", "max_drawdown", "calmar", "avg_leverage", "turnover_annual")
            },
            "params": params,
        }

        print(
            f"{key:10s} CAGR {best['cagr']*100:6.2f}% (B&H {bench['cagr']*100:5.2f}%)  "
            f"Sharpe {best['sharpe']:.2f} ({bench['sharpe']:.2f})  "
            f"MaxDD {best['max_drawdown']*100:6.1f}% ({bench['max_drawdown']*100:.1f}%)  "
            f"Calmar {best['calmar']:.2f} ({bench['calmar']:.2f})"
        )
        print(f"           params {params}")

    out_cfg = CONFIG_DIR / "strategy.yaml"
    with out_cfg.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"start": args.start, "cost_bps": 5.0, "tickers": chosen}, fh, sort_keys=False
        )
    (ARTIFACT_DIR / "optimize_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nwrote {out_cfg}")


if __name__ == "__main__":
    main()
