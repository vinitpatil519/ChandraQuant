"""Run the strategy on every ticker and write artifacts/metrics.json.

metrics.json is the single source of truth for every number displayed anywhere - the
TUI backtest card, the web dashboard, the README table and the regenerated report all
read from it. Nothing downstream is allowed to hard-code a performance figure.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from chandraquant.astro import engine as astro_engine
from chandraquant.backtest import engine as bt_engine
from chandraquant.backtest import strategy as st
from chandraquant.config import ARTIFACT_DIR, TICKER_KEYS
from chandraquant.features import dataset as ds
from chandraquant.inference import _strategy_params


def crisis_slices(equity: pd.Series, windows: list[dict]) -> dict:
    out = {}
    for w in windows:
        seg = equity.loc[str(w["start"]): str(w["end"])]
        if len(seg) > 2:
            out[w["key"]] = {
                "label": w["label"],
                "return": float(seg.iloc[-1] / seg.iloc[0] - 1.0),
                "max_drawdown": float((seg / seg.cummax() - 1.0).min()),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    args = ap.parse_args()

    from chandraquant.config import tickers_cfg

    windows = tickers_cfg()["crisis_windows"]
    report: dict = {}

    for key in TICKER_KEYS:
        d = ds.build(key, refresh=args.refresh)
        prices = d.prices
        astro = astro_engine.build(prices.index, key)
        params = _strategy_params(key)
        idx = prices.index[prices.index >= args.start]

        res = st.run(prices, astro, params, index=idx)
        bh = bt_engine.buy_and_hold(prices.loc[idx])

        strat_m = res["metrics"]
        bench_m = bh["metrics"]

        report[key] = {
            "period": f"{idx[0].date()} to {idx[-1].date()}",
            "rows": int(len(idx)),
            "params": {
                "target_vol": params.target_vol,
                "leverage_cap": params.leverage_cap,
                "trend_span": params.trend_span,
                "vol_window": params.vol_window,
                "smooth_days": params.smooth_days,
                "cost_bps": params.cost_bps,
            },
            "strategy": {
                k: float(v)
                for k, v in strat_m.items()
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)
            },
            "benchmark": {
                "cagr": float(bench_m["cagr"]),
                "sharpe": float(bench_m["sharpe"]),
                "max_drawdown": float(bench_m["max_drawdown"]),
                "calmar": float(bench_m["calmar"]),
                "total_return": float(bench_m["total_return"]),
            },
            "crisis": crisis_slices(res["equity"], windows),
            "equity_curve": {
                "dates": [str(x.date()) for x in res["equity"].index[::5]],
                "strategy": [round(float(v), 5) for v in res["equity"].iloc[::5]],
                "benchmark": [
                    round(float(v), 5)
                    for v in (bh["equity"] / bh["equity"].iloc[0]).iloc[::5]
                ],
            },
        }

        s, b = report[key]["strategy"], report[key]["benchmark"]
        print(
            f"{key:10s} CAGR {s['cagr']*100:6.2f}% (B&H {b['cagr']*100:5.2f}%)  "
            f"Sharpe {s['sharpe']:.2f} ({b['sharpe']:.2f})  "
            f"MaxDD {s['max_drawdown']*100:6.1f}% ({b['max_drawdown']*100:.1f}%)  "
            f"Calmar {s['calmar']:.2f} ({b['calmar']:.2f})"
        )

    out = ARTIFACT_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
