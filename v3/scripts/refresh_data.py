"""Pull fresh OHLCV, warm the astro cache, and rewrite the offline snapshot.

The snapshot is what keeps `chandraquant` working on a plane, behind a corporate
proxy, or in the middle of a demo when Yahoo decides to rate-limit. Run this while
online; everything afterwards can run without a network.
"""

from __future__ import annotations

import argparse

import pandas as pd

from chandraquant.astro import engine as astro_engine
from chandraquant.config import TICKER_KEYS
from chandraquant.data import loaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every ticker (default)")
    ap.add_argument("--ticker", help="just one")
    ap.add_argument("--no-astro", action="store_true", help="skip warming the astro cache")
    args = ap.parse_args()

    keys = [args.ticker] if args.ticker else TICKER_KEYS

    for key in keys:
        df, status = loaders.load_ohlcv(key, refresh=True)
        path = loaders.write_snapshot(key, df)
        print(
            f"{key:10s} {status.source:9s} {len(df):5d} bars  "
            f"{df.index[0].date()} to {df.index[-1].date()}  -> {path}"
        )
        if not args.no_astro:
            # Warm the cache for the traded history plus a year of forward calendar.
            span = df.index.append(
                pd.date_range(df.index[-1], df.index[-1] + pd.Timedelta(days=400), freq="D")
            )
            mat = astro_engine.build(span.unique().sort_values(), key)
            print(f"{'':10s} astro cache {mat.shape[0]} rows x {mat.shape[1]} features")


if __name__ == "__main__":
    main()
