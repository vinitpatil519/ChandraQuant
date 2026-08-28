"""Purged walk-forward cross-validation with an embargo.

v1 used a single chronological cut at October 2017. That is better than shuffling, but
it wastes most of the data as a single test set and gives one noisy number with no
confidence interval.

Walk-forward instead: train on everything up to a point, test on the block that
follows, roll forward, repeat. Two corrections are essential and both are missing from
most published financial backtests:

  PURGE    A label at bar t looks H bars into the future. If t is the last bar of the
           training set, its label overlaps the first H bars of the test set - the
           model has literally seen the test period's returns. So the final H bars of
           every training block are dropped.
  EMBARGO  Features are built from rolling windows, so bars just after the training
           cut are correlated with bars just before it. An embargo of E bars after the
           training end is excluded from testing to break that adjacency.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    n_splits: int = 6
    horizon: int = 5          # label lookahead - determines the purge width
    embargo: int = 5
    min_train: int = 750      # ~3 years before the first fold is allowed
    expanding: bool = True    # expanding train window; False = rolling


def walk_forward(
    index: pd.DatetimeIndex, cfg: WalkForwardConfig | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_positions, test_positions) for each fold."""
    cfg = cfg or WalkForwardConfig()
    n = len(index)
    if n <= cfg.min_train + cfg.n_splits * 20:
        raise ValueError(f"Not enough rows ({n}) for {cfg.n_splits} walk-forward folds")

    usable = n - cfg.min_train
    fold = usable // cfg.n_splits

    for k in range(cfg.n_splits):
        test_start = cfg.min_train + k * fold
        test_end = test_start + fold if k < cfg.n_splits - 1 else n
        # Purge the label window, then the embargo, off the end of training.
        train_end = test_start - cfg.horizon - cfg.embargo
        if train_end <= 50:
            continue
        train_start = 0 if cfg.expanding else max(0, train_end - cfg.min_train)
        yield np.arange(train_start, train_end), np.arange(test_start, test_end)


def single_split(index: pd.DatetimeIndex, cut: str, horizon: int = 5, embargo: int = 5):
    """The v1-style single chronological cut, retained for comparability."""
    cut_ts = pd.Timestamp(cut)
    pos = int(index.searchsorted(cut_ts))
    train_end = max(0, pos - horizon - embargo)
    return np.arange(0, train_end), np.arange(pos, len(index))


def describe(index: pd.DatetimeIndex, cfg: WalkForwardConfig | None = None) -> pd.DataFrame:
    rows = []
    for i, (tr, te) in enumerate(walk_forward(index, cfg), start=1):
        rows.append(
            {
                "fold": i,
                "train_start": index[tr[0]].date(),
                "train_end": index[tr[-1]].date(),
                "train_n": len(tr),
                "test_start": index[te[0]].date(),
                "test_end": index[te[-1]].date(),
                "test_n": len(te),
            }
        )
    return pd.DataFrame(rows)
