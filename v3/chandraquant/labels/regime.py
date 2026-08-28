"""Four-state regime labels: Vriddhi, Sthira, Kshaya, Kshobha.

v1 predicted a binary "will the next H days be up". That throws away the distinction
that actually matters for position sizing: a quiet drift up and a violent whipsaw that
happens to end higher are not the same regime, and no trader should hold the same
position through both.

So the label space is four states, named in Sanskrit because the whole system speaks
that language:

  Vriddhi  growth     forward return clears +mu, forward volatility normal
  Sthira   stability  forward return inside +/-mu, low volatility - a range
  Kshaya   decay      forward return below -mu, forward volatility normal
  Kshobha  agitation  forward volatility in the top decile, regardless of direction

mu is volatility-scaled per ticker (k * trailing_vol * sqrt(H)), so the same threshold
means the same thing for placid CNXIT and violent BANKNIFTY.

HORIZON. H is deliberately chosen to match lunar transit lengths - the Moon spends
~2.3 days in a rashi and ~2.5 days in a nakshatra, and a tithi lasts ~0.98 days. A
label horizon of 3-7 trading days is therefore the window over which a lunar state is
actually constant. That is both the honest choice and, conveniently, the one that
gives the astro block its best shot.

The Kshobha threshold is a ROLLING quantile of forward volatility, never full-sample,
so the label definition at time t depends only on data available by t (plus the H-day
forward window that defines the label itself, which is unavoidable and is handled by
purging in the CV splitter). A rolling window also matters substantively: with an
expanding one the 2008 crisis dominates the quantile forever after and starves Kshobha
to ~4% of rows, when the intended rate is ~12%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VRIDDHI, STHIRA, KSHAYA, KSHOBHA = 0, 1, 2, 3

REGIME_NAMES = {
    VRIDDHI: "Vriddhi",
    STHIRA: "Sthira",
    KSHAYA: "Kshaya",
    KSHOBHA: "Kshobha",
}

REGIME_DISPLAY = {
    VRIDDHI: "VRIDDHI",
    STHIRA: "STHIRA",
    KSHAYA: "KSHAYA",
    KSHOBHA: "KSHOBHA",
}

REGIME_MEANING = {
    VRIDDHI: "expansion - growth regime, trend favours the long side",
    STHIRA: "stability - consolidation, the market is coiling rather than trending",
    KSHAYA: "decay - decline regime, distribution is in control",
    KSHOBHA: "agitation - turbulence, direction is unreliable and risk is elevated",
}

REGIME_COLOUR = {
    VRIDDHI: "#22c55e",
    STHIRA: "#f59e0b",
    KSHAYA: "#ef4444",
    KSHOBHA: "#a855f7",
}


@dataclass
class LabelConfig:
    horizon: int = 5            # trading days - tuned against lunar transit length
    mu_k: float = 0.45          # threshold as a multiple of trailing vol * sqrt(H)
    kshobha_quantile: float = 0.88
    vol_window: int = 500       # ~2 years - Kshobha is relative to the recent regime
    min_periods: int = 120


def compute(prices: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame:
    """Forward-looking regime labels. The last `horizon` rows are NaN by construction."""
    cfg = cfg or LabelConfig()
    close = prices["Close"]
    ret1 = close.pct_change()
    H = cfg.horizon

    fwd_ret = close.shift(-H) / close - 1.0
    # Realised volatility over the FORWARD window, annualised. Reversing the series
    # before rolling makes the window look forward; the trailing shift(-1) moves it
    # off the current bar so it spans ret1[t+1 .. t+H].
    fwd_vol = (
        ret1.iloc[::-1].rolling(H, min_periods=max(2, H // 2)).std().iloc[::-1].shift(-1)
        * np.sqrt(252)
    )

    trailing_vol = ret1.rolling(20, min_periods=10).std()
    mu = cfg.mu_k * trailing_vol * np.sqrt(H)

    # Kshobha means "turbulent *relative to the current regime*", not "turbulent by the
    # standards of 2008". An expanding quantile is dominated by the GFC and starves the
    # class to ~4% of rows; a rolling window keeps it at the intended rate and is still
    # strictly causal.
    vol_thresh = fwd_vol.rolling(cfg.vol_window, min_periods=cfg.min_periods).quantile(
        cfg.kshobha_quantile
    )
    vol_thresh = vol_thresh.fillna(
        fwd_vol.expanding(min_periods=60).quantile(cfg.kshobha_quantile)
    )

    state = pd.Series(np.nan, index=close.index, dtype=float)
    valid = fwd_ret.notna() & mu.notna()

    is_kshobha = valid & (fwd_vol > vol_thresh)
    is_vriddhi = valid & ~is_kshobha & (fwd_ret > mu)
    is_kshaya = valid & ~is_kshobha & (fwd_ret < -mu)
    is_sthira = valid & ~is_kshobha & ~is_vriddhi & ~is_kshaya

    state[is_vriddhi] = VRIDDHI
    state[is_sthira] = STHIRA
    state[is_kshaya] = KSHAYA
    state[is_kshobha] = KSHOBHA

    out = pd.DataFrame(
        {
            "fwd_return": fwd_ret,
            "fwd_vol": fwd_vol,
            "mu": mu,
            "regime": state,
            "y_vriddhi": (state == VRIDDHI).astype(float).where(valid),
            "y_kshobha": (state == KSHOBHA).astype(float).where(valid),
            "y_up": (fwd_ret > 0).astype(float).where(valid),
        },
        index=close.index,
    )
    out["regime_name"] = out["regime"].map(
        lambda v: REGIME_NAMES.get(int(v)) if pd.notna(v) else None
    )
    return out


def base_rates(labels: pd.DataFrame) -> dict[str, float]:
    valid = labels["regime"].notna()
    counts = labels.loc[valid, "regime"].value_counts(normalize=True)
    return {REGIME_NAMES[int(k)]: float(v) for k, v in counts.items()}
