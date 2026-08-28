"""Technical features - the endogenous half of the hybrid matrix.

Carries forward the 31 indicators v1 used (so the two are comparable) and adds the
regime-descriptive terms it lacked: realised-volatility ratios, Donchian position,
ADX, gap statistics and drawdown state.

Every column is strictly causal. Anything computed from a rolling window uses only
bars at or before t, and nothing is centred or back-filled. `assert_causal` is called
by the pipeline to prove it rather than assume it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Technical feature matrix from daily OHLCV."""
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df else pd.Series(0.0, index=df.index)
    out: dict[str, pd.Series] = {}

    ret1 = close.pct_change()
    out["ret_1d"] = ret1
    out["logret_1d"] = np.log(close / close.shift(1))

    # --- Momentum across horizons -----------------------------------------------------
    for h in (2, 3, 5, 10, 20, 60, 120, 200):
        out[f"ret_{h}d"] = close.pct_change(h)
    for h in (5, 10, 20, 60):
        out[f"mom_rank_{h}"] = close.pct_change(h).rolling(252, min_periods=60).rank(pct=True)

    # --- Moving averages and trend alignment --------------------------------------------
    for w in (5, 10, 20, 50, 100, 200):
        ma = close.rolling(w, min_periods=w // 2).mean()
        out[f"ma{w}_ratio"] = close / ma - 1.0
    out["ma50_200_ratio"] = (
        close.rolling(50, min_periods=25).mean() / close.rolling(200, min_periods=100).mean() - 1.0
    )
    out["golden_cross"] = (out["ma50_200_ratio"] > 0).astype(int)
    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    out["ema20_50_ratio"] = ema20 / ema50 - 1.0
    out["ema_trend_up"] = ((close > ema20) & (ema20 > ema50)).astype(int)

    # --- MACD ------------------------------------------------------------------------------
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    hist = macd - signal
    out["macd"] = macd / close
    out["macd_signal"] = signal / close
    out["macd_hist"] = hist / close
    out["macd_hist_norm"] = hist / _atr(df, 14).replace(0.0, np.nan)
    out["macd_hist_slope"] = hist.diff() / close
    out["macd_cross_up"] = ((hist > 0) & (hist.shift(1) <= 0)).astype(int)

    # --- Oscillators -------------------------------------------------------------------------
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_7"] = _rsi(close, 7)
    out["rsi_14_slope"] = _rsi(close, 14).diff()
    low14 = low.rolling(14, min_periods=7).min()
    high14 = high.rolling(14, min_periods=7).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0.0, np.nan)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()
    tp = (high + low + close) / 3.0
    tp_ma = tp.rolling(20, min_periods=10).mean()
    md = (tp - tp_ma).abs().rolling(20, min_periods=10).mean()
    out["cci_20"] = (tp - tp_ma) / (0.015 * md.replace(0.0, np.nan))
    out["williams_r"] = -100 * (high14 - close) / (high14 - low14).replace(0.0, np.nan)

    # --- Volatility ------------------------------------------------------------------------------
    atr14 = _atr(df, 14)
    out["atr_pct"] = atr14 / close
    out["atr_ratio_50"] = atr14 / atr14.rolling(50, min_periods=25).mean()
    for w in (5, 10, 20, 60):
        out[f"realvol_{w}"] = ret1.rolling(w, min_periods=w // 2).std() * np.sqrt(252)
    out["vol_ratio_5_60"] = out["realvol_5"] / out["realvol_60"].replace(0.0, np.nan)
    out["vol_ratio_20_60"] = out["realvol_20"] / out["realvol_60"].replace(0.0, np.nan)
    out["vol_rank_252"] = out["realvol_20"].rolling(252, min_periods=60).rank(pct=True)

    # --- Bollinger ----------------------------------------------------------------------------------
    ma20 = close.rolling(20, min_periods=10).mean()
    sd20 = close.rolling(20, min_periods=10).std()
    out["bb_percent_b"] = (close - (ma20 - 2 * sd20)) / (4 * sd20).replace(0.0, np.nan)
    out["bb_bandwidth"] = (4 * sd20) / ma20.replace(0.0, np.nan)
    out["bb_squeeze"] = (
        out["bb_bandwidth"] < out["bb_bandwidth"].rolling(120, min_periods=60).quantile(0.2)
    ).astype(int)

    # --- Channel position and breakout ------------------------------------------------------------------
    for w in (20, 55):
        hh = high.rolling(w, min_periods=w // 2).max()
        ll = low.rolling(w, min_periods=w // 2).min()
        out[f"donchian_pos_{w}"] = (close - ll) / (hh - ll).replace(0.0, np.nan)
        out[f"breakout_{w}"] = (close >= hh.shift(1)).astype(int)
        out[f"breakdown_{w}"] = (close <= ll.shift(1)).astype(int)

    # --- Trend strength ----------------------------------------------------------------------------------
    out["adx_14"] = _adx(df, 14)
    out["adx_strong"] = (out["adx_14"] > 25).astype(int)

    # --- Drawdown state ------------------------------------------------------------------------------------
    running_max = close.cummax()
    out["drawdown"] = close / running_max - 1.0
    out["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1.0
    out["days_since_high"] = (
        close.rolling(252, min_periods=60).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)
    )

    # --- Gaps and candle shape ---------------------------------------------------------------------------------
    prev_close = close.shift(1)
    out["gap_pct"] = (df["Open"] - prev_close) / prev_close
    rng = (high - low).replace(0.0, np.nan)
    out["close_position"] = (close - low) / rng
    out["body_ratio"] = (close - df["Open"]).abs() / rng
    out["upper_wick"] = (high - np.maximum(close, df["Open"])) / rng
    out["lower_wick"] = (np.minimum(close, df["Open"]) - low) / rng

    # --- Volume -------------------------------------------------------------------------------------------------
    vol_ma = volume.rolling(20, min_periods=10).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0.0, np.nan)
    obv = (np.sign(ret1.fillna(0.0)) * volume).cumsum()
    out["obv_slope_20"] = obv.diff(20) / vol_ma.replace(0.0, np.nan) / 20.0

    # --- Streaks --------------------------------------------------------------------------------------------------
    up = (ret1 > 0).astype(int)
    grp = (up != up.shift(1)).cumsum()
    out["streak"] = up.groupby(grp).cumcount().add(1) * np.where(up == 1, 1, -1)

    frame = pd.DataFrame(out, index=df.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def assert_causal(df: pd.DataFrame, features: pd.DataFrame, sample: int = 40) -> None:
    """Prove no lookahead: recomputing on a truncated history must reproduce the tail.

    If any feature peeked at future bars, truncating the input would change its value
    at earlier timestamps. This checks that it does not.
    """
    rng = np.random.default_rng(0)
    n = len(df)
    if n < 400:
        return
    cut_points = rng.integers(300, n - 1, size=min(sample, 12))
    for cut in cut_points:
        truncated = compute(df.iloc[: int(cut)])
        common = [c for c in truncated.columns if c in features.columns]
        a = truncated[common].iloc[-1]
        b = features[common].iloc[int(cut) - 1]
        both = a.notna() & b.notna()
        if not both.any():
            continue
        diff = (a[both] - b[both]).abs()
        scale = b[both].abs().clip(lower=1e-6)
        bad = diff / scale > 1e-6
        if bad.any():
            raise AssertionError(
                f"Lookahead detected at cut {cut} in: {list(bad[bad].index)[:8]}"
            )
