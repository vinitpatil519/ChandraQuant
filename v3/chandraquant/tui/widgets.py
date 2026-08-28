"""Small rendering primitives - sparklines, gauges, meters, regime chips."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.text import Text

BLOCKS = "▁▂▃▄▅▆▇█"
METER_FULL = "█"
METER_EMPTY = "░"

REGIME_STYLE = {
    "Vriddhi": "bold #22c55e",
    "Sthira": "bold #f59e0b",
    "Kshaya": "bold #ef4444",
    "Kshobha": "bold #a855f7",
}


def sparkline(values, width: int = 60, style: str = "#22d3ee") -> Text:
    """Compact block sparkline of the last `width` points."""
    s = pd.Series(values).dropna()
    if s.empty:
        return Text("(no data)", style="dim")
    if len(s) > width:
        # Bucket rather than slice, so the whole window stays visible.
        idx = np.linspace(0, len(s) - 1, width).astype(int)
        s = s.iloc[idx]
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return Text(BLOCKS[0] * len(s), style=style)
    scaled = ((s - lo) / (hi - lo) * (len(BLOCKS) - 1)).round().astype(int)
    return Text("".join(BLOCKS[int(v)] for v in scaled), style=style)


def coloured_sparkline(values, colours: list[str], width: int = 60) -> Text:
    """Sparkline where each column carries its own colour (used for regime bands)."""
    s = pd.Series(values).dropna()
    if s.empty:
        return Text("(no data)", style="dim")
    c = pd.Series(colours, index=pd.Series(values).index).reindex(s.index).ffill().fillna("#666666")
    if len(s) > width:
        idx = np.linspace(0, len(s) - 1, width).astype(int)
        s, c = s.iloc[idx], c.iloc[idx]
    lo, hi = float(s.min()), float(s.max())
    out = Text()
    for v, colour in zip(s, c):
        level = 0 if hi - lo < 1e-12 else int(round((v - lo) / (hi - lo) * (len(BLOCKS) - 1)))
        out.append(BLOCKS[level], style=str(colour))
    return out


def gauge(value: float, width: int = 26, lo: float = 0.0, hi: float = 1.0,
          style: str = "#22d3ee", show_pct: bool = True) -> Text:
    """Horizontal filled meter."""
    if not np.isfinite(value):
        return Text("  n/a", style="dim")
    frac = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    filled = int(round(frac * width))
    t = Text()
    t.append(METER_FULL * filled, style=style)
    t.append(METER_EMPTY * (width - filled), style="dim")
    if show_pct:
        t.append(f"  {frac * 100:5.1f}%", style=style)
    return t


def bipolar_meter(value: float, width: int = 24, style_pos: str = "#22c55e",
                  style_neg: str = "#ef4444") -> Text:
    """Meter for a value in [-1, 1], filling out from the centre."""
    if not np.isfinite(value):
        return Text(" n/a", style="dim")
    v = float(np.clip(value, -1.0, 1.0))
    half = width // 2
    cells = int(round(abs(v) * half))
    t = Text()
    if v < 0:
        t.append(METER_EMPTY * (half - cells), style="dim")
        t.append(METER_FULL * cells, style=style_neg)
        t.append("│", style="dim")
        t.append(METER_EMPTY * half, style="dim")
    else:
        t.append(METER_EMPTY * half, style="dim")
        t.append("│", style="dim")
        t.append(METER_FULL * cells, style=style_pos)
        t.append(METER_EMPTY * (half - cells), style="dim")
    t.append(f" {v:+.2f}", style=style_pos if v >= 0 else style_neg)
    return t


def regime_chip(name: str, big: bool = False) -> Text:
    style = REGIME_STYLE.get(name, "bold white")
    label = name.upper() if big else name
    return Text(f" {label} ", style=f"{style} reverse")


def trend_arrow(change: float) -> Text:
    if change > 0.002:
        return Text("▲", style="#22c55e")
    if change < -0.002:
        return Text("▼", style="#ef4444")
    return Text("■", style="#f59e0b")


def pct(value: float, digits: int = 2, colour: bool = True) -> Text:
    if not np.isfinite(value):
        return Text("n/a", style="dim")
    style = "#22c55e" if value > 0 else ("#ef4444" if value < 0 else "dim")
    return Text(f"{value * 100:+.{digits}f}%", style=style if colour else "")
