"""The Kala-Chakra splash - an animated wheel of time on launch."""

from __future__ import annotations

import math
import time

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text

WORDMARK = [
    "  ____ _                     _            ___                   _   ",
    " / ___| |__   __ _ _ __   __| |_ __ __ _ / _ \\ _   _  __ _ _ __ | |_ ",
    "| |   | '_ \\ / _` | '_ \\ / _` | '__/ _` | | | | | | |/ _` | '_ \\| __|",
    "| |___| | | | (_| | | | | (_| | | | (_| | |_| | |_| | (_| | | | | |_ ",
    " \\____|_| |_|\\__,_|_| |_|\\__,_|_|  \\__,_|\\__\\_\\\\__,_|\\__,_|_| |_|\\__|",
]

# The twelve rashis around the wheel, and the nine grahas that ride it.
SPOKES = "|/-\\"
GRAHA_GLYPHS = ["Su", "Ch", "Ma", "Bu", "Gu", "Sk", "Sa", "Ra", "Ke"]

GRADIENT = ["#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd", "#67e8f9", "#22d3ee", "#06b6d4"]


def _wheel(frame: int, radius: int = 7) -> Text:
    """A rotating chakra rendered as an ASCII ring."""
    size = radius * 2 + 1
    grid = [[" " for _ in range(size * 2)] for _ in range(size)]

    for i in range(24):
        angle = 2 * math.pi * i / 24 + frame * 0.13
        x = int(round(radius + radius * math.cos(angle)))
        y = int(round(radius + radius * math.sin(angle) * 0.55))
        if 0 <= y < size and 0 <= 2 * x < size * 2:
            grid[y][2 * x] = "*" if i % 2 == 0 else "."

    # Spokes.
    for i in range(8):
        angle = 2 * math.pi * i / 8 + frame * 0.13
        for rr in range(1, radius):
            x = int(round(radius + rr * math.cos(angle)))
            y = int(round(radius + rr * math.sin(angle) * 0.55))
            if 0 <= y < size and 0 <= 2 * x < size * 2:
                grid[y][2 * x] = SPOKES[i % 4]

    # Grahas orbiting at different rates, fastest first - Chandra outruns Shani.
    speeds = [0.9, 3.1, 0.5, 1.3, 0.08, 1.6, 0.03, -0.05, -0.05]
    for gi, (glyph, sp) in enumerate(zip(GRAHA_GLYPHS, speeds)):
        angle = frame * 0.09 * (0.4 + sp) + gi * 0.7
        rr = radius - 1.5
        x = int(round(radius + rr * math.cos(angle)))
        y = int(round(radius + rr * math.sin(angle) * 0.55))
        if 0 <= y < size and 0 <= 2 * x < size * 2 - 1:
            grid[y][2 * x] = glyph[0]
            grid[y][2 * x + 1] = glyph[1]

    text = Text()
    for r, rowchars in enumerate(grid):
        colour = GRADIENT[(r + frame // 3) % len(GRADIENT)]
        text.append("".join(rowchars).rstrip() + "\n", style=colour)
    return text


def show(console: Console | None = None, duration: float = 1.4, skip: bool = False) -> None:
    """Play the splash. Any exception (or a dumb terminal) degrades to a static banner."""
    console = console or Console()
    if skip or not console.is_terminal:
        _static(console)
        return
    try:
        frames = max(1, int(duration / 0.06))
        with Live(console=console, refresh_per_second=24, transient=True) as live:
            for f in range(frames):
                body = Text()
                body.append_text(_wheel(f))
                body.append("\n")
                for i, line in enumerate(WORDMARK):
                    body.append(line + "\n", style=f"bold {GRADIENT[(i + f // 4) % len(GRADIENT)]}")
                body.append("\n      Kala-Chakra Alpha Engine", style="dim italic")
                body.append("   ·   where Jyotisha meets quantitative alpha\n", style="dim")
                live.update(Align.center(body))
                time.sleep(0.06)
    except Exception:
        pass
    _static(console)


def _static(console: Console) -> None:
    text = Text()
    for i, line in enumerate(WORDMARK):
        text.append(line + "\n", style=f"bold {GRADIENT[i % len(GRADIENT)]}")
    text.append("\n  Kala-Chakra Alpha Engine", style="bold #22d3ee")
    text.append("  ·  where Jyotisha meets quantitative alpha\n", style="dim")
    console.print(Align.center(text))
