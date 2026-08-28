"""The ticker picker - arrow-key selection between the three indices.

Uses questionary when the terminal supports it, and falls back to a numbered prompt
otherwise (Windows terminals without full ANSI, piped stdin, CI). Each row carries a
live one-line status so the choice is informed rather than blind.
"""

from __future__ import annotations

import sys

from rich.console import Console

from ..config import TICKER_KEYS
from ..labels.regime import REGIME_COLOUR

ARROW = "❯"


def _row(status: dict) -> str:
    chg = status["change_1d"] * 100
    return (
        f"{status['display']:<11} {status['yahoo']:<10} "
        f"{status['regime_name']:<8} "
        f"{status['close']:>10,.1f} {chg:>+6.2f}%   "
        f"{status['nakshatra']}"
    )


def gather_statuses(console: Console, refresh: bool = False) -> list[dict]:
    from .. import inference

    out = []
    with console.status("[dim]casting the chart...", spinner="dots"):
        for key in TICKER_KEYS:
            try:
                out.append(inference.quick_status(key, refresh=refresh))
            except Exception as exc:  # a broken ticker must not kill the picker
                out.append(
                    {
                        "key": key,
                        "display": key,
                        "yahoo": "?",
                        "regime_name": "unknown",
                        "close": float("nan"),
                        "change_1d": 0.0,
                        "nakshatra": f"({exc.__class__.__name__})",
                        "colour": "#888888",
                    }
                )
    return out


def choose(console: Console, refresh: bool = False) -> str | None:
    """Return the chosen ticker key, or None if the user aborted."""
    statuses = gather_statuses(console, refresh=refresh)

    console.print()
    console.print("  [bold]Select an index[/bold]  [dim](arrow keys, enter to confirm)[/dim]\n")

    try:
        import questionary
        from questionary import Choice, Style

        style = Style(
            [
                ("qmark", "fg:#22d3ee bold"),
                ("pointer", "fg:#22d3ee bold"),
                ("highlighted", "fg:#22d3ee bold"),
                ("selected", "fg:#a78bfa bold"),
                ("answer", "fg:#a78bfa bold"),
            ]
        )
        choices = [Choice(title=_row(s), value=s["key"]) for s in statuses]
        choices.append(Choice(title="quit", value=None))
        answer = questionary.select(
            "", choices=choices, style=style, qmark="", instruction=" "
        ).ask()
        return answer
    except (ImportError, OSError, EOFError):
        return _fallback(console, statuses)


def _fallback(console: Console, statuses: list[dict]) -> str | None:
    for i, s in enumerate(statuses, start=1):
        colour = s.get("colour", "#888888")
        console.print(f"  [{colour}]{i}.[/{colour}] {_row(s)}")
    console.print("  [dim]q. quit[/dim]\n")
    try:
        raw = input("  choice [1]: ").strip().lower() or "1"
    except (EOFError, KeyboardInterrupt):
        return None
    if raw in ("q", "quit", "exit"):
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(statuses):
            return statuses[idx]["key"]
    except ValueError:
        pass
    console.print("  [red]not a valid choice[/red]")
    return None
