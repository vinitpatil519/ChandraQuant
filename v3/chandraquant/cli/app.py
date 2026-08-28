"""`chandraquant` - the terminal entry point.

Flow: splash -> ticker picker -> dashboard -> optional interactive loop.
Flags let every step be skipped for scripting and screenshots.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from rich.console import Console
from rich.text import Text

from ..config import TICKER_KEYS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chandraquant",
        description="ChandraQuant - Kala-Chakra Alpha Engine. Jyotisha-driven regime "
                    "detection for NIFTY 50, BANKNIFTY and CNX IT.",
    )
    p.add_argument("--ticker", "-t", help="skip the picker (NIFTY / BANKNIFTY / CNXIT)")
    p.add_argument("--date", "-d", help="historical replay, e.g. 2020-03-23")
    p.add_argument("--web", action="store_true", help="open the browser dashboard")
    p.add_argument("--honest", action="store_true", help="show unconditional metrics and caveats")
    p.add_argument("--no-refresh", action="store_true", help="never hit the network")
    p.add_argument("--no-splash", action="store_true", help="skip the animation")
    p.add_argument("--compact", action="store_true", help="shorter dashboard")
    p.add_argument("--retrain", action="store_true", help="force model retraining")
    p.add_argument("--once", action="store_true", help="render once and exit")
    p.add_argument("--version", action="version", version="ChandraQuant 3.0.0")
    return p


def _honest_panel(console: Console) -> None:
    from rich.panel import Panel

    text = Text()
    text.append("METHODOLOGY — what this system does and does not know\n\n", style="bold #f59e0b")
    text.append(
        "The astro layer is a genuine, high-precision Jyotisha engine: 831 features from "
        "NASA JPL DE440s, natal charts per index, Vimshottari dasha, Ashtakavarga, "
        "Shadbala. Eclipse detection matches the NASA catalogue exactly.\n\n",
        style="white",
    )
    text.append("What it drives: ", style=f"bold #22d3ee")
    text.append(
        "regime narrative, the forward calendar, the composite indices and the Pine "
        "visuals. Astro features are deterministic functions of date, so the forward "
        "calendar is a real forecast, not a fit.\n\n",
        style="white",
    )
    text.append("What it does not drive: ", style="bold #ef4444")
    text.append(
        "position sizing. Measured out-of-sample, celestial state does not reliably "
        "predict Indian index returns. The astro block reaches AUC ~0.50 and its "
        "conditional edges are not stationary across walk-forward folds. Hand-assigned "
        "classical biases backtested as actively harmful. Position sizing is therefore "
        "trend + volatility targeting.\n\n",
        style="white",
    )
    text.append("The honest headline: ", style="bold #22c55e")
    text.append(
        "Calmar 0.38-0.39 against a 0.24-0.25 benchmark across all three indices, with "
        "max drawdown roughly halved. That is a real risk-adjusted improvement, and it "
        "comes from volatility targeting rather than from the sky.\n",
        style="white",
    )
    console.print(Panel(text, border_style="#f59e0b", padding=(1, 2),
                        title="[bold]--honest[/bold]"))


def _open_web(console: Console, ticker: str) -> None:
    try:
        from ..web.server import serve

        console.print("[dim]starting local dashboard...[/dim]")
        serve(ticker, open_browser=True)
    except Exception as exc:
        console.print(f"[red]web dashboard unavailable:[/red] {exc}")


def run_dashboard(console: Console, args, ticker: str) -> str | None:
    """Render the dashboard; return the next action."""
    from .. import inference
    from ..tui import dashboard

    with console.status(f"[dim]computing {ticker}...", spinner="dots"):
        snap = inference.snapshot(
            ticker,
            when=args.date,
            refresh=not args.no_refresh,
            retrain=args.retrain,
        )
    console.print()
    dashboard.render(snap, console, compact=args.compact)
    if args.honest:
        _honest_panel(console)

    if args.once or not sys.stdin.isatty():
        return None

    try:
        choice = console.input("\n  [dim]command >[/dim] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice in ("q", "quit", "exit", ""):
        return None
    if choice in ("r", "refresh"):
        return ticker
    if choice in ("b", "back"):
        return "__picker__"
    if choice in ("w", "web"):
        _open_web(console, ticker)
        return ticker
    if choice in ("h", "honest"):
        args.honest = not args.honest
        return ticker
    return ticker


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    from .picker import choose
    from .splash import show as show_splash

    if not args.ticker:
        show_splash(console, skip=args.no_splash)

    if args.web and args.ticker:
        _open_web(console, args.ticker)
        return 0

    ticker = args.ticker
    while True:
        if ticker is None:
            ticker = choose(console, refresh=not args.no_refresh)
            if ticker is None:
                console.print("\n  [dim]shubham astu — may it be auspicious.[/dim]\n")
                return 0
        try:
            nxt = run_dashboard(console, args, ticker)
        except KeyboardInterrupt:
            console.print("\n  [dim]interrupted[/dim]")
            return 130
        except Exception as exc:
            console.print(f"\n[red]error:[/red] {exc}")
            if "--debug" in (argv or sys.argv):
                raise
            return 1
        if nxt is None:
            console.print("\n  [dim]shubham astu — may it be auspicious.[/dim]\n")
            return 0
        ticker = None if nxt == "__picker__" else nxt


if __name__ == "__main__":
    raise SystemExit(main())
