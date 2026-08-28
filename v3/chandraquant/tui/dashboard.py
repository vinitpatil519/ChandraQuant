"""The ChandraQuant dashboard - the screen the whole project exists to produce."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..astro.ephemeris import NAVAGRAHA
from ..config import graha_cfg
from ..explain import lexicon
from ..explain.narrative import plain
from ..inference import Snapshot
from ..labels.regime import REGIME_COLOUR
from . import widgets as W

PANEL_BORDER = "#334155"
ACCENT = "#22d3ee"
MUTED = "#94a3b8"


def _kv(rows: list[tuple[str, Text | str]], key_style: str = MUTED) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=key_style, justify="left", no_wrap=True)
    t.add_column(justify="left")
    for k, v in rows:
        t.add_row(k, v)
    return t


# --------------------------------------------------------------------------------
def header(s: Snapshot) -> Panel:
    left = Table.grid(padding=(0, 2))
    left.add_column()
    title = Text()
    title.append(f"{s.display}", style=f"bold {ACCENT}")
    title.append(f"   {s.close:,.2f}  ", style="bold white")
    title.append_text(W.trend_arrow(s.change_1d))
    title.append(" ")
    title.append_text(W.pct(s.change_1d))
    title.append(f"    20d ", style=MUTED)
    title.append_text(W.pct(s.change_20d))
    left.add_row(title)

    sub = Text()
    sub.append(f"{s.date.strftime('%A, %d %B %Y')}", style=MUTED)
    sub.append("   ·   ", style="dim")
    sub.append(s.status.badge(), style="#22c55e" if s.status.is_live else MUTED)
    sub.append("   ·   ", style="dim")
    sub.append(f"natal {s.chart['natal_local'].strftime('%Y-%m-%d %H:%M IST')}", style="dim")
    left.add_row(sub)
    return Padding(left, (0, 1))


def verdict(s: Snapshot) -> Panel:
    colour = REGIME_COLOUR[s.detected_regime]
    body = Table.grid(padding=(0, 1))
    body.add_column()

    big = Text()
    big.append_text(W.regime_chip(s.detected_name, big=True))
    body.add_row(big)
    body.add_row(Text(s.detected_meaning, style=MUTED))
    body.add_row(Text(""))

    prob = s.probability if np.isfinite(s.probability) else 0.5
    body.add_row(Text("P(Vriddhi) next 5 sessions", style=MUTED))
    body.add_row(W.gauge(prob, width=30, style=colour))
    body.add_row(Text(""))

    split = Table.grid(padding=(0, 2))
    split.add_column(style=MUTED)
    split.add_column()
    split.add_row("technical block", W.gauge(s.tech_probability, width=14, style="#38bdf8", show_pct=True))
    split.add_row("astro block", W.gauge(s.astro_probability, width=14, style="#a78bfa", show_pct=True))
    split.add_row("astro weight", W.gauge(s.astro_weight, width=14, style="#f59e0b", show_pct=True))
    body.add_row(split)
    body.add_row(Text(""))

    pos = Text()
    pos.append("target exposure  ", style=MUTED)
    pos.append(f"{s.position:.2f}x", style=f"bold {colour}")
    pos.append(f"   (trend {s.trend:.2f} × vol {s.vol_scalar:.2f})", style="dim")
    body.add_row(pos)

    return Panel(body, title="[bold]VERDICT[/bold]", border_style=colour, padding=(1, 2))


def panchanga(s: Snapshot) -> Panel:
    r = s.astro_row
    g = lambda k, d="": r[k] if k in r and pd.notna(r[k]) else d
    rows = [
        ("Tithi", Text(str(g("tithi_name")), style="white")),
        ("Nakshatra", Text(f"{g('nakshatra_name')}  ·  pada {int(g('pada', 1))}", style="white")),
        ("", Text(str(g("nakshatra_archetype")), style="dim italic")),
        ("Yoga", Text(str(g("yoga_name")), style="#ef4444" if g("yoga_is_malefic", 0) else "white")),
        ("Karana", Text(str(g("karana_name")), style="#ef4444" if g("karana_is_vishti", 0) else "white")),
        ("Vara", Text(f"{s.date.strftime('%A')}  ·  {g('vara_lord')}", style="white")),
        ("Hora", Text(str(g("hora_lord")), style="white")),
        ("Lagna", Text(f"{float(g('lagna', 0)):.1f}°  ·  {g('lagna_lord')} ruled", style="white")),
        ("Gana", Text(str(g("nakshatra_gana")), style="white")),
    ]
    return Panel(_kv(rows), title="[bold]PANCHANGA[/bold]", border_style=PANEL_BORDER, padding=(1, 2))


def natal_dasha(s: Snapshot) -> Panel:
    r = s.astro_row
    d = s.dasha
    g = lambda k, dv=0: float(r[k]) if k in r and pd.notna(r[k]) else dv

    body = Table.grid(padding=(0, 1))
    body.add_column()

    if d:
        md_prog = float(g("dasha_md_progress"))
        ad_prog = float(g("dasha_ad_progress"))
        t = Text()
        t.append(f"{d['md']}", style="bold #a78bfa")
        t.append(" Mahadasha", style=MUTED)
        t.append(f"   ends {d['md_end'].strftime('%b %Y')}", style="dim")
        body.add_row(t)
        body.add_row(W.gauge(md_prog, width=32, style="#a78bfa", show_pct=True))
        t2 = Text()
        t2.append(f"{d['ad']}", style="bold #67e8f9")
        t2.append(" Antardasha", style=MUTED)
        t2.append(f"   ends {d['ad_end'].strftime('%b %Y')}", style="dim")
        body.add_row(t2)
        body.add_row(W.gauge(ad_prog, width=32, style="#67e8f9", show_pct=True))
        body.add_row(Text(f"  {d['md_character']}", style="dim italic"))
        body.add_row(Text(""))

    tara_q = g("nat_tarabala_quality")
    tara_style = "#22c55e" if tara_q > 0.2 else ("#ef4444" if tara_q < -0.2 else "#f59e0b")
    rows = [
        ("Natal Moon", Text(f"{s.chart['moon_nakshatra_name']}  ({s.chart['moon_nakshatra_lord']})", style="white")),
        ("Tarabala", Text(f"tara {int(g('nat_tarabala'))}   quality {tara_q:+.2f}", style=tara_style)),
        ("Chandrabala", Text("favourable" if g("nat_chandrabala") else "unfavourable",
                             style="#22c55e" if g("nat_chandrabala") else "#ef4444")),
        ("Chandrashtama", Text("ACTIVE" if g("nat_chandrashtama") else "no",
                               style="#ef4444" if g("nat_chandrashtama") else "dim")),
        ("Sade Sati", Text("ACTIVE" if g("nat_sade_sati") else "no",
                           style="#ef4444" if g("nat_sade_sati") else "dim")),
    ]
    body.add_row(_kv(rows))
    return Panel(body, title="[bold]NATAL & DASHA[/bold]", border_style=PANEL_BORDER, padding=(1, 2))


def graha_table(s: Snapshot) -> Panel:
    r = s.astro_row
    rashis = [x["name"] for x in graha_cfg()["rashis"]]
    t = Table(box=None, pad_edge=False, padding=(0, 1), header_style=f"bold {MUTED}")
    t.add_column("Graha", style="white", no_wrap=True)
    t.add_column("Rashi", style=MUTED, no_wrap=True)
    t.add_column("Deg", justify="right", style=MUTED)
    t.add_column("Speed", justify="right", style=MUTED)
    t.add_column("State", no_wrap=True)
    t.add_column("Bala", justify="right")

    for gname in NAVAGRAHA:
        rashi_i = int(r.get(f"{gname}_rashi", 0))
        deg = float(r.get(f"{gname}_rashi_deg", 0.0))
        speed = float(r.get(f"{gname}_speed", 0.0))
        flags = Text()
        if float(r.get(f"{gname}_vakri", 0)):
            flags.append("Vakri ", style="#f59e0b")
        if float(r.get(f"{gname}_stambhana", 0)):
            flags.append("Stambhana ", style="#ef4444")
        if float(r.get(f"{gname}_asta", 0)):
            flags.append("Asta ", style="#a855f7")
        if float(r.get(f"{gname}_is_exalted", 0)):
            flags.append("Uccha ", style="#22c55e")
        if float(r.get(f"{gname}_is_debilitated", 0)):
            flags.append("Neecha ", style="#ef4444")
        if not flags.plain:
            flags.append("-", style="dim")

        rupas = r.get(f"sb_{gname}_rupas", np.nan)
        ratio = r.get(f"sb_{gname}_ratio", np.nan)
        if pd.notna(rupas):
            style = "#22c55e" if (pd.notna(ratio) and ratio >= 1) else "#f59e0b"
            bala = Text(f"{float(rupas):.1f}", style=style)
        else:
            bala = Text("-", style="dim")

        t.add_row(gname, rashis[rashi_i], f"{deg:5.1f}°", f"{speed:+7.3f}", flags, bala)

    return Panel(t, title="[bold]NAVAGRAHA[/bold]", border_style=PANEL_BORDER, padding=(1, 2))


def composites(s: Snapshot) -> Panel:
    r = s.astro_row
    g = lambda k, d=0.0: float(r[k]) if k in r and pd.notna(r[k]) else d
    rows = [
        ("CBI  Chandra Bala", W.bipolar_meter(g("CBI"))),
        ("GSI  Graha Shakti", W.bipolar_meter(g("GSI"))),
        ("VRI  Vriddhi", W.bipolar_meter(g("VRI"))),
        ("BHY  Bhaya (panic)", W.gauge(g("BHY", 0.5), width=24, style="#ef4444")),
        ("KTW  Kala Taranga", W.bipolar_meter(g("KTW"))),
    ]
    return Panel(_kv(rows), title="[bold]COMPOSITE INDICES[/bold]",
                 border_style=PANEL_BORDER, padding=(1, 2))


def charts(s: Snapshot, width: int = 78) -> Panel:
    close = s.prices["Close"].loc[:s.date].tail(180)
    ktw = s.astro_history["KTW"].reindex(close.index)

    body = Table.grid(padding=(0, 1))
    body.add_column()

    body.add_row(Text(f"price · last {len(close)} sessions", style=MUTED))
    body.add_row(W.sparkline(close, width=width, style=ACCENT))
    lo, hi = float(close.min()), float(close.max())
    rng = Text()
    rng.append(f"{lo:,.0f}", style="dim")
    rng.append(" " * max(1, width - len(f"{lo:,.0f}") - len(f"{hi:,.0f}")))
    rng.append(f"{hi:,.0f}", style="dim")
    body.add_row(rng)
    body.add_row(Text(""))

    body.add_row(Text("Kala Taranga · the cosmic tide (astro only, no price input)", style=MUTED))
    body.add_row(W.sparkline(ktw, width=width, style="#a78bfa"))

    if s.forward is not None and len(s.forward):
        body.add_row(Text(""))
        body.add_row(Text("forward 30 days · Bhaya (panic) projection", style=MUTED))
        body.add_row(W.sparkline(s.forward["BHY"], width=width, style="#ef4444"))

    return Panel(body, title="[bold]CHARTS[/bold]", border_style=PANEL_BORDER, padding=(1, 2))


def why(s: Snapshot) -> Panel:
    text = Text()
    for i, sentence in enumerate(s.narrative["sentences"]):
        style = "bold white" if i == 0 else "white"
        text.append(plain(sentence), style=style)
        text.append("\n\n")

    terms = []
    for driver in s.narrative["drivers"][:5]:
        meaning = lexicon.gloss(driver.term) if driver.term else None
        if meaning and driver.term not in [t[0] for t in terms]:
            terms.append((driver.term, meaning))
    if terms:
        text.append("glossary\n", style=f"bold {MUTED}")
        for term, meaning in terms:
            text.append(f"  {term}", style="#a78bfa")
            text.append(f" — {meaning}\n", style="dim")

    return Panel(text, title="[bold]WHY[/bold]", border_style="#a78bfa", padding=(1, 2))


def forward_panel(s: Snapshot, days: int = 12) -> Panel:
    if s.forward is None or s.forward.empty:
        return Panel(Text("(forward calendar unavailable)", style="dim"),
                     title="[bold]FORWARD CALENDAR[/bold]", border_style=PANEL_BORDER)
    t = Table(box=None, pad_edge=False, padding=(0, 1), header_style=f"bold {MUTED}")
    t.add_column("Date", style=MUTED, no_wrap=True)
    t.add_column("Nakshatra", style="white", no_wrap=True)
    t.add_column("Tithi", style=MUTED, no_wrap=True)
    t.add_column("BHY", justify="right")
    t.add_column("Events", style="#f59e0b")

    for date, row in s.forward.head(days).iterrows():
        bhy = float(row["BHY"])
        bstyle = "#ef4444" if bhy > 0.6 else ("#22c55e" if bhy < 0.4 else "#f59e0b")
        events = "; ".join(row["events"]) if row["events"] else ""
        t.add_row(
            date.strftime("%d %b"),
            str(row["nakshatra"]),
            str(row["tithi"]),
            Text(f"{bhy:.2f}", style=bstyle),
            events or Text("-", style="dim"),
        )
    return Panel(t, title="[bold]FORWARD CALENDAR — computable years ahead[/bold]",
                 border_style=PANEL_BORDER, padding=(1, 2))


def backtest_card(s: Snapshot) -> Panel:
    m = s.backtest or {}
    if not m:
        return Panel(
            Text("run  python scripts/backtest.py --all  to populate", style="dim"),
            title="[bold]BACKTEST[/bold]", border_style=PANEL_BORDER, padding=(1, 2),
        )
    strat = m.get("strategy", {})
    bench = m.get("benchmark", {})
    t = Table(box=None, pad_edge=False, padding=(0, 2), header_style=f"bold {MUTED}")
    t.add_column("", style=MUTED)
    t.add_column("ChandraQuant", justify="right", style=f"bold {ACCENT}")
    t.add_column("Buy & Hold", justify="right", style=MUTED)

    def row(label, key, fmt="{:.2%}", better_high=True):
        a, b = strat.get(key), bench.get(key)
        if a is None:
            return
        astr = fmt.format(a)
        bstr = fmt.format(b) if b is not None else "-"
        style = ""
        if b is not None:
            win = (a > b) if better_high else (a < b)
            style = "#22c55e" if win else "#ef4444"
        t.add_row(label, Text(astr, style=style or ACCENT), bstr)

    row("CAGR", "cagr")
    row("Sharpe", "sharpe", "{:.2f}")
    row("Max drawdown", "max_drawdown", "{:.2%}", better_high=True)
    row("Calmar", "calmar", "{:.2f}")
    if "win_rate" in strat:
        row("Win rate", "win_rate")
    return Panel(t, title=f"[bold]BACKTEST — {m.get('period','')}[/bold]",
                 border_style=PANEL_BORDER, padding=(1, 2))


# --------------------------------------------------------------------------------
def render(s: Snapshot, console: Console, compact: bool = False) -> None:
    """Print the whole dashboard."""
    console.print(header(s))
    console.print(Rule(style=PANEL_BORDER))

    # A grid rather than Columns: Columns reflows panels onto their own rows as soon
    # as one is wide, which breaks the three-across layout on normal terminals.
    def side_by_side(*panels):
        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in panels:
            grid.add_column(ratio=1)
        grid.add_row(*panels)
        return grid

    wide = console.width >= 150
    if wide:
        console.print(side_by_side(verdict(s), panchanga(s), natal_dasha(s)))
    else:
        console.print(side_by_side(verdict(s), panchanga(s)))
        console.print(natal_dasha(s))
    console.print(side_by_side(graha_table(s), composites(s)))
    console.print(charts(s, width=min(90, max(40, console.width - 12))))
    console.print(why(s))
    if not compact:
        console.print(side_by_side(forward_panel(s), backtest_card(s)))

    footer = Text()
    footer.append("  q ", style="reverse")
    footer.append(" quit    ", style=MUTED)
    footer.append(" r ", style="reverse")
    footer.append(" refresh    ", style=MUTED)
    footer.append(" w ", style="reverse")
    footer.append(" web dashboard    ", style=MUTED)
    footer.append(" h ", style="reverse")
    footer.append(" honest mode    ", style=MUTED)
    footer.append(" b ", style="reverse")
    footer.append(" back to picker", style=MUTED)
    console.print(footer)
