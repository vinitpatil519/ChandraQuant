"""Optional browser dashboard - launched only by `--web` or the `w` key, never on its own.

Self-contained: one HTML page, inline SVG charts, no CDN, no build step. It renders the
same Snapshot the terminal does, so the two surfaces cannot disagree.
"""

from __future__ import annotations

import json
import threading
import webbrowser

import numpy as np
import pandas as pd

from ..config import TICKER_KEYS
from ..explain.narrative import plain
from ..inference import Snapshot, snapshot

COLOURS = {
    "Vriddhi": "#22c55e",
    "Sthira": "#f59e0b",
    "Kshaya": "#ef4444",
    "Kshobha": "#a855f7",
}


def _sparkline_svg(values, width=760, height=120, colour="#22d3ee", fill=True) -> str:
    s = pd.Series(values).dropna()
    if s.empty:
        return ""
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo if hi > lo else 1.0
    n = len(s)
    pts = [
        (i / max(1, n - 1) * width, height - (float(v) - lo) / rng * (height - 8) - 4)
        for i, v in enumerate(s)
    ]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    area = ""
    if fill:
        area = (
            f'<path d="{path} L{width},{height} L0,{height} Z" '
            f'fill="{colour}" opacity="0.12"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="width:100%;height:{height}px">{area}'
        f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"/></svg>'
    )


def _meter(label: str, value: float, lo=-1.0, hi=1.0, colour="#22d3ee") -> str:
    frac = float(np.clip((value - lo) / (hi - lo), 0, 1)) * 100
    return f"""
    <div class="meter">
      <div class="meter-label"><span>{label}</span><span class="mono">{value:+.2f}</span></div>
      <div class="track"><div class="fill" style="width:{frac:.1f}%;background:{colour}"></div></div>
    </div>"""


def render_html(s: Snapshot) -> str:
    colour = COLOURS.get(s.detected_name, "#22d3ee")
    close = s.prices["Close"].loc[: s.date].tail(250)
    ktw = s.astro_history["KTW"].reindex(close.index)
    r = s.astro_row
    g = lambda k, d=0.0: float(r[k]) if k in r and pd.notna(r[k]) else d
    t = lambda k, d="": str(r[k]) if k in r and pd.notna(r[k]) else d

    bt = s.backtest or {}
    strat, bench = bt.get("strategy", {}), bt.get("benchmark", {})

    def row(label, key, fmt="{:.2%}"):
        if key not in strat:
            return ""
        a = fmt.format(strat[key])
        b = fmt.format(bench[key]) if key in bench else "-"
        win = strat[key] > bench.get(key, -9e9)
        return (
            f"<tr><td>{label}</td>"
            f"<td class='mono' style='color:{'#22c55e' if win else '#ef4444'}'>{a}</td>"
            f"<td class='mono muted'>{b}</td></tr>"
        )

    forward_rows = ""
    if s.forward is not None:
        for date, fr in s.forward.head(14).iterrows():
            ev = "; ".join(fr["events"]) if fr["events"] else "—"
            bhy = float(fr["BHY"])
            bc = "#ef4444" if bhy > 0.6 else "#22c55e" if bhy < 0.4 else "#f59e0b"
            forward_rows += (
                f"<tr><td class='mono'>{date:%d %b}</td><td>{fr['nakshatra']}</td>"
                f"<td class='muted'>{fr['tithi']}</td>"
                f"<td class='mono' style='color:{bc}'>{bhy:.2f}</td>"
                f"<td class='muted'>{ev}</td></tr>"
            )

    graha_rows = ""
    from ..astro.ephemeris import NAVAGRAHA
    from ..config import graha_cfg

    rashis = [x["name"] for x in graha_cfg()["rashis"]]
    for gr in NAVAGRAHA:
        flags = []
        if g(f"{gr}_vakri"):
            flags.append("<span class='tag warn'>Vakrī</span>")
        if g(f"{gr}_asta"):
            flags.append("<span class='tag purple'>Asta</span>")
        if g(f"{gr}_is_exalted"):
            flags.append("<span class='tag good'>Uccha</span>")
        if g(f"{gr}_is_debilitated"):
            flags.append("<span class='tag bad'>Nīcha</span>")
        graha_rows += (
            f"<tr><td>{gr}</td><td class='muted'>{rashis[int(g(f'{gr}_rashi'))]}</td>"
            f"<td class='mono'>{g(f'{gr}_rashi_deg'):.1f}°</td>"
            f"<td class='mono'>{g(f'{gr}_speed'):+.3f}</td>"
            f"<td>{''.join(flags) or '—'}</td></tr>"
        )

    narrative_html = "".join(
        f"<p>{plain(x)}</p>" for x in s.narrative["sentences"]
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ChandraQuant — {s.display}</title>
<style>
:root {{ --bg:#0b1120; --panel:#111827; --line:#1f2937; --text:#e5e7eb; --muted:#94a3b8; --accent:#22d3ee; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
.mono {{ font-family:'SF Mono',Menlo,Consolas,monospace; font-variant-numeric:tabular-nums }}
.muted {{ color:var(--muted) }}
.wrap {{ max-width:1180px; margin:0 auto; padding:28px 20px 64px }}
header {{ display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; margin-bottom:6px }}
h1 {{ margin:0; font-size:26px; color:var(--accent); letter-spacing:-.01em }}
.price {{ font-size:26px; font-weight:600 }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:24px }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)) }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px }}
.panel h2 {{ margin:0 0 14px; font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600 }}
.verdict {{ font-size:34px; font-weight:700; letter-spacing:-.02em; color:{colour} }}
table {{ width:100%; border-collapse:collapse; font-size:14px }}
td, th {{ padding:5px 8px 5px 0; text-align:left; border-bottom:1px solid var(--line) }}
tr:last-child td {{ border-bottom:none }}
th {{ color:var(--muted); font-weight:500; font-size:12px }}
.meter {{ margin-bottom:11px }}
.meter-label {{ display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin-bottom:4px }}
.track {{ height:7px; background:#1f2937; border-radius:4px; overflow:hidden }}
.fill {{ height:100%; border-radius:4px }}
.tag {{ font-size:11px; padding:1px 7px; border-radius:4px; margin-right:4px }}
.tag.warn {{ background:#78350f; color:#fbbf24 }} .tag.bad {{ background:#7f1d1d; color:#fca5a5 }}
.tag.good {{ background:#14532d; color:#86efac }} .tag.purple {{ background:#4c1d95; color:#c4b5fd }}
.kv {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--line); font-size:14px }}
.kv:last-child {{ border-bottom:none }}
.wide {{ grid-column:1/-1 }}
p {{ margin:0 0 12px }}
footer {{ margin-top:36px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:16px }}
.pill {{ display:inline-block; padding:3px 10px; border-radius:999px; background:#0f172a; border:1px solid var(--line); font-size:12px; margin-right:6px }}
</style></head><body><div class="wrap">

<header>
  <h1>{s.display}</h1>
  <span class="price mono">{s.close:,.2f}</span>
  <span class="mono" style="color:{'#22c55e' if s.change_1d>=0 else '#ef4444'}">{s.change_1d*100:+.2f}%</span>
</header>
<div class="sub">{s.date:%A, %d %B %Y} &nbsp;·&nbsp; {s.status.badge()} &nbsp;·&nbsp;
  natal {s.chart['natal_local']:%Y-%m-%d %H:%M} IST &nbsp;·&nbsp;
  natal Moon {s.chart['moon_nakshatra_name']}</div>

<div class="grid">
  <div class="panel">
    <h2>Verdict</h2>
    <div class="verdict">{s.detected_name.upper()}</div>
    <p class="muted" style="font-size:13px">{s.detected_meaning}</p>
    {_meter("P(Vṛddhi) next 5 sessions", s.probability if np.isfinite(s.probability) else .5, 0, 1, colour)}
    {_meter("technical block", s.tech_probability if np.isfinite(s.tech_probability) else .5, 0, 1, "#38bdf8")}
    {_meter("astro block", s.astro_probability if np.isfinite(s.astro_probability) else .5, 0, 1, "#a78bfa")}
    <div class="kv"><span class="muted">target exposure</span><span class="mono">{s.position:.2f}×</span></div>
  </div>

  <div class="panel">
    <h2>Pañchāṅga</h2>
    <div class="kv"><span class="muted">Tithi</span><span>{t('tithi_name')}</span></div>
    <div class="kv"><span class="muted">Nakṣatra</span><span>{t('nakshatra_name')} · pada {int(g('pada',1))}</span></div>
    <div class="kv"><span class="muted">Yoga</span><span>{t('yoga_name')}</span></div>
    <div class="kv"><span class="muted">Karaṇa</span><span>{t('karana_name')}</span></div>
    <div class="kv"><span class="muted">Vāra</span><span>{s.date:%A} · {t('vara_lord')}</span></div>
    <div class="kv"><span class="muted">Hora</span><span>{t('hora_lord')}</span></div>
    <div class="kv"><span class="muted">Gaṇa</span><span>{t('nakshatra_gana')}</span></div>
  </div>

  <div class="panel">
    <h2>Natal &amp; Daśā</h2>
    <div class="kv"><span class="muted">Mahādaśā</span><span>{s.dasha.get('md','—')} → {s.dasha.get('md_end').strftime('%b %Y') if s.dasha else '—'}</span></div>
    <div class="kv"><span class="muted">Antardaśā</span><span>{s.dasha.get('ad','—')}</span></div>
    <div class="kv"><span class="muted">Tārābala</span><span>tara {int(g('nat_tarabala'))} · {g('nat_tarabala_quality'):+.2f}</span></div>
    <div class="kv"><span class="muted">Chandrabala</span><span>{'favourable' if g('nat_chandrabala') else 'unfavourable'}</span></div>
    <div class="kv"><span class="muted">Chandrāṣṭama</span><span>{'ACTIVE' if g('nat_chandrashtama') else 'no'}</span></div>
    <div class="kv"><span class="muted">Sade Sati</span><span>{'ACTIVE' if g('nat_sade_sati') else 'no'}</span></div>
  </div>

  <div class="panel wide">
    <h2>Price · last {len(close)} sessions</h2>
    {_sparkline_svg(close, colour=colour)}
    <h2 style="margin-top:20px">Kāla Taraṅga · the cosmic tide (astro only, no price input)</h2>
    {_sparkline_svg(ktw, colour="#a78bfa", fill=False)}
  </div>

  <div class="panel">
    <h2>Composite indices</h2>
    {_meter("CBI · Chandra Bala", g("CBI"))}
    {_meter("GSI · Graha Śakti", g("GSI"))}
    {_meter("VRI · Vṛddhi", g("VRI"))}
    {_meter("BHY · Bhaya (panic)", g("BHY", .5), 0, 1, "#ef4444")}
    {_meter("KTW · Kāla Taraṅga", g("KTW"), -1, 1, "#a78bfa")}
  </div>

  <div class="panel">
    <h2>Navagraha</h2>
    <table><tr><th>Graha</th><th>Rāśi</th><th>Deg</th><th>Speed</th><th>State</th></tr>
    {graha_rows}</table>
  </div>

  <div class="panel">
    <h2>Backtest</h2>
    <table><tr><th></th><th>ChandraQuant</th><th>Buy &amp; Hold</th></tr>
      {row("CAGR","cagr")}{row("Sharpe","sharpe","{:.2f}")}
      {row("Max drawdown","max_drawdown")}{row("Calmar","calmar","{:.2f}")}
    </table>
    <p class="muted" style="font-size:12px;margin-top:10px">{bt.get('period','')}</p>
  </div>

  <div class="panel wide">
    <h2>Why</h2>
    {narrative_html}
  </div>

  <div class="panel wide">
    <h2>Forward calendar — computable years ahead</h2>
    <table><tr><th>Date</th><th>Nakṣatra</th><th>Tithi</th><th>BHY</th><th>Events</th></tr>
    {forward_rows}</table>
  </div>
</div>

<footer>
  <span class="pill">ChandraQuant v3</span>
  <span class="pill">Skyfield · JPL DE440s</span>
  <span class="pill">831 astro features</span>
  <span class="pill">Lahiri ayanāṁśa {g('ayanamsa'):.4f}°</span>
  <br><br>
  Position sizing is trend + volatility targeting. The Jyotiṣa engine drives regime
  narrative, the forward calendar and the composite indices. See docs/METHODOLOGY.md for
  what the astro layer does and does not predict.
</footer>
</div></body></html>"""


def serve(ticker: str = "NIFTY", port: int = 8731, open_browser: bool = True) -> None:
    """Render once and serve on localhost. Blocks until interrupted."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    pages: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            key = self.path.strip("/").upper() or ticker.upper()
            if key not in TICKER_KEYS:
                key = ticker.upper()
            if key not in pages:
                pages[key] = render_html(snapshot(key, refresh=False))
            body = pages[key].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # silence the default access log
            pass

    httpd = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{ticker.upper()}"
    print(f"  ChandraQuant dashboard → {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


def write_static(ticker: str, path: str) -> str:
    """Render to a standalone HTML file - handy for screenshots and sharing."""
    html = render_html(snapshot(ticker, refresh=False))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
