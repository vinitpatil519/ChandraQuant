# Setting up ChandraQuant

A step-by-step guide to running ChandraQuant on **Linux, macOS or Windows**. If you follow
it top to bottom you will have the terminal app running in about five minutes, most of
which is downloading packages.

No prior knowledge of astrology, finance or the codebase is assumed.

---

## What you need first

| | Requirement | How to check | If missing |
|---|---|---|---|
| 1 | **Python 3.10 or newer** | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| 2 | **Git** | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| 3 | **~600 MB free disk** | — | mostly scientific packages |
| 4 | **Internet, for first run only** | — | after setup it runs fully offline |

> **Windows note.** If `python --version` opens the Microsoft Store, use `py --version`
> instead and substitute `py` for `python` in every command below.
>
> **macOS note.** If `python` is not found, try `python3`. macOS ships an old Python 2 alias
> in some setups; `python3` is the one you want.

---

## Step 1 — Get the code

```bash
git clone https://github.com/vinitpatil519/ChandraQuant.git
cd ChandraQuant/v3
```

Everything from here happens inside the `v3` folder. (The repository root holds the earlier
v1 research notebooks; `v3` is the engine this guide is about.)

---

## Step 2 — Create a virtual environment

This keeps ChandraQuant's packages separate from the rest of your system. Strongly
recommended, occasionally mandatory — recent Ubuntu and Homebrew Pythons refuse to install
packages system-wide.

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows — PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows — Command Prompt**
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

**Windows — Git Bash**
```bash
py -m venv .venv
source .venv/Scripts/activate
```

Your prompt should now start with `(.venv)`. That means it worked.

> **PowerShell blocks the activate script?** You will see *"running scripts is disabled on
> this system"*. Run this once, then try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

---

## Step 3 — Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The `-e` (editable) flag matters: ChandraQuant reads its classical reference tables from
`config/` and its market snapshot from `data/` relative to the project folder, so it needs
to stay installed in place rather than copied into site-packages.

This pulls in NumPy, pandas, scikit-learn, LightGBM, Skyfield and a few others. Two to four
minutes on a normal connection.

To also run the test suite, install the dev extra instead:

```bash
python -m pip install -e ".[dev]"
```

---

## Step 4 — Fetch market data and the ephemeris

```bash
python scripts/refresh_data.py --all
```

This does three things:

1. Downloads daily OHLCV for **NIFTY 50**, **BANKNIFTY** and **CNX IT** from Yahoo Finance
2. Downloads **NASA JPL DE440s**, the planetary ephemeris (~32 MB, one time, into `data/ephem/`)
3. Writes an offline snapshot to `data/snapshot/` so the app never needs the network again

Expect output like:

```
NIFTY      live      4649 bars  2007-09-17 to 2026-08-28  -> data/snapshot/NIFTY.parquet
           astro cache 6921 rows x 886 features
BANKNIFTY  live      4664 bars  2007-09-17 to 2026-08-28  -> data/snapshot/BANKNIFTY.parquet
...
```

The first run takes **a few minutes**, and most of that is the 32 MB ephemeris download — on
a slow link it can be considerably longer, so let it finish. You will see a progress bar for
the download, then it computes 831 astronomical features for every trading day since 2007
(about 4,650 per ticker). All of it is cached under `data/cache/`, so every subsequent run
starts in a couple of seconds.

Trained models ship with the repository, so the dashboard is ready as soon as this finishes.
If you ever want to retrain from scratch, run `python scripts/train.py`.

> **This step needs internet, and it is the only one that does.** The ephemeris download in
> particular is not optional — without `de440s.bsp` there are no planetary positions and
> nothing else can run. Afterwards the app is fully offline.
>
> **If Yahoo is blocked but you have general internet**, the market-data pull will fail and
> fall back to the snapshot committed in the repository; the ephemeris still downloads and
> everything works. The dashboard header then reads `○ cached` rather than `● live`, which
> is the app telling you the truth about where its data came from.

---

## Step 5 — Run it

```bash
chandraquant
```

You get an animated Kāla-Chakra splash, then a picker:

```
  Select an index  (arrow keys, enter to confirm)

❯ NIFTY 50    ^NSEI      Sthira     24,175.7  +0.35%   Shatabhishaj
  BANKNIFTY   ^NSEBANK   Sthira     57,496.3  -0.02%   Shatabhishaj
  CNX IT      ^CNXIT     Vriddhi    38,204.1  +0.71%   Shatabhishaj
  quit
```

Arrow keys to move, **Enter** to select. The dashboard renders: regime verdict, pañchāṅga,
natal chart and daśā, the nine grahas, composite indices, price and Kāla-Taraṅga
sparklines, a plain-English explanation, the 30-day forward calendar, and backtest results.

At the prompt afterwards: `r` refresh · `b` back to picker · `w` browser dashboard ·
`h` honest mode · `q` quit.

> **`chandraquant: command not found`?** The console script landed somewhere not on your
> PATH. This always works instead:
> ```bash
> python -m chandraquant.cli.app
> ```

---

## Step 6 (optional) — Everything else

```bash
# The full pipeline, in order
python scripts/train.py                  # fit models, cache to artifacts/models/
python scripts/optimize.py               # tune strategy parameters per ticker
python scripts/backtest.py               # -> artifacts/metrics.json
python scripts/validate.py               # -> artifacts/validation.json

# Tests: 60 of them, anchored to NASA and Drik Panchang ground truth
pytest -q
```

---

## Useful flags

| Command | What it does |
|---|---|
| `chandraquant` | splash → picker → dashboard |
| `chandraquant --ticker NIFTY` | skip the picker |
| `chandraquant --date 2020-03-23` | historical replay — the COVID crash bottom |
| `chandraquant --web` | open the browser dashboard |
| `chandraquant --honest` | show unconditional metrics and every caveat |
| `chandraquant --no-refresh` | never touch the network |
| `chandraquant --no-splash` | skip the animation |
| `chandraquant --compact` | shorter dashboard, small terminals |
| `chandraquant --once` | render once and exit (good for screenshots) |
| `chandraquant --retrain` | force model retraining |

---

## Troubleshooting

**The dashboard looks cramped or panels stack vertically.**
It adapts to terminal width and wants **150+ columns** for the three-across layout. Maximise
the window, or reduce the font size, or use `--compact`.

**Boxes render as `??????` or garbled characters (Windows).**
The old `cmd.exe` console does not do UTF-8 by default. Either use **Windows Terminal**
(pre-installed on Windows 11, free on the Store — strongly recommended), or run:
```cmd
chcp 65001
set PYTHONIOENCODING=utf-8
```

**`ModuleNotFoundError: No module named 'chandraquant'`.**
Either the virtual environment is not active — re-run the Step 2 activate command and check
for `(.venv)` in your prompt — or you skipped `pip install -e .`.

**`FileNotFoundError: No data for NIFTY`.**
Run `python scripts/refresh_data.py --all` once while online.

**LightGBM fails to load on macOS.**
It needs the OpenMP runtime:
```bash
brew install libomp
```

**LightGBM or scikit-learn fails to build on Linux.**
You are on a Python version without prebuilt wheels. Use Python 3.11 or 3.12, or install
build tools: `sudo apt install build-essential python3-dev`.

**First run is slow.**
Expected. It computes 831 astronomical features per trading day across ~4,650 days per
ticker. Everything is cached in `data/cache/`; subsequent runs take a few seconds.

**Delete `data/cache/` to force recomputation** if anything ever looks stale.

---

## TradingView scripts

Nothing to install. Open [TradingView](https://www.tradingview.com/chart/), open the
**Pine Editor** at the bottom, and paste in either file from `v3/pine/`:

- **`ChandraQuant_KalaChakra.pine`** — the indicator: regime bands, nakṣatra markers,
  eclipse shading, live pañchāṅga table
- **`ChandraQuant_Strategy.pine`** — the backtestable strategy, with the Jyotiṣa gate as a
  toggle so you can measure what it contributes

Click **Add to chart**. Apply to `NSE:NIFTY` on the daily timeframe. Both compute planetary
positions live from Meeus series — no data feed, no expiry date, works on any bar.

---

## What gets created where

```
v3/
├── data/ephem/de440s.bsp     NASA planetary ephemeris (~32 MB, downloaded once)
├── data/raw/                 live OHLCV pulls
├── data/snapshot/            committed offline fallback
├── data/cache/               computed astro features (delete to recompute)
└── artifacts/
    ├── metrics.json          backtest results — the source of every number displayed
    ├── validation.json       ablation, permutation, bootstrap, crisis analysis
    ├── models/               trained models
    └── figures/              rendered HTML dashboards
```

---

## One-liner, for the impatient

```bash
git clone https://github.com/vinitpatil519/ChandraQuant.git && cd ChandraQuant/v3 && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -e . && python scripts/refresh_data.py --all && chandraquant
```

(On Windows, swap the venv activation line for `.venv\Scripts\Activate.ps1`.)
