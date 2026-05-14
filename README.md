# ChandraQuant / Graha-Sūcana

Converted Python project codebase for the **Graha-Sūcana v2.1 / ChandraQuant** notebook.

The notebook was originally a single Google Colab-style cell. This repository keeps the model logic intact and moves it into a GitHub-ready Python package structure.

## What the project does

The pipeline evaluates whether astronomical phase embeddings add signal to Indian market regime prediction. It compares five models:

1. **M1 Buy & Hold** — always bullish baseline
2. **M2 Moving Average Crossover** — MA50 > MA200 baseline
3. **M3 Technical RF** — Random Forest using technical indicators
4. **M4 Astro-Only RF** — Random Forest using Moon/Sun phase and Nakshatra features
5. **M5 Hybrid RF** — Random Forest using both technical and astronomical features

The pipeline downloads market data, computes astronomical features using Skyfield, engineers technical indicators, trains the models, runs classification/economic backtests, performs bootstrap and permutation tests, generates visualizations, and prints a 30-day future probability projection.

## Project structure

```text
.
├── notebooks/
│   └── ChandraQuant_Model_original.ipynb
├── scripts/
│   └── run_experiment.py
├── src/
│   └── chandraquant/
│       ├── __init__.py
│       ├── __main__.py
│       └── pipeline.py
├── tests/
│   └── test_structure.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

## Run

From the project root:

```bash
python scripts/run_experiment.py
```

Alternative module run:

```bash
python -m chandraquant
```

## Important conversion note

The uploaded notebook contains two very similar large code cells. The second cell was used as the project source because it appears to be the newer/final version: it uses a 15-day forward horizon and includes the updated future technical-feature extrapolation logic.

## Notes

- The original notebook-time `pip install` commands were moved into `requirements.txt` and `pyproject.toml`.
- The main execution code is wrapped in `run_experiment()` so importing the package does not automatically start the full experiment.
- The only robustness adjustment is date handling inside `compute_astro()`, so the same feature calculation works across pandas/Python datetime types.
- The experiment downloads live market data through `yfinance`, so results can vary depending on the date and data availability.
- Skyfield may download `de421.bsp` the first time the experiment runs.
