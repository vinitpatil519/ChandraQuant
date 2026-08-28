"""Project paths and cached loaders for the classical reference tables in config/."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshot"
EPHEM_DIR = DATA_DIR / "ephem"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
FIGURE_DIR = ARTIFACT_DIR / "figures"
PINE_DIR = ROOT / "pine"

for _d in (RAW_DIR, SNAPSHOT_DIR, EPHEM_DIR, CACHE_DIR, MODEL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML table from config/. Cached - these are read constantly."""
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def nakshatra_cfg() -> dict[str, Any]:
    return load_yaml("nakshatra.yaml")


def tithi_cfg() -> dict[str, Any]:
    return load_yaml("tithi.yaml")


def yoga_cfg() -> dict[str, Any]:
    return load_yaml("yoga.yaml")


def karana_cfg() -> dict[str, Any]:
    return load_yaml("karana.yaml")


def graha_cfg() -> dict[str, Any]:
    return load_yaml("graha.yaml")


def ashtakavarga_cfg() -> dict[str, Any]:
    return load_yaml("ashtakavarga.yaml")


def tickers_cfg() -> dict[str, Any]:
    return load_yaml("tickers.yaml")


@functools.lru_cache(maxsize=None)
def ticker_map() -> dict[str, dict[str, Any]]:
    """Ticker key -> its config block, with `defaults` merged in."""
    cfg = tickers_cfg()
    defaults = cfg["defaults"]
    out: dict[str, dict[str, Any]] = {}
    for entry in cfg["tickers"]:
        merged = dict(defaults)
        merged.update(entry)
        out[entry["key"]] = merged
    return out


def get_ticker(key: str) -> dict[str, Any]:
    """Resolve a ticker by key, name or Yahoo symbol. Case-insensitive."""
    tm = ticker_map()
    if key in tm:
        return tm[key]
    lowered = key.lower().replace(" ", "").replace("-", "")
    for tkey, entry in tm.items():
        candidates = {
            tkey.lower(),
            entry["name"].lower().replace(" ", ""),
            entry["display"].lower().replace(" ", ""),
            entry["yahoo"].lower(),
        }
        if lowered in candidates:
            return entry
    raise KeyError(f"Unknown ticker {key!r}. Known: {sorted(tm)}")


TICKER_KEYS = ["NIFTY", "BANKNIFTY", "CNXIT"]
