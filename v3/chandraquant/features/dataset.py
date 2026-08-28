"""Dataset assembly - joins OHLCV, technical features, the Jyotisha matrix and labels.

The join is on the *trading* calendar. Astro is computed for every calendar day and
then reindexed onto trading days; no forward-filling is needed because every astro
quantity is defined for every instant, and none of them is a market observation.

Leak discipline, made explicit:
  - technical features at bar t use only bars <= t (proved by technical.assert_causal)
  - astro features at bar t are deterministic functions of t itself, so they are
    computable years ahead - the one genuinely forward-looking family in the system
  - labels look forward by H bars and are the ONLY forward-looking columns; the CV
    splitter purges an H-bar embargo around every fold boundary to keep them honest
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..astro import engine as astro_engine
from ..data.loaders import DataStatus, load_ohlcv
from ..labels import regime as regime_labels
from ..labels.regime import LabelConfig
from . import technical

# Astro column families. Kept explicit so the model layer can ablate cleanly - the
# permutation and astro-only experiments both need to know exactly which columns are
# celestial and which are not.
ASTRO_PREFIXES = (
    "tithi_", "nakshatra_", "nak_", "yoga_", "karana_", "vara_", "paksha_", "gana_",
    "moon_", "phase_", "shara_", "kriya_", "avastha_", "chandra_", "gandanta_",
    "surya_", "masa_", "ritu_", "sun_", "days_to_sankranti", "days_since_sankranti",
    "sankranti_", "is_makar", "is_mesha", "is_uttarayana", "is_purnima", "is_amavasya",
    "sep_", "asp_", "aspect_", "drishti_", "siderograph", "kala_taranga",
    "is_eclipse", "is_solar", "is_lunar", "is_central", "eclipse_", "grahana_",
    "sutak_", "guru", "rahu_", "great_conjunction", "event_stress", "gurushani_",
    "nat_", "dasha_", "av_", "sb_", "lagna", "hora_", "rahu_kaal", "yamaganda",
    "gulika", "abhijit", "daylight", "sunrise", "sunset",
    "benefic_", "malefic_", "graha_", "n_vakri", "n_asta", "n_exalted", "n_debilitated",
    "CBI", "GSI", "VRI", "BHY", "KTW", "ASTRO_", "GATE_", "ayanamsa", "elongation",
    "lunar_instability", "Surya_", "Chandra_", "Mangala_", "Budha_", "Guru_",
    "Shukra_", "Shani_", "Rahu_", "Ketu_", "Aruna_", "Varuna_",
)


def is_astro_column(col: str) -> bool:
    return col.startswith(ASTRO_PREFIXES)


@dataclass
class Dataset:
    ticker: str
    prices: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    status: DataStatus
    astro_cols: list[str] = field(default_factory=list)
    tech_cols: list[str] = field(default_factory=list)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.features.index

    def xy(self, target: str = "y_vriddhi", dropna: bool = True):
        """Aligned (X, y) with rows lacking a label or too many NaNs removed."""
        X = self.features
        y = self.labels[target]
        mask = y.notna()
        if dropna:
            # Require most features present; early bars lack 200-day windows.
            coverage = X.notna().mean(axis=1)
            mask &= coverage > 0.85
        return X.loc[mask], y.loc[mask]

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "rows": len(self.features),
            "start": str(self.index[0].date()),
            "end": str(self.index[-1].date()),
            "n_features": self.features.shape[1],
            "n_astro": len(self.astro_cols),
            "n_technical": len(self.tech_cols),
            "source": self.status.source,
            "base_rates": regime_labels.base_rates(self.labels),
        }


def build(
    ticker_key: str,
    refresh: bool = True,
    label_cfg: LabelConfig | None = None,
    verify_causal: bool = False,
) -> Dataset:
    """Assemble the full modelling dataset for one index."""
    prices, status = load_ohlcv(ticker_key, refresh=refresh)

    tech = technical.compute(prices)
    if verify_causal:
        technical.assert_causal(prices, tech)

    astro_full = astro_engine.build(prices.index, ticker_key)
    astro_num = astro_engine.numeric_matrix(astro_full).reindex(prices.index)

    # Guard the core invariant: no astro column may be a market observation.
    overlap = set(astro_num.columns) & set(tech.columns)
    if overlap:
        raise AssertionError(f"astro/technical column collision: {sorted(overlap)[:5]}")

    features = pd.concat([tech, astro_num], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)

    labels = regime_labels.compute(prices, label_cfg)

    astro_cols = [c for c in features.columns if c in astro_num.columns]
    tech_cols = [c for c in features.columns if c in tech.columns]

    return Dataset(
        ticker=ticker_key,
        prices=prices,
        features=features,
        labels=labels,
        status=status,
        astro_cols=astro_cols,
        tech_cols=tech_cols,
    )


def prune_features(
    X: pd.DataFrame,
    max_nan_frac: float = 0.25,
    corr_threshold: float = 0.995,
    min_unique: int = 3,
) -> list[str]:
    """Drop dead and duplicate columns.

    831 astro features against ~4,600 rows is a punishing ratio, and much of the astro
    matrix is near-duplicated by construction (a cyclic encoding and its own harmonic,
    a flag and its own ramp). This removes columns that are empty, constant, or
    effectively a copy of a column already kept - without looking at the target, so it
    cannot leak.
    """
    keep = [c for c in X.columns if X[c].isna().mean() <= max_nan_frac]
    keep = [c for c in keep if X[c].nunique(dropna=True) >= min_unique]

    sub = X[keep].fillna(X[keep].median(numeric_only=True))
    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if (upper[c] > corr_threshold).any()}
    return [c for c in keep if c not in drop]
