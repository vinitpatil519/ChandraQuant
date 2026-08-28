"""Learned astro edges - empirical-Bayes conditional win rates per celestial state.

WHY THIS EXISTS. The first version of `config/*.yaml` carried a hand-assigned
`market_bias` for every nakshatra, tithi, yoga and karana - Pushya +0.60 because it is
the "nourisher", Mula -0.55 because it tears out roots. Those priors are faithful to
the classical texts and completely unvalidated against price. Backtested directly they
were actively harmful: filtering on ASTRO_SCORE > 0 cut NIFTY's CAGR from 9.2% to
0.1%. Reading a Sanskrit adjective and guessing a sign does not produce alpha.

So the biases are *measured* instead. For each celestial state family - the 27
nakshatras, the 30 tithis, the 9 taras, the dasha lords, and so on - this estimates the
conditional edge from TRAINING DATA ONLY, then shrinks it toward the global mean:

    edge(state) = (n_s * mean_s + k * mean_global) / (n_s + k)

Empirical-Bayes shrinkage is what keeps this honest. A nakshatra the Moon has occupied
40 times in the training window will show a wild raw win rate purely by chance; with
k = 60 pseudo-observations, a state needs real, repeated evidence before its edge moves
far from the base rate. Rare states collapse to the mean and contribute nothing, which
is the correct behaviour.

Fitted inside each walk-forward fold on the training block alone, so the edges applied
to a test block were never exposed to it. This is also precisely the "Nakshatra-
stratified win rate" analysis the v1 paper describes - only computed out-of-sample
rather than over the full history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Celestial state families worth conditioning on. Each maps a column to the number of
# distinct levels it can take, which is used for sanity checks and reporting.
STATE_FAMILIES: dict[str, int] = {
    "nakshatra_index": 27,      # the Moon's mansion
    "tithi_index": 30,          # lunar day
    "yoga_index": 27,           # nitya yoga
    "karana_index": 11,         # half-tithi
    "vara": 7,                  # weekday
    "nat_tarabala": 9,          # natal-relative tara - ticker specific
    "nat_chandra_house": 12,    # transit Moon from natal Moon
    "dasha_md_idx": 9,          # Mahadasha lord
    "dasha_ad_idx": 9,          # Antardasha lord
    "surya_rashi": 12,          # solar month
    "lagna_rashi": 12,          # rising sign at the open
    "Guru_rashi": 12,           # Jupiter's sign - the macro band
    "Shani_rashi": 12,          # Saturn's sign
    "Rahu_rashi": 12,           # nodal band
    "chandra_avastha": 12,      # lunar phase state
    "masa_index": 12,           # sidereal solar month
    "ritu_index": 6,            # season
}

# Binary / flag families, handled the same way but with only two levels.
FLAG_FAMILIES = [
    "karana_is_vishti",
    "tithi_is_rikta",
    "yoga_is_malefic",
    "paksha_shukla",
    "moon_in_gandanta",
    "nat_chandrashtama",
    "nat_sade_sati",
    "eclipse_window_3d",
    "Budha_vakri",
    "Shani_vakri",
    "Guru_vakri",
    "is_uttarayana",
]


@dataclass
class AstroEdgeModel:
    """Empirical-Bayes conditional edges for celestial state families."""

    shrinkage: float = 60.0     # pseudo-observations pulling each state to the mean
    min_count: int = 15         # below this a state is forced to the global mean
    families: dict[str, int] = field(default_factory=lambda: dict(STATE_FAMILIES))
    flags: list[str] = field(default_factory=lambda: list(FLAG_FAMILIES))

    def __post_init__(self):
        self.global_mean_: float = 0.5
        self.tables_: dict[str, pd.Series] = {}
        self.counts_: dict[str, pd.Series] = {}
        self.fitted_: list[str] = []

    # --- fitting -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AstroEdgeModel":
        y = y.astype(float)
        self.global_mean_ = float(y.mean())
        self.tables_.clear()
        self.counts_.clear()
        self.fitted_ = []

        for col in list(self.families) + list(self.flags):
            if col not in X.columns:
                continue
            states = X[col]
            if states.isna().all():
                continue
            key = states.fillna(-999).astype(float).round(0)
            grouped = y.groupby(key)
            counts = grouped.count()
            means = grouped.mean()

            shrunk = (counts * means + self.shrinkage * self.global_mean_) / (
                counts + self.shrinkage
            )
            shrunk[counts < self.min_count] = self.global_mean_

            self.tables_[col] = shrunk
            self.counts_[col] = counts
            self.fitted_.append(col)
        return self

    # --- transform ----------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """One learned-edge column per family, plus their aggregate."""
        out: dict[str, np.ndarray] = {}
        for col in self.fitted_:
            key = X[col].fillna(-999).astype(float).round(0)
            mapped = key.map(self.tables_[col]).astype(float)
            out[f"edge_{col}"] = mapped.fillna(self.global_mean_).to_numpy()

        frame = pd.DataFrame(out, index=X.index)
        if frame.empty:
            return frame
        # Centre each edge on the global base rate so 0 means "no information".
        centred = frame - self.global_mean_
        frame["edge_mean"] = centred.mean(axis=1)
        frame["edge_sum"] = centred.sum(axis=1)
        # Weight each family by how far it deviates - families that separate the target
        # strongly get more say than families that sit on the base rate.
        weights = centred.abs().mean(axis=0)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        frame["edge_weighted"] = (centred * weights).sum(axis=1)
        frame["ASTRO_EDGE"] = frame["edge_weighted"] / (centred.std(axis=0).mean() + 1e-9)
        return frame

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # --- reporting -----------------------------------------------------------------
    def stratified_table(self, family: str, labels: dict | None = None) -> pd.DataFrame:
        """The per-state table: raw win rate, shrunken edge, and sample size.

        This is the headline artefact - the v1 paper's "Nakshatra-stratified win rates",
        computed properly with shrinkage and reported with the counts that justify them.
        """
        if family not in self.tables_:
            raise KeyError(f"{family} not fitted. Available: {self.fitted_}")
        counts = self.counts_[family]
        shrunk = self.tables_[family]
        raw = (shrunk * (counts + self.shrinkage) - self.shrinkage * self.global_mean_) / counts
        df = pd.DataFrame(
            {
                "state": counts.index.astype(int),
                "n": counts.to_numpy(),
                "raw_rate": raw.to_numpy(),
                "shrunk_rate": shrunk.to_numpy(),
                "edge": shrunk.to_numpy() - self.global_mean_,
            }
        )
        if labels:
            df["name"] = df["state"].map(labels)
        return df.sort_values("edge", ascending=False).reset_index(drop=True)

    def family_strength(self) -> pd.Series:
        """How much each family separates the target - the ranking used for weighting."""
        rows = {}
        for col in self.fitted_:
            rows[col] = float((self.tables_[col] - self.global_mean_).abs().mean())
        return pd.Series(rows).sort_values(ascending=False)
