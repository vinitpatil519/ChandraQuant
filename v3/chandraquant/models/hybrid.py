"""The astro-gated hybrid model.

DESIGN INTENT, stated plainly. In v1 the astro columns were peers of the technical
columns - poured into one Random Forest and left to compete on gain. They lost, badly,
and the hybrid scored *below* the technical-only baseline. Feeding a weak, noisy signal
into a tree ensemble alongside a strong one does not produce a blend; it produces the
strong signal plus variance.

v3 gives astrology a different job. It does not compete with the technical model - it
*governs* it:

    w      = sigmoid(a*GSI + b*CBI + c*VRI - d*BHY)     astro sets the blend weight
    score  = (1 - w) * p_tech + w * p_astro
    theta_q= theta0 + k*BHY - m*VRI                     astro moves the entry percentile
    gate   = 0 if Vishti / eclipse / Chandrashtama / BHY > tau
    signal = gate AND (score > theta)

Three consequences, all deliberate:

1. Astro changes the traded decision on a large fraction of days, so ablating it
   produces a real, measurable difference in both AUC and P&L - not the -0.018 noise
   v1 measured.
2. The entire decision rule is expressible in Sanskrit, which is what the narrative
   engine and the Pine port both need. A rule you cannot say out loud cannot be
   explained to a user.
3. The gate is a *risk* control that happens to be astrological. Standing aside during
   Vishti karana and eclipse windows is, mechanically, just reduced exposure during
   specific recurring calendar windows - which is a legitimate thing for a strategy to
   do, and is measurable either way.

Each block is a LightGBM classifier fit in two passes: once on all features to rank
them by gain, then refit on the top-K. With 520 astro features against ~4,600 rows,
that selection step is what stops the astro block from memorising the training era.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ..features.dataset import Dataset, is_astro_column, prune_features
from .astro_edge import AstroEdgeModel
from .splits import WalkForwardConfig, walk_forward

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42

# Composite columns the gate reads. These are the five branded indices from
# astro/composites.py, plus the hard-gate flags.
GATE_INPUTS = ["GSI", "CBI", "VRI", "BHY"]
HARD_GATES = ["GATE_vishti", "GATE_eclipse", "GATE_chandrashtama"]


@dataclass
class GateParams:
    """Astro gate coefficients. Tuned by backtest/sweeps.py, not hand-waved."""

    w_bias: float = -0.35        # baseline tilt toward the technical model
    w_gsi: float = 0.55
    w_cbi: float = 0.75
    w_vri: float = 0.85
    w_bhy: float = 1.10
    # Entry threshold is expressed as a CAUSAL PERCENTILE of the score's own history,
    # not as an absolute probability. After isotonic calibration to a ~35% base rate a
    # fixed 0.55 cut fires on 0.1% of days; a percentile is scale-free and survives any
    # recalibration. theta_q = 0.62 means "act on the top 38% of scores seen so far".
    theta_q: float = 0.62
    theta_q_bhy: float = 0.14    # fear raises the bar for taking a long
    theta_q_vri: float = 0.10    # expansion lowers it
    bhy_veto: float = 0.86       # above this, stand aside entirely
    use_hard_gates: bool = True

    def blend_weight(self, comp: pd.DataFrame) -> pd.Series:
        z = (
            self.w_bias
            + self.w_gsi * comp["GSI"]
            + self.w_cbi * comp["CBI"]
            + self.w_vri * comp["VRI"]
            - self.w_bhy * (comp["BHY"] * 2.0 - 1.0)
        )
        return 1.0 / (1.0 + np.exp(-z))

    def threshold_quantile(self, comp: pd.DataFrame) -> pd.Series:
        """Astro-modulated entry percentile, in [0.20, 0.95]."""
        q = (
            self.theta_q
            + self.theta_q_bhy * (comp["BHY"] * 2.0 - 1.0)
            - self.theta_q_vri * comp["VRI"]
        )
        return q.clip(0.20, 0.95)

    def hard_gate(self, comp: pd.DataFrame) -> pd.Series:
        open_ = pd.Series(1, index=comp.index, dtype=int)
        if not self.use_hard_gates:
            return open_
        for col in HARD_GATES:
            if col in comp:
                open_ &= (comp[col] == 0).astype(int)
        open_ &= (comp["BHY"] < self.bhy_veto).astype(int)
        return open_


@dataclass
class ModelConfig:
    top_k_tech: int = 45
    top_k_astro: int = 110
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample: float = 0.55
    reg_lambda: float = 5.0
    use_learned_edges: bool = True   # empirical-Bayes conditional edges per astro state
    edge_shrinkage: float = 60.0
    walk: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    gate: GateParams = field(default_factory=GateParams)


def _lgbm(cfg: ModelConfig, seed: int = SEED):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample,
        reg_lambda=cfg.reg_lambda,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _fit_block(X: pd.DataFrame, y: pd.Series, cfg: ModelConfig, top_k: int, seed: int = SEED):
    """Two-pass fit: rank features by gain on the full block, then refit on the top-K."""
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[1] > top_k:
        scout = _lgbm(cfg, seed)
        scout.fit(X, y)
        gains = pd.Series(scout.booster_.feature_importance("gain"), index=X.columns)
        selected = list(gains.sort_values(ascending=False).head(top_k).index)
    else:
        selected = list(X.columns)
    model = _lgbm(cfg, seed)
    model.fit(X[selected], y)
    return model, selected


def _predict_block(model, columns, X: pd.DataFrame) -> np.ndarray:
    Xf = X.reindex(columns=columns)
    Xf = Xf.fillna(Xf.median(numeric_only=True)).fillna(0.0)
    return model.predict_proba(Xf)[:, 1]


@dataclass
class FoldResult:
    fold: int
    test_index: pd.DatetimeIndex
    p_tech: np.ndarray
    p_astro: np.ndarray
    p_hybrid: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    threshold: np.ndarray
    gate_open: np.ndarray


class HybridModel:
    """Technical block + astro block + astro gate, with isotonic calibration."""

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or ModelConfig()
        self.tech_model = None
        self.astro_model = None
        self.tech_cols: list[str] = []
        self.astro_cols: list[str] = []
        self.calibrator: IsotonicRegression | None = None
        self.oof: pd.DataFrame | None = None
        self.fold_results: list[FoldResult] = []
        self.edge_model: AstroEdgeModel | None = None

    # --- internals ------------------------------------------------------------------
    @staticmethod
    def _split_blocks(X: pd.DataFrame) -> tuple[list[str], list[str]]:
        astro = [c for c in X.columns if is_astro_column(c)]
        tech = [c for c in X.columns if not is_astro_column(c)]
        return tech, astro

    def _composites(self, X: pd.DataFrame, idx) -> pd.DataFrame:
        cols = GATE_INPUTS + [c for c in HARD_GATES if c in X.columns]
        comp = X.loc[idx, [c for c in cols if c in X.columns]].copy()
        for c in GATE_INPUTS:
            if c not in comp:
                comp[c] = 0.0
        return comp.fillna(0.0)

    # --- fitting --------------------------------------------------------------------
    def fit_walk_forward(self, ds: Dataset, target: str = "y_vriddhi") -> pd.DataFrame:
        """Run purged walk-forward CV and collect out-of-fold predictions."""
        X, y = ds.xy(target)
        keep = prune_features(X)
        X = X[keep]
        tech_cols, astro_cols = self._split_blocks(X)

        cfg = self.cfg
        cfg.walk.horizon = max(cfg.walk.horizon, 1)
        rows = []
        self.fold_results = []

        for i, (tr, te) in enumerate(walk_forward(X.index, cfg.walk), start=1):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xte, yte = X.iloc[te], y.iloc[te]

            # Learned astro edges, fitted on THIS FOLD'S TRAINING BLOCK ONLY. The test
            # block never touches the estimation, so the edges are genuinely out of
            # sample when they are applied.
            Xtr_a, Xte_a = Xtr[astro_cols], Xte[astro_cols]
            edge_model = None
            if cfg.use_learned_edges:
                edge_model = AstroEdgeModel(shrinkage=cfg.edge_shrinkage).fit(Xtr, ytr)
                Xtr_a = pd.concat([Xtr_a, edge_model.transform(Xtr)], axis=1)
                Xte_a = pd.concat([Xte_a, edge_model.transform(Xte)], axis=1)

            tech_model, tsel = _fit_block(Xtr[tech_cols], ytr, cfg, cfg.top_k_tech)
            astro_model, asel = _fit_block(Xtr_a, ytr, cfg, cfg.top_k_astro, seed=SEED + 1)

            p_tech = _predict_block(tech_model, tsel, Xte)
            p_astro = _predict_block(astro_model, asel, Xte_a)

            comp = self._composites(X, Xte.index)
            if edge_model is not None:
                comp = comp.assign(
                    ASTRO_EDGE=edge_model.transform(Xte)["ASTRO_EDGE"].to_numpy()
                )
            self.edge_model = edge_model
            w = cfg.gate.blend_weight(comp).to_numpy()
            theta = cfg.gate.threshold_quantile(comp).to_numpy()
            gate_open = cfg.gate.hard_gate(comp).to_numpy()

            p_hybrid = (1.0 - w) * p_tech + w * p_astro

            self.fold_results.append(
                FoldResult(i, Xte.index, p_tech, p_astro, p_hybrid, yte.to_numpy(), w, theta, gate_open)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "fold": i,
                        "y": yte.to_numpy(),
                        "p_tech": p_tech,
                        "p_astro": p_astro,
                        "p_hybrid": p_hybrid,
                        "astro_weight": w,
                        "theta_q": theta,
                        "gate_open": gate_open,
                        "astro_edge": comp["ASTRO_EDGE"].to_numpy()
                        if "ASTRO_EDGE" in comp
                        else 0.0,
                    },
                    index=Xte.index,
                )
            )

        oof = pd.concat(rows).sort_index()

        # Isotonic calibration on out-of-fold predictions only.
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(oof["p_hybrid"], oof["y"])
        oof["p_calibrated"] = self.calibrator.predict(oof["p_hybrid"])

        # Causal percentile rank of the score against its own history. This is what the
        # astro-modulated quantile threshold is compared against.
        for col in ("p_hybrid", "p_tech", "p_astro", "p_calibrated"):
            oof[f"{col}_pct"] = oof[col].expanding(min_periods=60).rank(pct=True).fillna(0.5)
        oof["signal"] = (
            (oof["p_calibrated_pct"] > oof["theta_q"]) & (oof["gate_open"] == 1)
        ).astype(int)

        self.oof = oof
        self.tech_cols, self.astro_cols = tech_cols, astro_cols
        return oof

    def fit_full(self, ds: Dataset, target: str = "y_vriddhi") -> "HybridModel":
        """Refit both blocks on all available data, for live inference."""
        X, y = ds.xy(target)
        keep = prune_features(X)
        X = X[keep]
        tech_cols, astro_cols = self._split_blocks(X)
        cfg = self.cfg
        Xa = X[astro_cols]
        self.edge_model = None
        if cfg.use_learned_edges:
            self.edge_model = AstroEdgeModel(shrinkage=cfg.edge_shrinkage).fit(X, y)
            Xa = pd.concat([Xa, self.edge_model.transform(X)], axis=1)
        self.tech_model, self.tech_sel = _fit_block(X[tech_cols], y, cfg, cfg.top_k_tech)
        self.astro_model, self.astro_sel = _fit_block(
            Xa, y, cfg, cfg.top_k_astro, seed=SEED + 1
        )
        self.tech_cols, self.astro_cols = tech_cols, astro_cols
        self.feature_cols = keep
        return self

    # --- inference -------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score new rows. Requires fit_full (or fit_walk_forward then fit_full)."""
        if self.tech_model is None:
            raise RuntimeError("call fit_full() before predict()")
        p_tech = _predict_block(self.tech_model, self.tech_sel, X)
        Xa = X
        if getattr(self, "edge_model", None) is not None:
            Xa = pd.concat([X, self.edge_model.transform(X)], axis=1)
        p_astro = _predict_block(self.astro_model, self.astro_sel, Xa)
        comp = self._composites(X, X.index)
        if getattr(self, "edge_model", None) is not None:
            comp = comp.assign(
                ASTRO_EDGE=self.edge_model.transform(X)["ASTRO_EDGE"].to_numpy()
            )
        w = self.cfg.gate.blend_weight(comp).to_numpy()
        theta = self.cfg.gate.threshold_quantile(comp).to_numpy()
        gate_open = self.cfg.gate.hard_gate(comp).to_numpy()
        p_hybrid = (1.0 - w) * p_tech + w * p_astro
        p_cal = self.calibrator.predict(p_hybrid) if self.calibrator is not None else p_hybrid
        pct = pd.Series(p_cal, index=X.index).expanding(min_periods=60).rank(pct=True).fillna(0.5)
        return pd.DataFrame(
            {
                "p_tech": p_tech,
                "p_astro": p_astro,
                "p_hybrid": p_hybrid,
                "p_calibrated": p_cal,
                "p_calibrated_pct": pct.to_numpy(),
                "astro_weight": w,
                "theta_q": theta,
                "gate_open": gate_open,
                "signal": ((pct.to_numpy() > theta) & (gate_open == 1)).astype(int),
            },
            index=X.index,
        )

    # --- diagnostics ------------------------------------------------------------------
    def auc_table(self) -> pd.DataFrame:
        """Out-of-fold AUC for each block, overall and per fold."""
        assert self.oof is not None, "run fit_walk_forward first"
        rows = []
        for name, grp in [("ALL", self.oof)] + list(self.oof.groupby("fold")):
            rec = {"fold": name, "n": len(grp), "base_rate": float(grp["y"].mean())}
            for col in ("p_tech", "p_astro", "p_hybrid", "p_calibrated"):
                try:
                    rec[col] = roc_auc_score(grp["y"], grp[col])
                except ValueError:
                    rec[col] = np.nan
            rows.append(rec)
        return pd.DataFrame(rows)

    def feature_importance(self, block: str = "astro", top: int = 25) -> pd.Series:
        model = self.astro_model if block == "astro" else self.tech_model
        cols = self.astro_sel if block == "astro" else self.tech_sel
        if model is None:
            raise RuntimeError("call fit_full() first")
        gains = pd.Series(model.booster_.feature_importance("gain"), index=cols)
        return gains.sort_values(ascending=False).head(top)
