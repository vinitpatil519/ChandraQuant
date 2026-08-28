"""The validation battery - ablation, permutation, bootstrap, crisis, stratification.

Writes artifacts/validation.json. Every claim the report makes about the astro layer
has to survive one of these tests, and the ones it fails are recorded too. A validation
suite that only reports its wins is decoration, not evidence.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from chandraquant.config import ARTIFACT_DIR, TICKER_KEYS, nakshatra_cfg, tickers_cfg
from chandraquant.features import dataset as ds
from chandraquant.features.dataset import is_astro_column, prune_features
from chandraquant.models.astro_edge import AstroEdgeModel
from chandraquant.models.hybrid import HybridModel

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------------
def ablation(model: HybridModel) -> dict:
    """M1..M5 style comparison, on the same out-of-fold predictions."""
    oof = model.oof
    out = {}
    for name, col in (
        ("M3_technical", "p_tech"),
        ("M4_astro_only", "p_astro"),
        ("M5_hybrid", "p_hybrid"),
        ("M5_calibrated", "p_calibrated"),
    ):
        out[name] = float(roc_auc_score(oof["y"], oof[col]))
    out["delta_hybrid_vs_technical"] = out["M5_hybrid"] - out["M3_technical"]
    out["delta_astro_vs_chance"] = out["M4_astro_only"] - 0.5
    out["base_rate"] = float(oof["y"].mean())
    return out


def bootstrap_auc(model: HybridModel, n: int = 1000) -> dict:
    """Confidence intervals on each block's AUC and on the hybrid-minus-technical delta."""
    oof = model.oof
    y = oof["y"].to_numpy()
    pt, ph = oof["p_tech"].to_numpy(), oof["p_hybrid"].to_numpy()
    idx = np.arange(len(y))
    deltas, hybrids = [], []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        a = roc_auc_score(y[s], ph[s])
        b = roc_auc_score(y[s], pt[s])
        hybrids.append(a)
        deltas.append(a - b)
    return {
        "n_resamples": len(deltas),
        "hybrid_auc_mean": float(np.mean(hybrids)),
        "hybrid_auc_ci95": [float(np.percentile(hybrids, 2.5)), float(np.percentile(hybrids, 97.5))],
        "delta_mean": float(np.mean(deltas)),
        "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "delta_positive_fraction": float(np.mean(np.array(deltas) > 0)),
    }


def permutation_test(data: ds.Dataset, n: int = 60) -> dict:
    """Shuffle the astro block against the calendar and see if the edge survives.

    Shuffling *rows* of the astro columns severs the link between sky and date while
    preserving each feature's marginal distribution exactly. If the real arrangement
    carries information, the true AUC should sit outside the permuted null.
    """
    X, y = data.xy("y_vriddhi")
    keep = prune_features(X)
    X = X[keep]
    astro_cols = [c for c in X.columns if is_astro_column(c)]
    cut = int(len(X) * 0.65)
    Xtr, ytr, Xte, yte = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

    def fit_score(Xa_tr, Xa_te):
        em = AstroEdgeModel().fit(Xa_tr, ytr)
        return roc_auc_score(yte, em.transform(Xa_te)["ASTRO_EDGE"])

    actual = fit_score(Xtr, Xte)

    null = []
    for _ in range(n):
        perm = RNG.permutation(len(Xtr))
        Xs = Xtr.copy()
        Xs[astro_cols] = Xtr[astro_cols].to_numpy()[perm]
        try:
            null.append(fit_score(Xs, Xte))
        except Exception:
            continue
    null = np.array(null)
    p_value = float((null >= actual).mean()) if len(null) else float("nan")
    return {
        "actual_auc": float(actual),
        "null_mean": float(null.mean()) if len(null) else None,
        "null_std": float(null.std()) if len(null) else None,
        "null_p05_p95": [float(np.percentile(null, 5)), float(np.percentile(null, 95))] if len(null) else None,
        "p_value": p_value,
        "n_permutations": int(len(null)),
        "significant_at_005": bool(p_value < 0.05) if np.isfinite(p_value) else False,
    }


def crisis_auc(model: HybridModel, windows: list[dict]) -> dict:
    oof = model.oof
    out = {}
    for w in windows:
        seg = oof.loc[str(w["start"]): str(w["end"])]
        if len(seg) < 20 or seg["y"].nunique() < 2:
            continue
        out[w["key"]] = {
            "label": w["label"],
            "n": int(len(seg)),
            "base_rate": float(seg["y"].mean()),
            "technical": float(roc_auc_score(seg["y"], seg["p_tech"])),
            "astro": float(roc_auc_score(seg["y"], seg["p_astro"])),
            "hybrid": float(roc_auc_score(seg["y"], seg["p_hybrid"])),
        }
        out[w["key"]]["hybrid_minus_technical"] = (
            out[w["key"]]["hybrid"] - out[w["key"]]["technical"]
        )
    return out


def stratification(data: ds.Dataset) -> dict:
    """Out-of-sample conditional win rates per celestial state family."""
    X, y = data.xy("y_vriddhi")
    cut = int(len(X) * 0.65)
    em = AstroEdgeModel().fit(X.iloc[:cut], y.iloc[:cut])

    nak_names = {n["index"]: n["display"] for n in nakshatra_cfg()["nakshatras"]}
    tara_names = {t["index"]: t["name"] for t in nakshatra_cfg()["tarabala"]}

    # Evaluate the fitted edges on the HELD-OUT block only.
    Xte, yte = X.iloc[cut:], y.iloc[cut:]
    out: dict = {"base_rate_test": float(yte.mean()), "families": {}}

    for family, labels in (
        ("nakshatra_index", nak_names),
        ("nat_tarabala", tara_names),
        ("tithi_index", None),
        ("dasha_md_idx", None),
        ("karana_index", None),
    ):
        if family not in Xte.columns:
            continue
        key = Xte[family].fillna(-999).astype(float).round(0)
        grouped = yte.groupby(key)
        rows = []
        for state, vals in grouped:
            if len(vals) < 20:
                continue
            rows.append(
                {
                    "state": int(state),
                    "name": (labels or {}).get(int(state), str(int(state))),
                    "n": int(len(vals)),
                    "win_rate": float(vals.mean()),
                    "vs_base": float(vals.mean() - yte.mean()),
                }
            )
        rows.sort(key=lambda r: -r["win_rate"])
        out["families"][family] = rows
    out["family_strength"] = {k: float(v) for k, v in em.family_strength().items()}
    return out


# --------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=60)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    windows = tickers_cfg()["crisis_windows"]
    report: dict = {}

    for key in TICKER_KEYS:
        print(f"\n=== {key} ===")
        data = ds.build(key, refresh=args.refresh)
        model = HybridModel()
        model.fit_walk_forward(data)

        ab = ablation(model)
        print(
            f"  ablation   technical {ab['M3_technical']:.4f}  astro {ab['M4_astro_only']:.4f}  "
            f"hybrid {ab['M5_hybrid']:.4f}  (delta {ab['delta_hybrid_vs_technical']:+.4f})"
        )

        bs = bootstrap_auc(model, args.bootstrap)
        print(
            f"  bootstrap  delta {bs['delta_mean']:+.4f} "
            f"CI95 [{bs['delta_ci95'][0]:+.4f}, {bs['delta_ci95'][1]:+.4f}]  "
            f"P(delta>0) {bs['delta_positive_fraction']:.2f}"
        )

        perm = permutation_test(data, args.permutations)
        print(
            f"  permutation actual {perm['actual_auc']:.4f}  null {perm['null_mean']:.4f}"
            f" +/- {perm['null_std']:.4f}  p={perm['p_value']:.4f}"
            f"  {'SIGNIFICANT' if perm['significant_at_005'] else 'not significant'}"
        )

        cr = crisis_auc(model, windows)
        for k, v in cr.items():
            print(
                f"  crisis {k:10s} tech {v['technical']:.3f}  astro {v['astro']:.3f}  "
                f"hybrid {v['hybrid']:.3f}  ({v['hybrid_minus_technical']:+.3f})"
            )

        strat = stratification(data)
        report[key] = {
            "ablation": ab,
            "bootstrap": bs,
            "permutation": perm,
            "crisis": cr,
            "stratification": strat,
        }

    out = ARTIFACT_DIR / "validation.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
