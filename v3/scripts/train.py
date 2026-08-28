"""Fit the gated hybrid model per ticker and cache it for the app.

Runs purged walk-forward CV first (which produces the out-of-fold predictions used for
isotonic calibration and every AUC reported anywhere), then refits both blocks on the
full history for live inference.
"""

from __future__ import annotations

import argparse
import json

import joblib

from chandraquant.config import ARTIFACT_DIR, TICKER_KEYS
from chandraquant.features import dataset as ds
from chandraquant.inference import model_path
from chandraquant.models.hybrid import HybridModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ALL")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    keys = TICKER_KEYS if args.ticker.upper() == "ALL" else [args.ticker.upper()]
    report = {}

    for key in keys:
        data = ds.build(key, refresh=args.refresh)
        model = HybridModel()
        model.fit_walk_forward(data)
        model.fit_full(data)

        path = model_path(key)
        joblib.dump(model, path)

        auc = model.auc_table()
        overall = auc[auc["fold"] == "ALL"].iloc[0]
        report[key] = {
            "rows": int(len(model.oof)),
            "base_rate": float(overall["base_rate"]),
            "auc_technical": float(overall["p_tech"]),
            "auc_astro": float(overall["p_astro"]),
            "auc_hybrid": float(overall["p_hybrid"]),
            "auc_calibrated": float(overall["p_calibrated"]),
            "model_path": str(path),
        }
        print(
            f"{key:10s} rows {report[key]['rows']:5d}  "
            f"tech {report[key]['auc_technical']:.4f}  "
            f"astro {report[key]['auc_astro']:.4f}  "
            f"hybrid {report[key]['auc_hybrid']:.4f}  -> {path.name}"
        )

        top = model.feature_importance("astro", top=8)
        print(f"{'':10s} top astro features: {', '.join(top.index[:8])}")

    out = ARTIFACT_DIR / "training.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
