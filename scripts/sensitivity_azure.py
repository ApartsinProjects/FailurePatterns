"""Phase 7 (scoped): min_support sensitivity sweep on Azure predictive eval.

Re-runs the Phase 6 head-to-head at min_support in {0.02, 0.05, 0.10, 0.15}
across the informative horizons (24h, last5, last10) and plots
AUROC / AUPRC vs min_support per feature set per horizon.

Emits:
- results/tables/azure_sensitivity_min_support.parquet
- results/figures/azure_sensitivity_min_support.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.predict import evaluate

SUPPORTS = [0.02, 0.05, 0.10, 0.15]
HORIZONS = ("24h", "last5", "last10")
CUTOFF = pd.Timestamp("2015-09-01")


def main() -> int:
    windows = pd.read_parquet(ROOT / "data" / "processed" / "azure_windows.parquet")

    all_rows: list[pd.DataFrame] = []
    for s in SUPPORTS:
        print(f"--- min_support = {s} ---", flush=True)
        results, _ = evaluate(windows, horizons=HORIZONS, cutoff=CUTOFF, min_support=s)
        results = results.copy()
        results["min_support"] = s
        all_rows.append(results)
    combined = pd.concat(all_rows, ignore_index=True)

    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_dir / "azure_sensitivity_min_support.parquet", index=False)
    print(f"Wrote {out_dir / 'azure_sensitivity_min_support.parquet'}")

    # ---- Figure: 3 horizons x 2 metrics grid ----
    feat_sets = ["event_count", "itemsets_only", "sequences_only", "combined"]
    colors = {
        "event_count": "#95a5a6", "itemsets_only": "#3498db",
        "sequences_only": "#e67e22", "combined": "#27ae60",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, metric in enumerate(("auroc", "auprc")):
        for col, h in enumerate(HORIZONS):
            ax = axes[row, col]
            for fs in feat_sets:
                sub = combined[
                    (combined["horizon"] == h) & (combined["feature_set"] == fs)
                ].sort_values("min_support")
                ax.plot(
                    sub["min_support"], sub[metric],
                    marker="o", color=colors[fs], label=fs, linewidth=2,
                )
            ax.set_xlabel("min_support")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{h}   {metric.upper()}")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Azure PdM Phase 7: sensitivity to min_support (temporal holdout)",
        y=1.03,
    )
    fig_dir = ROOT / "results" / "figures"
    fig.savefig(fig_dir / "azure_sensitivity_min_support.png", dpi=140, bbox_inches="tight")
    print(f"Wrote {fig_dir / 'azure_sensitivity_min_support.png'}")

    # Print a compact table.
    piv = combined.pivot_table(
        index=["horizon", "feature_set"], columns="min_support",
        values="auroc",
    ).round(3)
    print("\nAUROC pivot (rows: horizon x feature_set, cols: min_support):")
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
