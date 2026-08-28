"""Cross-dataset AUROC comparison across all 4 traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

az = pd.read_parquet(TAB / "azure_predictive.parquet")
al = pd.read_parquet(TAB / "alibaba_predictive.parquet")
bg = pd.read_parquet(TAB / "bgl_predictive.parquet")
sc = pd.read_parquet(TAB / "scania_predictive.parquet")

FEAT = ["event_count", "itemsets_only", "sequences_only", "combined"]
COLORS = {
    "event_count":    "#95a5a6",
    "itemsets_only":  "#3498db",
    "sequences_only": "#e67e22",
    "combined":       "#27ae60",
}

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
for ax, (name, df, horizons) in zip(axes, [
    ("Azure PdM (per-machine)",   az, ["24h", "last5", "last10"]),
    ("Alibaba v2018 (per-job)",   al, ["last3", "last5", "last10"]),
    ("BGL (per-rack)",            bg, ["last5", "last10", "last20"]),
    ("SCANIA (per-vehicle)",      sc, ["last5", "last10", "last20"]),
]):
    x = np.arange(len(horizons))
    width = 0.2
    for i, fs in enumerate(FEAT):
        vals = [
            df[(df["horizon"] == h) & (df["feature_set"] == fs)]["auroc"].iloc[0]
            if len(df[(df["horizon"] == h) & (df["feature_set"] == fs)]) else np.nan
            for h in horizons
        ]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fs, color=COLORS[fs])
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(name, fontsize=10.5)
    if name.startswith("Azure"):
        ax.legend(loc="lower right", fontsize=7.5)

fig.suptitle(
    "Cross-dataset Phase 6 predictive evaluation: two wins (Azure, Alibaba) "
    "and two boundary conditions (BGL self-triggering alerts; SCANIA derived tokens)",
    y=1.05, fontsize=11,
)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "four_dataset_predictive_comparison.png", dpi=140, bbox_inches="tight")
print(f"Wrote {FIG / 'four_dataset_predictive_comparison.png'}")
