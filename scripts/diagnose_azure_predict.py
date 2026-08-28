"""Cross-cut diagnostic figure + markdown for Phase 6 predictive eval.

Emits:
- results/figures/azure_predictive_comparison.png
- results/tables/azure_predictive.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "results" / "tables" / "azure_predictive.parquet")

HORIZONS = ["24h", "last5", "last10"]
FEAT_SETS = ["event_count", "itemsets_only", "sequences_only", "combined"]
COLORS = {
    "event_count":    "#95a5a6",
    "itemsets_only":  "#3498db",
    "sequences_only": "#e67e22",
    "combined":       "#27ae60",
}

# ---- Figure: grouped bars, AUROC and AUPRC per (horizon, feature_set) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, metric, title in [
    (axes[0], "auroc", "AUROC"),
    (axes[1], "auprc", "AUPRC"),
]:
    x = np.arange(len(HORIZONS))
    width = 0.2
    for i, fs in enumerate(FEAT_SETS):
        vals = []
        for h in HORIZONS:
            sub = df[(df["horizon"] == h) & (df["feature_set"] == fs)]
            vals.append(sub[metric].iloc[0] if len(sub) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               label=fs, color=COLORS[fs])
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    if metric == "auroc":
        ax.legend(loc="lower right", fontsize=8)
fig.suptitle(
    "Azure PdM Phase 6: feature-set head-to-head on temporally-held-out test set\n"
    "(train: anchor < 2015-09-01; test: anchor >= 2015-09-01)",
    y=1.10,
)
fig.savefig(FIG / "azure_predictive_comparison.png", dpi=140, bbox_inches="tight")
print(f"Wrote {FIG / 'azure_predictive_comparison.png'}")

# ---- Markdown summary ----
lines: list[str] = []
lines.append("# Azure PdM Phase 6: predictive evaluation\n")
lines.append("Temporal split at 2015-09-01. Patterns mined on train only.\n")
lines.append("Logistic regression on each feature set.\n\n")

tbl = df[[
    "horizon", "feature_set", "n_features",
    "auroc", "auprc", "f1_at_0.5", "precision_at_0.5", "recall_at_0.5",
]].round(3)
lines.append(tbl.to_markdown(index=False))
lines.append("")

lines.append("## Reading the numbers\n")
lines.append(
    "- **24h horizon:** event-count alone already reaches AUROC 0.97 "
    "(mean 1.58 events in failure windows vs 0.077 in controls). "
    "Itemsets push to 0.996. **No sequence survived the shuffle-null "
    "at 24h**, so sequences_only has n_features = 0 and combined "
    "equals itemsets_only. At this horizon order does not help; the "
    "itemset already captures everything.\n"
    "- **last5 horizon:** event-count is chance because n_events = 5 "
    "for both classes. Itemsets_only reaches AUROC 0.75. Sequences_only "
    "at 6 features is high-precision (0.83) but low-recall (0.36). "
    "Combined: **AUROC 0.81, AUPRC 0.72 - +5.6 AUROC and +15.7 AUPRC "
    "points above itemsets_only.**\n"
    "- **last10 horizon:** same shape as last5 but with more features. "
    "Combined reaches AUROC 0.70, +5.3 AUROC over itemsets_only.\n\n"
    "**Answer to the paper's Experiment 4 question:** temporal order in "
    "mined sequences contributes real predictive information beyond "
    "the itemset representation, but only when the window definition is "
    "rich enough for order to be a real degree of freedom. On Azure PdM "
    "that is the count-based (last-K events) horizons, not the short "
    "time horizons where 24h windows contain at most 1-2 events.\n"
)

(TAB / "azure_predictive.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {TAB / 'azure_predictive.md'}")
