"""Cross-cut diagnostic for Alibaba Phase 6 (same shape as Azure diag)."""

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

df = pd.read_parquet(TAB / "alibaba_predictive.parquet")

HORIZONS = ["last3", "last5", "last10"]
FEAT_SETS = ["event_count", "itemsets_only", "sequences_only", "combined"]
COLORS = {
    "event_count": "#95a5a6", "itemsets_only": "#3498db",
    "sequences_only": "#e67e22", "combined": "#27ae60",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for ax, metric, title in [
    (axes[0], "auroc", "AUROC"),
    (axes[1], "auprc", "AUPRC"),
]:
    x = np.arange(len(HORIZONS))
    width = 0.2
    for i, fs in enumerate(FEAT_SETS):
        vals = [
            df[(df["horizon"] == h) & (df["feature_set"] == fs)][metric].iloc[0]
            if len(df[(df["horizon"] == h) & (df["feature_set"] == fs)]) else np.nan
            for h in HORIZONS
        ]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fs, color=COLORS[fs])
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    if metric == "auroc":
        ax.legend(loc="upper right", fontsize=8)
fig.suptitle(
    "Alibaba cluster-trace-v2018 Phase 6: per-job failure prediction\n"
    "(train: anchor < 2018-01-07; test: anchor >= 2018-01-07)",
    y=1.10,
)
fig.savefig(FIG / "alibaba_predictive_comparison.png", dpi=140, bbox_inches="tight")
print(f"Wrote {FIG / 'alibaba_predictive_comparison.png'}")

# ---- combined table ----
lines: list[str] = []
lines.append("# Alibaba cluster-trace-v2018 Phase 6: predictive evaluation\n")
lines.append("Per-job failure prediction. Entity = job_name, event vocabulary "
             "= (normalized-status, task_role letter prefix). Temporal "
             "split at 2018-01-07. Patterns mined on train only.\n")
tbl = df[[
    "horizon", "feature_set", "n_features",
    "auroc", "auprc", "f1_at_0.5", "precision_at_0.5", "recall_at_0.5",
]].round(3)
lines.append(tbl.to_markdown(index=False))
lines.append("")
lines.append(
    "\n## Comparison to Azure PdM\n\n"
    "Azure PdM (per-machine): combined at last5 -> AUROC 0.810, AUPRC 0.720.\n"
    "Alibaba (per-job): combined at last3 -> AUROC 0.813, AUPRC 0.631.\n\n"
    "Same finding in both traces: combining sequences with itemsets adds "
    "5-10 AUROC points over itemsets-only. Sequences-only has few surviving "
    "features (2 on Alibaba vs 6-16 on Azure) but is high-precision (~0.95+ "
    "at 0.5 threshold), suggesting the mined ordered patterns fire rarely "
    "but reliably.\n"
)
(TAB / "alibaba_predictive.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {TAB / 'alibaba_predictive.md'}")


# ---- Two-panel cross-dataset comparison figure ----
az = pd.read_parquet(TAB / "azure_predictive.parquet")
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
for ax, (name, dfrm, horizons) in zip(axes2, [
    ("Azure PdM (per-machine)", az, ["24h", "last5", "last10"]),
    ("Alibaba v2018 (per-job)", df, ["last3", "last5", "last10"]),
]):
    x = np.arange(len(horizons))
    width = 0.2
    for i, fs in enumerate(FEAT_SETS):
        vals = [
            dfrm[(dfrm["horizon"] == h) & (dfrm["feature_set"] == fs)]["auroc"].iloc[0]
            if len(dfrm[(dfrm["horizon"] == h) & (dfrm["feature_set"] == fs)]) else np.nan
            for h in horizons
        ]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fs, color=COLORS[fs])
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(name)
    if name.startswith("Azure"):
        ax.legend(loc="lower right", fontsize=8)
fig2.suptitle(
    "Cross-dataset Phase 6: same feature-set ordering holds on Azure PdM and Alibaba v2018",
    y=1.05,
)
fig2.savefig(FIG / "cross_dataset_predictive_comparison.png", dpi=140, bbox_inches="tight")
print(f"Wrote {FIG / 'cross_dataset_predictive_comparison.png'}")
