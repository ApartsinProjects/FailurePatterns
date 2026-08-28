"""Cross-cut diagnostic: itemset lift vs sequence lift, per horizon.

Emits:
- results/figures/azure_itemset_vs_sequence_lift.png
- results/tables/azure_top_patterns.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

items = pd.read_parquet(ROOT / "results" / "patterns" / "azure_itemsets.parquet")
seqs = pd.read_parquet(ROOT / "results" / "patterns" / "azure_sequences.parquet")

ORDER = ["1h", "6h", "24h", "last5", "last10"]

# ---- Figure: scatter of sequence-lift vs itemset-lift (per horizon) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, h in zip(axes, ["24h", "last5", "last10"]):
    sub = seqs[seqs["horizon"] == h].dropna(subset=["lift_failure", "itemset_lift_failure"])
    if sub.empty:
        ax.set_title(f"{h}: no data")
        continue
    ax.scatter(
        sub["itemset_lift_failure"], sub["lift_failure"],
        s=25, alpha=0.55, color="#2c3e50",
    )
    lim = max(sub["lift_failure"].max(), sub["itemset_lift_failure"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#c0392b", linestyle="--", linewidth=1,
            label="sequence-lift = itemset-lift")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("itemset lift (same items, order-blind)")
    ax.set_ylabel("sequence lift (order-preserving)")
    ax.set_title(f"horizon: {h}   (n_seq={len(sub)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle(
    "Azure PdM: does preserving event order add signal beyond the itemset?",
    y=1.05,
)
fig.savefig(FIG / "azure_itemset_vs_sequence_lift.png", dpi=140, bbox_inches="tight")
print(f"Wrote {FIG / 'azure_itemset_vs_sequence_lift.png'}")

# ---- Table: top patterns per horizon ----
lines: list[str] = []
lines.append("# Azure PdM: top mined patterns\n")
lines.append("Both mining passes at ``min_support = 0.05``. Sequences with "
             "``survives_shuffle_null = True`` beat the within-window "
             "order-permutation null at that horizon.\n")

lines.append("## Top itemsets (Phase 3, FP-Growth)\n")
for h in ORDER:
    sub = items[items["horizon"] == h]
    if sub.empty:
        lines.append(f"### {h}\n\n_no patterns above min_support_\n")
        continue
    top = sub.sort_values("lift_failure", ascending=False).head(6)
    lines.append(f"### {h} ({len(sub)} patterns)\n")
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "itemset": " + ".join(r["itemset"]),
            "supp_fail": round(r["support_failure"], 3),
            "supp_ctrl": round(r["support_control"], 3),
            "lift": round(r["lift_failure"], 2),
            "RR": round(r["relative_risk"], 2),
            "P(fail|patt)": round(r["p_fail_given_pattern"], 3),
            "survives": bool(r["survives_permutation_null"]),
        })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

lines.append("## Top sequences (Phase 4, PrefixSpan)\n")
for h in ORDER:
    sub = seqs[seqs["horizon"] == h]
    if sub.empty:
        lines.append(f"### {h}\n\n_no patterns above min_support_\n")
        continue
    top = sub.sort_values("lift_failure", ascending=False).head(6)
    lines.append(f"### {h} ({len(sub)} patterns)\n")
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "sequence": " -> ".join(r["sequence"]),
            "supp_fail": round(r["support_failure"], 3),
            "seq_lift": round(r["lift_failure"], 2),
            "iset_lift": round(r["itemset_lift_failure"], 2),
            "order_gain": round(r["order_gain"], 2),
            "P(fail|patt)": round(r["p_fail_given_pattern"], 3),
            "survives_shuf": bool(r["survives_shuffle_null"]),
        })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")

lines.append("## Where order matters most (top 10 by order_gain)\n")
top_gain = seqs.sort_values("order_gain", ascending=False).head(10)
gain_rows = [{
    "horizon": r["horizon"],
    "sequence": " -> ".join(r["sequence"]),
    "seq_lift": round(r["lift_failure"], 2),
    "iset_lift": round(r["itemset_lift_failure"], 2),
    "order_gain": round(r["order_gain"], 2),
    "supp_fail": round(r["support_failure"], 3),
} for _, r in top_gain.iterrows()]
lines.append(pd.DataFrame(gain_rows).to_markdown(index=False))
lines.append("")

(TAB / "azure_top_patterns.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {TAB / 'azure_top_patterns.md'}")
