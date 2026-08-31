"""Figure 7: presence / multiplicity / order decomposition per trace.

Reads results/patterns/representation_experiment.json and plots, for each
trace, the three incremental AUROC effects with entity-bootstrap 95% CIs:
  presence effect       = AUROC(presence) - AUROC(event_count)
  multiplicity increment= AUROC(counts)   - AUROC(presence)
  order increment       = AUROC(bigram)   - AUROC(counts)
A bar whose CI excludes zero is drawn solid; one whose CI includes zero is
drawn hatched (not distinguishable from no effect).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

INK = "#1a1a1a"
COL = {"presence": "#6b7280", "multiplicity": "#2563eb", "order": "#b45309"}


def main() -> Path:
    d = json.load(open(ROOT / "results/patterns/representation_experiment.json"))
    traces = [r["trace"] for r in d]
    comps = [("presence", "presence_effect"), ("multiplicity", "multiplicity_increment"),
             ("order", "order_increment")]

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    n = len(traces)
    w = 0.26
    x = np.arange(n)
    for k, (label, key) in enumerate(comps):
        deltas = [r["decomposition"][key]["delta"] for r in d]
        los = [r["decomposition"][key]["ci95"][0] for r in d]
        his = [r["decomposition"][key]["ci95"][1] for r in d]
        excl0 = [(lo > 0) or (hi < 0) for lo, hi in zip(los, his)]
        pos = x + (k - 1) * w
        for xi, dv, lo, hi, sig in zip(pos, deltas, los, his, excl0):
            ax.bar(xi, dv, w, color=COL[label], alpha=1.0 if sig else 0.35,
                   hatch=None if sig else "///", edgecolor=COL[label], linewidth=1.0,
                   label=label if xi == pos[0] else None)
            ax.plot([xi, xi], [lo, hi], color=INK, linewidth=1.0, zorder=3)
            ax.plot([xi - w * 0.2, xi + w * 0.2], [lo, lo], color=INK, linewidth=1.0, zorder=3)
            ax.plot([xi - w * 0.2, xi + w * 0.2], [hi, hi], color=INK, linewidth=1.0, zorder=3)

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(traces, fontsize=10)
    ax.set_ylabel("incremental held-out AUROC", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # legend with the three components
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in ("presence", "multiplicity", "order")]
    ax.legend(handles, ["presence (set)", "multiplicity (counts)", "order (bigram)"],
              loc="upper right", fontsize=8.5, frameon=False)
    ax.annotate("solid = CI excludes 0; hatched = indistinguishable from 0",
                (0.0, 1.02), xycoords="axes fraction", fontsize=8, color=INK)
    out = FIG / "decomposition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
