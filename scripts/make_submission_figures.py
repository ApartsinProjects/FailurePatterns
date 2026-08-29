"""Generate the two submission figures the reviewer requested:

  Figure 5: pipeline schematic (windows -> discovery/inference split ->
            mining -> inference-half test -> FDR -> catalog, with the
            SCANIA risk-set branch).
  Figure 6: a real Kelmarsh generator-fan cascade timeline (alarm codes
            on a time axis leading to a Forced outage).

Saves PNGs to results/figures/. Deterministic (no RNG).
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
BLUE = "#2563eb"
GREEN = "#15803d"
AMBER = "#b45309"
GREY = "#6b7280"
LIGHT = "#eef2ff"


# ----------------------------------------------------------------------
# Figure 5: pipeline schematic
# ----------------------------------------------------------------------
def pipeline_schematic() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.8); ax.axis("off")

    def box(x, y, w, h, text, fc=LIGHT, ec=BLUE, tc=INK, fs=9.5, bold=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                           linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, color=GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=13, linewidth=1.3, color=color,
                     shrinkA=2, shrinkB=2))

    # Main row
    box(0.2, 3.6, 1.9, 1.0, "Operational\nevent log", fc="#ffffff", ec=INK, bold=True)
    box(2.5, 3.6, 1.9, 1.0, "Pre-failure\nwindows +\nmatched controls")
    box(4.8, 3.6, 2.1, 1.0, "Entity-disjoint\ndiscovery / inference\nsplit (50 / 50)", ec=GREEN)
    box(7.3, 3.6, 2.5, 1.0, "Mine on discovery half\n(FP-Growth, PrefixSpan;\nclosed via CloSpan)")
    arrow(2.1, 4.1, 2.5, 4.1); arrow(4.4, 4.1, 4.8, 4.1)
    arrow(6.9, 4.1, 7.3, 4.1)

    # Down to inference row
    arrow(8.55, 3.6, 8.55, 2.55)
    box(6.9, 1.5, 2.9, 1.0, "Score on inference half\n(exact hypergeometric;\nBH / BY FDR)", ec=GREEN)
    box(3.7, 1.5, 2.7, 1.0, "Count-preserving\norder null\n(order vs multiplicity)", ec=AMBER)
    box(0.2, 1.5, 3.0, 1.0, "Validated signature\ncatalog\n(evidence + use)", fc="#ecfdf5", ec=GREEN, bold=True)
    arrow(6.9, 2.0, 6.4, 2.0); arrow(3.7, 2.0, 3.2, 2.0)

    # Risk-set branch (SCANIA)
    box(4.8, 0.1, 2.1, 0.9, "Risk-set matched\nsampling (SCANIA)", fc="#fff7ed", ec=AMBER)
    box(7.3, 0.1, 2.5, 0.9, "Matched conditional\nlogistic (hazard ratio)", ec=AMBER)
    arrow(5.85, 1.5, 5.85, 1.0, color=AMBER)
    arrow(6.9, 0.55, 7.3, 0.55, color=AMBER)
    ax.text(4.75, 0.55, "right-\ncensored", ha="right", va="center", fontsize=8, color=AMBER, style="italic")

    out = FIG / "pipeline_schematic.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 6: real Kelmarsh generator-fan cascade timeline
# ----------------------------------------------------------------------
FAN = {"2550", "2650", "2655"}
FAMILY_LABEL = {
    "2550": "2550 gen-fan 1", "2650": "2650 gen-fan 2", "2655": "2655 gen-fan 3",
    "2125": "2125 brake timeout", "5720": "5720 brake accumulator",
    "2000": "2000 converter", "20": "20 stop", "25": "25 stop",
    "6410": "6410 info", "1552": "1552 info", "1555": "1555 info", "1920": "1920 warn",
}
TYPE_COLOR = {"system_warning": AMBER, "system_stop": "#b91c1c",
              "system_info": GREY, "system_comm": "#0891b2",
              "terminal_failure": "#7c3aed"}


def _dedupe(pre: pd.DataFrame, gap_min: float = 4.0) -> pd.DataFrame:
    """Collapse consecutive events with the same code closer than gap_min."""
    pre = pre.sort_values("timestamp")
    if pre.empty:
        return pre
    keep = []
    last = {}
    for _, row in pre.iterrows():
        c = str(row["event_subtype"])
        t = row["timestamp"]
        if c in last and (t - last[c]).total_seconds() / 60.0 < gap_min:
            continue
        last[c] = t
        keep.append(row)
    return pd.DataFrame(keep, columns=pre.columns)


def cascade_timeline() -> Path | None:
    e = pd.read_parquet(ROOT / "data/processed/kelmarsh_events.parquet")
    fo = e[e["event_type"] == "terminal_failure"]
    non = e[e["event_type"] != "terminal_failure"]
    fam_warn = FAN | {"2125", "5720", "2000", "2650"}
    # Choose the fan-overload forced outage with the best-spread same-family
    # warning precursors strictly before the anchor.
    best = None
    best_score = -1.0
    for _, r in fo[fo["event_subtype"].isin(FAN)].iterrows():
        pre = non[(non["entity_id"] == r["entity_id"]) &
                  (non["timestamp"] < r["timestamp"] - pd.Timedelta(minutes=2)) &
                  (non["timestamp"] >= r["timestamp"] - pd.Timedelta(hours=12))]
        pre = _dedupe(pre)
        if pre.empty:
            continue
        warn_fam = pre[(pre["event_type"] == "system_warning") &
                       (pre["event_subtype"].isin(fam_warn))]
        ndistinct = pre["event_subtype"].nunique()
        if len(warn_fam) >= 2 and ndistinct >= 3 and len(pre) <= 10:
            span = (pre["timestamp"].max() - pre["timestamp"].min()).total_seconds() / 3600.0
            score = len(warn_fam) + 0.3 * ndistinct + 0.2 * span
            if score > best_score:
                best_score = score; best = (r, pre)
    if best is None:
        return None
    r, pre = best
    t0 = r["timestamp"]
    pre = pre.sort_values("timestamp")

    # Group simultaneous events (same minute) into ONE stem: the generator-fan
    # codes 2550/2650/2655 fire together, which is the co-occurrence the itemset
    # mining captures, so they belong on one labelled marker.
    SHORT = {"2550": "2550", "2650": "2650", "2655": "2655", "2125": "brake 2125",
             "5720": "brake 5720", "2000": "converter 2000", "20": "stop 20",
             "25": "stop 25", "6410": "info 6410"}
    groups = []  # (hours_before, dominant_type, label)
    for tstamp, g in pre.groupby(pre["timestamp"].dt.floor("min")):
        codes = list(dict.fromkeys(str(c) for c in g["event_subtype"]))
        fan = [c for c in codes if c in FAN]
        other = [c for c in codes if c not in FAN]
        if fan:
            label = "/".join(fan) + " gen-fan"
            dom = "system_warning"
        else:
            label = ", ".join(SHORT.get(c, c) for c in other[:2])
            dom = g.iloc[0]["event_type"]
        hours = -(t0 - tstamp).total_seconds() / 3600.0
        groups.append((hours, dom, label, "fan" if fan else "other"))
    # Keep the fan-family group plus up to 4 others, sorted in time.
    fan_groups = [x for x in groups if x[3] == "fan"]
    other_groups = [x for x in groups if x[3] != "fan"]
    groups = sorted(fan_groups + other_groups[:4], key=lambda z: z[0])

    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.axhline(0, color=INK, linewidth=1.0, zorder=1)
    seen = set()
    heights = [1.0, 1.7, 1.0, 1.7, 1.0, 1.7]
    for i, (x, dom, label, kind) in enumerate(groups):
        c = TYPE_COLOR.get(dom, GREY)
        h = heights[i % len(heights)]
        lab = dom.replace("system_", "") if dom not in seen else None
        seen.add(dom)
        lw = 2.2 if kind == "fan" else 1.4
        ax.plot([x, x], [0, h], color=c, linewidth=lw, zorder=2)
        ax.scatter([x], [h], color=c, s=(48 if kind == "fan" else 30), zorder=3, label=lab)
        ax.annotate(label, (x, h + 0.08), rotation=0, ha="center", va="bottom",
                    fontsize=8.2, color=c, weight="bold" if kind == "fan" else "normal")
    ax.scatter([0], [0], marker="X", s=170, color=TYPE_COLOR["terminal_failure"], zorder=4)
    xs = [g[0] for g in groups]
    ax.annotate(f"Forced outage\ncode {r['event_subtype']}", (0, -0.12), ha="center", va="top",
                fontsize=9, color=TYPE_COLOR["terminal_failure"], weight="bold")

    ax.set_xlabel("hours before forced outage", fontsize=9.5)
    ax.set_xlim(min(xs) - 0.4, 0.6)
    ax.set_ylim(-1.0, 2.8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    out = FIG / "kelmarsh_cascade_timeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    p1 = pipeline_schematic()
    print("wrote", p1)
    p2 = cascade_timeline()
    if p2:
        print("wrote", p2)
    else:
        print("WARNING: no suitable cascade example found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
