"""Are the wind-farm signatures trivial co-located events, or genuine
multi-step degradation chains?

Two diagnostics on the raw event stream (not the last-K windows):

(1) CO-LOCATION test. For each forced outage, take its 24h pre-anchor
    window and, for the generator-fan family that heads the catalog,
    measure the wall-clock span between the events of the mined pattern.
    If the span is ~0 the pattern is co-located (one instant); if it is
    minutes/hours it is a genuine trajectory.

(2) LONG-CHAIN extraction. Collapse each 24h pre-outage window to its
    ordered sequence of DISTINCT non-terminal codes (consecutive repeats
    of the same code merged, so co-located bursts count once), then mine
    frequent ordered chains of length >= 3 that recur across outages and
    span real time. Report the most frequent long chains, their support,
    their median total duration, and their control support (same 24h
    windows anchored on clean regions) to show they are not generic.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = pd.Timedelta(hours=24)
FAN = {"2550", "2650", "2655"}


def _distinct_code_seq(win: pd.DataFrame):
    """Ordered list of (code, timestamp) with consecutive same-code merged."""
    win = win.sort_values("timestamp")
    out = []
    last = None
    for _, r in win.iterrows():
        c = f"{r['event_type']}:{r['event_subtype']}"
        if c == last:
            continue
        out.append((c, r["timestamp"]))
        last = c
    return out


def analyse(name: str, path: Path) -> dict:
    ev = pd.read_parquet(path)
    fo = ev[ev["event_type"] == "terminal_failure"]
    non = ev[ev["event_type"] != "terminal_failure"]

    # ---- (1) co-location span of the fan family within pre-outage windows
    fan_spans = []
    chains = Counter()
    chain_durations = defaultdict(list)
    n_windows = 0
    for _, o in fo.iterrows():
        w = non[(non["entity_id"] == o["entity_id"]) &
                (non["timestamp"] < o["timestamp"]) &
                (non["timestamp"] >= o["timestamp"] - H)]
        if w.empty:
            continue
        n_windows += 1
        fan = w[w["event_subtype"].astype(str).isin(FAN)]
        if len(fan) >= 2:
            span = (fan["timestamp"].max() - fan["timestamp"].min()).total_seconds() / 60.0
            fan_spans.append(span)
        # ---- (2) distinct-code ordered chain
        seq = _distinct_code_seq(w)
        codes = [c for c, _ in seq]
        # count all length-3 ordered sub-chains of DISTINCT codes
        for i in range(len(codes) - 2):
            tri = (codes[i], codes[i + 1], codes[i + 2])
            if len(set(tri)) == 3:
                chains[tri] += 1
                dur = (seq[i + 2][1] - seq[i][1]).total_seconds() / 60.0
                chain_durations[tri].append(dur)

    # control support for the top chains (clean-region 24h windows)
    rng = np.random.default_rng(20260828)
    ctrl_counts = Counter()
    n_ctrl = 0
    for ent, g in non.groupby("entity_id"):
        g = g.sort_values("timestamp")
        foent = fo[fo["entity_id"] == ent]["timestamp"].to_numpy()
        ts = g["timestamp"].to_numpy()
        if len(ts) < 5:
            continue
        for anchor in rng.choice(ts, size=min(40, len(ts)), replace=False):
            # skip anchors within 24h of a real outage
            if len(foent) and ((foent >= anchor) & (foent <= anchor + np.timedelta64(24, "h"))).any():
                continue
            w = g[(g["timestamp"] < anchor) & (g["timestamp"] >= anchor - H)]
            if w.empty:
                continue
            n_ctrl += 1
            codes = [c for c, _ in _distinct_code_seq(w)]
            seen = set()
            for i in range(len(codes) - 2):
                tri = (codes[i], codes[i + 1], codes[i + 2])
                if len(set(tri)) == 3:
                    seen.add(tri)
            for tri in seen:
                ctrl_counts[tri] += 1

    top = []
    for tri, cnt in chains.most_common(12):
        durs = chain_durations[tri]
        top.append({
            "chain": " -> ".join(tri),
            "n_distinct_codes": 3,
            "support_outage_windows": cnt,
            "support_frac_outage": round(cnt / max(n_windows, 1), 3),
            "support_control_windows": ctrl_counts.get(tri, 0),
            "support_frac_control": round(ctrl_counts.get(tri, 0) / max(n_ctrl, 1), 3),
            "median_span_min": round(float(np.median(durs)), 1) if durs else None,
        })

    return {
        "farm": name,
        "n_outage_windows_24h": n_windows,
        "n_control_windows_24h": n_ctrl,
        "fan_family_colocation_median_span_min": round(float(np.median(fan_spans)), 2) if fan_spans else None,
        "fan_family_span_frac_under_1min": round(float((np.array(fan_spans) < 1).mean()), 3) if fan_spans else None,
        "top_distinct_3chains": top,
    }


def main() -> int:
    out = {}
    for name, path in [("Kelmarsh", ROOT / "data/processed/kelmarsh_events.parquet"),
                       ("Penmanshiel", ROOT / "data/processed/penmanshiel_events.parquet")]:
        out[name] = analyse(name, path)
        r = out[name]
        print(f"\n=== {name}: {r['n_outage_windows_24h']} outage / {r['n_control_windows_24h']} control 24h windows ===")
        print(f"  fan-family co-location median span: {r['fan_family_colocation_median_span_min']} min "
              f"(frac <1min: {r['fan_family_span_frac_under_1min']})")
        print("  top distinct-code 3-chains (outage vs control support, median span):")
        for c in r["top_distinct_3chains"][:8]:
            print(f"    {c['chain']}  out {c['support_frac_outage']} ctrl {c['support_frac_control']}  span {c['median_span_min']}min")
    (ROOT / "results/patterns/degradation_chains.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
