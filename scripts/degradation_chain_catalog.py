"""A validated catalog of temporal degradation chains on the wind farms.

Distinct from the co-located burst signatures of the main catalog: here a
signature is an ORDERED chain of >=3 DISTINCT alarm codes that unfolds over
real time (co-located repeats merged) inside the 24h pre-outage window. The
same post-selection-valid protocol as the rest of the paper is used:

  1. Split turbines entity-disjoint into discovery / inference halves.
  2. On DISCOVERY turbines, build 24h pre-outage windows (cases) and clean-
     region 24h windows (controls), collapse each to its distinct-code
     ordered sequence, and mine frequent length-3 and length-4 ordered
     chains that are enriched in cases over controls (candidate set).
  3. On INFERENCE turbines, build fresh case/control windows and score each
     candidate chain by an exact hypergeometric test, then BH/BY correct.
  4. Report validated chains with their median time span (trajectory length),
     median lead time (last chain event to outage), inference-half case and
     control support, lift, and BY q.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.discovery_inference_split import bh_qvalues, by_qvalues

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--hours", type=int, default=24)
_ap.add_argument("--maxlen", type=int, default=4)
_ap.add_argument("--out", default="degradation_chain_catalog.json")
_ARGS, _ = _ap.parse_known_args()

H = pd.Timedelta(hours=_ARGS.hours)
SEED = 20260828
MIN_SUPPORT = 0.05          # fraction of discovery case windows
MAX_CTRL_FRAC = 0.02        # candidate must be rare in discovery controls
MIN_LEN, MAX_LEN = 3, _ARGS.maxlen


def _distinct_seq(win: pd.DataFrame):
    win = win.sort_values("timestamp")
    out, last = [], None
    for _, r in win.iterrows():
        c = f"{r['event_type']}:{r['event_subtype']}"
        if c == last:
            continue
        out.append((c, r["timestamp"]))
        last = c
    return out


def _case_windows(ev, turbines):
    fo = ev[(ev["event_type"] == "terminal_failure") & ev["entity_id"].isin(turbines)]
    non = ev[ev["event_type"] != "terminal_failure"]
    wins = []
    for _, o in fo.iterrows():
        w = non[(non["entity_id"] == o["entity_id"]) &
                (non["timestamp"] < o["timestamp"]) &
                (non["timestamp"] >= o["timestamp"] - H)]
        if not w.empty:
            wins.append((_distinct_seq(w), o["timestamp"]))
    return wins


def _ctrl_windows(ev, turbines, n_per=40):
    fo = ev[ev["event_type"] == "terminal_failure"]
    non = ev[ev["event_type"] != "terminal_failure"]
    rng = np.random.default_rng(SEED)
    wins = []
    for ent in turbines:
        g = non[non["entity_id"] == ent].sort_values("timestamp")
        ts = g["timestamp"].to_numpy()
        foent = fo[fo["entity_id"] == ent]["timestamp"].to_numpy()
        if len(ts) < 5:
            continue
        for anchor in rng.choice(ts, size=min(n_per, len(ts)), replace=False):
            if len(foent) and ((foent >= anchor) & (foent <= anchor + np.timedelta64(24, "h"))).any():
                continue
            w = g[(g["timestamp"] < anchor) & (g["timestamp"] >= anchor - H)]
            if not w.empty:
                wins.append((_distinct_seq(w), None))
    return wins


def _ordered_in(chain, codes):
    i = 0
    for c in codes:
        if c == chain[i]:
            i += 1
            if i == len(chain):
                return True
    return False


def _chain_stats(chain, win_seq):
    """If win contains chain as ordered subseq, return (span_min, lead_min)."""
    codes = [c for c, _ in win_seq]
    i, pos = 0, []
    for j, c in enumerate(codes):
        if c == chain[i]:
            pos.append(j)
            i += 1
            if i == len(chain):
                break
    if i < len(chain):
        return None
    t_first = win_seq[pos[0]][1]
    t_last = win_seq[pos[-1]][1]
    return t_first, t_last


def _candidates(case_wins, ctrl_wins):
    """Length-3/4 ordered distinct-code chains frequent in cases, rare in ctrl."""
    n_case = len(case_wins)
    case_counts = Counter()
    for seq, _ in case_wins:
        codes = [c for c, _ in seq]
        seen = set()
        for L in range(MIN_LEN, MAX_LEN + 1):
            for combo in combinations(range(len(codes)), L):
                tri = tuple(codes[k] for k in combo)
                if len(set(tri)) == L:      # distinct codes, keep source order
                    seen.add(tri)
        for tri in seen:
            case_counts[tri] += 1
    cand = {tri for tri, c in case_counts.items() if c >= MIN_SUPPORT * max(n_case, 1)}
    # rare in controls
    n_ctrl = len(ctrl_wins)
    ctrl_counts = Counter()
    for seq, _ in ctrl_wins:
        codes = [c for c, _ in seq]
        seen = set()
        for tri in cand:
            if _ordered_in(tri, codes):
                seen.add(tri)
        for tri in seen:
            ctrl_counts[tri] += 1
    return [tri for tri in cand if ctrl_counts.get(tri, 0) <= MAX_CTRL_FRAC * max(n_ctrl, 1)]


def _hyp_upper(hf, hc, nf, nc):
    K, N = hf + hc, nf + nc
    if K == 0 or K == N or nf == 0 or nc == 0:
        return 1.0
    return float(hypergeom.sf(hf - 1, N, K, nf))


def analyse(name, path):
    ev = pd.read_parquet(path)
    turbines = sorted(ev["entity_id"].unique())
    rng = np.random.default_rng(SEED)
    perm = list(rng.permutation(turbines))
    disc_t, inf_t = perm[:len(perm) // 2], perm[len(perm) // 2:]

    disc_case = _case_windows(ev, disc_t); disc_ctrl = _ctrl_windows(ev, disc_t)
    inf_case = _case_windows(ev, inf_t); inf_ctrl = _ctrl_windows(ev, inf_t)
    cand = _candidates(disc_case, disc_ctrl)

    nf, nc = len(inf_case), len(inf_ctrl)
    rows, pvals = [], []
    for chain in cand:
        hf = hc = 0
        spans, leads = [], []
        for seq, anchor in inf_case:
            st = _chain_stats(chain, seq)
            if st:
                hf += 1
                t_first, t_last = st
                spans.append((t_last - t_first).total_seconds() / 60.0)
                leads.append((anchor - t_last).total_seconds() / 60.0)
        for seq, _ in inf_ctrl:
            if _ordered_in(chain, [c for c, _ in seq]):
                hc += 1
        if hf == 0:
            continue
        p = _hyp_upper(hf, hc, nf, nc)
        pooled = (hf + hc) / (nf + nc)
        lift = (hf / nf) / pooled if pooled > 0 else float("nan")
        rows.append({"chain": " -> ".join(chain), "n_codes": len(chain),
                     "inf_case": hf, "inf_case_n": nf, "inf_ctrl": hc, "inf_ctrl_n": nc,
                     "lift": round(lift, 2),
                     "median_span_min": round(float(np.median(spans)), 1) if spans else None,
                     "median_lead_min": round(float(np.median(leads)), 1) if leads else None})
        pvals.append(p)
    if pvals:
        qb = by_qvalues(np.array(pvals)); qh = bh_qvalues(np.array(pvals))
        for r, p, qbb, qhh in zip(rows, pvals, qb, qh):
            r["p"] = p; r["q_by"] = float(qbb); r["q_bh"] = float(qhh)
        rows = [r for r in rows if r["q_by"] < 0.05]
        rows.sort(key=lambda r: (-r["lift"], r["q_by"]))
    return {"farm": name, "disc_turbines": disc_t, "inf_turbines": inf_t,
            "n_inf_case_windows": nf, "n_inf_ctrl_windows": nc,
            "n_candidates": len(cand), "n_validated_by": len(rows),
            "validated_chains": rows[:15]}


def main():
    out = {}
    for name, path in [("Kelmarsh", ROOT / "data/processed/kelmarsh_events.parquet"),
                       ("Penmanshiel", ROOT / "data/processed/penmanshiel_events.parquet")]:
        out[name] = analyse(name, path)
        r = out[name]
        print(f"\n=== {name}: {r['n_candidates']} candidates, {r['n_validated_by']} BY-validated "
              f"(inf {r['n_inf_case_windows']} case / {r['n_inf_ctrl_windows']} ctrl) ===")
        for c in r["validated_chains"][:8]:
            print(f"  {c['chain']}  lift {c['lift']}  span {c['median_span_min']}min  "
                  f"lead {c['median_lead_min']}min  case {c['inf_case']}/{c['inf_case_n']} "
                  f"ctrl {c['inf_ctrl']}/{c['inf_ctrl_n']}  q_by {c['q_by']:.1e}")
    (ROOT / "results/patterns" / _ARGS.out).write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
