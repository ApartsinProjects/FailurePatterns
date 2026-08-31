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


def _case_windows(ev, turbines, exclude_intervening=True):
    """Pre-outage windows. When exclude_intervening, drop any window that
    contains a PRIOR forced outage within the horizon, so a window cannot be
    the aftermath of an earlier outage (guards against repeat-outage
    clustering wearing a degradation costume)."""
    allfo = ev[ev["event_type"] == "terminal_failure"]
    fo = allfo[allfo["entity_id"].isin(turbines)]
    non = ev[ev["event_type"] != "terminal_failure"]
    fo_by_ent = {e: g["timestamp"].sort_values().to_numpy() for e, g in allfo.groupby("entity_id")}
    wins = []
    for _, o in fo.iterrows():
        if exclude_intervening:
            arr = fo_by_ent.get(o["entity_id"])
            a = np.datetime64(o["timestamp"])
            if arr is not None and ((arr < a) & (arr >= a - np.timedelta64(int(H.total_seconds()), "s"))).any():
                continue
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
    """Length-3..MAX_LEN ordered distinct-code chains frequent in cases, rare in ctrl."""
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
    # per-single-code control hits and median lead on the inference windows,
    # for the terminal-code incremental comparison (A2).
    code_ctrl = Counter()
    for seq, _ in inf_ctrl:
        for c in {c for c, _ in seq}:
            code_ctrl[c] += 1
    code_lead = {}
    for code in {c for seq, _ in inf_case for c, _ in seq}:
        leads = []
        for seq, anchor in inf_case:
            hits = [t for c, t in seq if c == code]
            if hits:
                leads.append((anchor - hits[-1]).total_seconds() / 60.0)
        if leads:
            code_lead[code] = float(np.median(leads))

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
        term = chain[-1]
        term_ctrl = code_ctrl.get(term, 0)
        term_lead = code_lead.get(term)
        chain_lead = float(np.median(leads)) if leads else None
        # chain earns its row only if it is more specific (fewer controls) than
        # its terminal code alone, or gives a longer lead than the terminal code.
        incremental = (hc < term_ctrl) or (
            chain_lead is not None and term_lead is not None and chain_lead > term_lead + 5)
        rows.append({"chain": " -> ".join(chain), "_tuple": chain, "n_codes": len(chain),
                     "inf_case": hf, "inf_case_n": nf, "inf_ctrl": hc, "inf_ctrl_n": nc,
                     "lift": round(lift, 2), "max_lift_ceiling": round((nf + nc) / nf, 2),
                     "terminal_code": term, "terminal_ctrl_hits": term_ctrl,
                     "terminal_median_lead_min": round(term_lead, 1) if term_lead else None,
                     "incremental_over_terminal": bool(incremental),
                     "median_span_min": round(float(np.median(spans)), 1) if spans else None,
                     "median_lead_min": round(chain_lead, 1) if chain_lead is not None else None})
        pvals.append(p)
    n_validated = 0
    closed_rows = []
    if pvals:
        qb = by_qvalues(np.array(pvals)); qh = bh_qvalues(np.array(pvals))
        for r, p, qbb, qhh in zip(rows, pvals, qb, qh):
            r["p"] = p; r["q_by"] = float(qbb); r["q_bh"] = float(qhh)
        val = [r for r in rows if r["q_by"] < 0.05]
        n_validated = len(val)
        # closed filtering: drop a chain if a super-chain with the SAME case
        # support is also validated (it is a redundant sub-chain).
        by_case = {}
        for r in val:
            by_case.setdefault(r["inf_case"], []).append(r)
        def is_sub(a, b):
            return a != b and _ordered_in(a, list(b))
        closed = []
        for r in val:
            supers = [o for o in by_case.get(r["inf_case"], [])
                      if is_sub(r["_tuple"], o["_tuple"])]
            if not supers:
                closed.append(r)
        closed.sort(key=lambda r: (-(r["median_lead_min"] or 0), r["q_by"]))
        closed_rows = closed
    n_incr = sum(1 for r in closed_rows if r["incremental_over_terminal"])
    for r in closed_rows:
        r.pop("_tuple", None)
    return {"farm": name, "disc_turbines": disc_t, "inf_turbines": inf_t,
            "n_inf_case_windows": nf, "n_inf_ctrl_windows": nc,
            "n_candidates": len(cand), "n_validated_by": n_validated,
            "n_validated_closed": len(closed_rows),
            "n_closed_incremental_over_terminal": n_incr,
            "validated_chains": closed_rows[:15]}


def main():
    out = {}
    for name, path in [("Kelmarsh", ROOT / "data/processed/kelmarsh_events.parquet"),
                       ("Penmanshiel", ROOT / "data/processed/penmanshiel_events.parquet")]:
        out[name] = analyse(name, path)
        r = out[name]
        print(f"\n=== {name}: {r['n_candidates']} cand, {r['n_validated_by']} BY-validated, "
              f"{r['n_validated_closed']} closed, {r['n_closed_incremental_over_terminal']} "
              f"incremental-over-terminal (inf {r['n_inf_case_windows']}/{r['n_inf_ctrl_windows']}) ===")
        for c in r["validated_chains"][:8]:
            print(f"  {c['chain']}  lift {c['lift']}/{c['max_lift_ceiling']}  lead {c['median_lead_min']}min "
                  f"(term {c['terminal_code']} ctrl {c['terminal_ctrl_hits']} lead {c['terminal_median_lead_min']})  "
                  f"case {c['inf_case']}/{c['inf_case_n']} ctrl {c['inf_ctrl']}  incr {c['incremental_over_terminal']}")
    (ROOT / "results/patterns" / _ARGS.out).write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
