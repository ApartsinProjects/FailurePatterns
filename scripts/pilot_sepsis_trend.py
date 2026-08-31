"""Trend/severity sepsis pilot: build windows over the transition-event stream,
run the presence/multiplicity/order decomposition, AND run the length-matched
verification (windows with exactly K events, so raw count cannot discriminate)
in one pass. A structural signal is believed only if it survives length matching.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.windows import run  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20260828
NSPLIT = 8
COUNT_K = {"last5": 5, "last10": 10}


def _items(r):
    return [f"{t}:{s}" for t, s in zip(r["event_type_seq"], r["event_subtype_seq"])]


def _pres(rows, v):
    X = np.zeros((len(rows), len(v))); idx = {x: j for j, x in enumerate(v)}
    for i, its in enumerate(rows):
        for it in set(its):
            if it in idx: X[i, idx[it]] = 1
    return X


def _mult(rows, v):
    X = np.zeros((len(rows), len(v))); idx = {x: j for j, x in enumerate(v)}
    for i, its in enumerate(rows):
        for it in its:
            if it in idx: X[i, idx[it]] += 1
    return X


def _bigrams(its):
    return [f"{a}>{b}" for a, b in zip(its[:-1], its[1:])]


def _big(rows, bv):
    X = np.zeros((len(rows), len(bv))); idx = {x: j for j, x in enumerate(bv)}
    for i, its in enumerate(rows):
        for bg in set(_bigrams(its)):
            if bg in idx: X[i, idx[bg]] = 1
    return X


def _cnt(rows):
    return np.array([[len(r)] for r in rows], float)


def _auc(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return 0.5
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1])


def _splits(sub, restrict_full=None):
    """Yield (tr, te) over NSPLIT random entity-disjoint splits.
    restrict_full=K keeps only windows with exactly K events (length matched)."""
    s = sub if restrict_full is None else sub[sub["n_events"] == restrict_full]
    ents = s["entity_id"].unique()
    for seed in range(NSPLIT):
        rng = np.random.default_rng(seed)
        e = ents.copy(); rng.shuffle(e)
        cut = int(len(e) * 0.7); tre = set(e[:cut].tolist())
        tr = s[s["entity_id"].isin(tre)]; te = s[~s["entity_id"].isin(tre)]
        if te["is_failure"].nunique() < 2 or tr["is_failure"].nunique() < 2:
            continue
        yield tr, te


def _measure(sub, restrict_full=None):
    cnt, pres, mult, big = [], [], [], []
    dmp, dob = [], []   # mult-pres, bigram-mult
    for tr, te in _splits(sub, restrict_full):
        tri = [_items(r) for _, r in tr.iterrows()]
        tei = [_items(r) for _, r in te.iterrows()]
        ytr = tr["is_failure"].astype(int).to_numpy()
        yte = te["is_failure"].astype(int).to_numpy()
        v = sorted({it for its in tri for it in its})
        bc = Counter(bg for its in tri for bg in _bigrams(its))
        bv = [b for b, _ in bc.most_common(200)]
        ac = _auc(_cnt(tri), ytr, _cnt(tei), yte)
        ap = _auc(_pres(tri, v), ytr, _pres(tei, v), yte)
        am = _auc(_mult(tri, v), ytr, _mult(tei, v), yte)
        ab = _auc(_big(tri, bv), ytr, _big(tei, bv), yte)
        cnt.append(ac); pres.append(ap); mult.append(am); big.append(ab)
        dmp.append(am - ap); dob.append(ab - am)

    def summ(x):
        x = np.array(x)
        return {"mean": round(float(x.mean()), 3), "min": round(float(x.min()), 3),
                "max": round(float(x.max()), 3)}
    return {
        "auroc": {"count": summ(cnt), "presence": summ(pres),
                  "multiset": summ(mult), "bigram": summ(big)},
        "multiplicity_increment": {**summ(dmp), "frac_pos": round(float(np.mean(np.array(dmp) > 0)), 2)},
        "order_increment": {**summ(dob), "frac_pos": round(float(np.mean(np.array(dob) > 0)), 2)},
    }


def main():
    ev_path = ROOT / "data/processed/sepsis_trend_events.parquet"
    out_dir = ROOT / "data/processed"
    print("building trend windows...", flush=True)
    run(ev_path, out_dir, output_stem="sepsis_trend_windows",
        failure_event_type="terminal_failure", seed_timestamps=None)
    w = pd.read_parquet(out_dir / "sepsis_trend_windows.parquet")

    report = {}
    for h in ["24h", "6h", "last10", "last5"]:
        sub = w[w["horizon"] == h]
        if sub.empty:
            continue
        full = {
            "all_windows": _measure(sub),
            "n_fail": int(sub["is_failure"].sum()),
            "n_ctrl": int((~sub["is_failure"]).sum()),
            "median_events": float(sub["n_events"].median()),
        }
        if h in COUNT_K:
            fsub = sub[sub["n_events"] == COUNT_K[h]]
            full["length_matched"] = _measure(sub, restrict_full=COUNT_K[h])
            full["length_matched_n_fail"] = int(fsub["is_failure"].sum())
            full["length_matched_n_ctrl"] = int((~fsub["is_failure"]).sum())
        report[h] = full
        a = full["all_windows"]
        print(f"\n[{h}] n_fail={full['n_fail']} n_ctrl={full['n_ctrl']} med_ev={full['median_events']}", flush=True)
        print(f"  ALL: count={a['auroc']['count']['mean']} presence={a['auroc']['presence']['mean']} "
              f"multiset={a['auroc']['multiset']['mean']} bigram={a['auroc']['bigram']['mean']} | "
              f"mult+={a['multiplicity_increment']['mean']}(f{a['multiplicity_increment']['frac_pos']}) "
              f"order+={a['order_increment']['mean']}(f{a['order_increment']['frac_pos']})", flush=True)
        if "length_matched" in full:
            m = full["length_matched"]
            print(f"  MATCHED (n_ev==K): count={m['auroc']['count']['mean']} presence={m['auroc']['presence']['mean']} "
                  f"multiset={m['auroc']['multiset']['mean']} bigram={m['auroc']['bigram']['mean']} | "
                  f"mult+={m['multiplicity_increment']['mean']}(f{m['multiplicity_increment']['frac_pos']}) "
                  f"order+={m['order_increment']['mean']}(f{m['order_increment']['frac_pos']})", flush=True)
    (ROOT / "results/patterns/pilot_sepsis_trend.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote results/patterns/pilot_sepsis_trend.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
