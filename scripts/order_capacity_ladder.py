"""Experiment 1 (autonomous loop): capacity-matched order test.

Closes the biggest hole in the paper's order claim: the current decomposition
operationalises order only as adjacent bigrams, so a reviewer can say "you showed
bigrams rarely help, not that order rarely helps." Here we give order every
reasonable chance and hold model capacity fixed by comparing the SAME
representation on real windows vs count-preserving-shuffled windows (order
destroyed, multiset preserved). The difference isolates pure order at that
representation's capacity:

    order_value(R) = AUROC(R on real) - AUROC(R on shuffled)

Representations, increasing order capacity:
    multiset      : token counts        (order-INVARIANT -> order_value must be ~0: shuffle sanity)
    bigram        : adjacent 2-grams     (local order)
    precedence    : all-pairs A-before-B (global order, top-K tokens)

Built-in validation:
  (V1) Positive control: Azure must show a clearly POSITIVE order_value (the assay
       can detect order at all). If it does not, the experiment is INVALID.
  (V2) Shuffle sanity: multiset order_value must be ~0 (shuffle preserves the
       multiset, so a count model cannot tell real from shuffled).
  (V3) Power calibration: inject a known A-before-B ordering into a fraction f of
       positive windows (counts preserved) on a currently-null real trace; the
       precedence order_value must RISE with f, proving the detector has power.

A robust result = Azure positive + power detected + real traces ~0 = a
power-calibrated negative ("order genuinely absent", not "weak detector").
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
warnings.filterwarnings("ignore")
NSPLIT = 8
TOPK_PREC = 20   # top-K tokens for all-pairs precedence (K*(K-1)/2 features)

TRACES = [
    ("Azure",    "azure_windows.parquet",        "last5"),   # positive control
    ("Alibaba",  "alibaba_windows.parquet",      "last3"),
    ("Kelmarsh", "kelmarsh_windows.parquet",     "last5"),
    ("Sepsis",   "sepsis_trend_windows.parquet", "last10"),
]


def _items(r):
    return [f"{t}:{s}" for t, s in zip(r["event_type_seq"], r["event_subtype_seq"])]


def _shuffle(items, rng):
    a = list(items)
    rng.shuffle(a)
    return a


def _multiset(rows, vocab):
    idx = {v: j for j, v in enumerate(vocab)}
    X = np.zeros((len(rows), len(vocab)))
    for i, its in enumerate(rows):
        for it in its:
            if it in idx:
                X[i, idx[it]] += 1
    return X


def _bigrams(its):
    return [f"{a}>{b}" for a, b in zip(its[:-1], its[1:])]


def _bigram_feats(rows, bvocab):
    idx = {v: j for j, v in enumerate(bvocab)}
    X = np.zeros((len(rows), len(bvocab)))
    for i, its in enumerate(rows):
        for bg in set(_bigrams(its)):
            if bg in idx:
                X[i, idx[bg]] = 1
    return X


def _precedence_feats(rows, toks):
    """All-pairs A-before-B indicators over the top-K tokens `toks`.
    Feature (a,b) = 1 if first occurrence of a precedes first occurrence of b."""
    pairs = [(a, b) for i, a in enumerate(toks) for b in toks[i + 1:]]
    pidx = {p: j for j, p in enumerate(pairs)}
    X = np.zeros((len(rows), len(pairs)))
    for i, its in enumerate(rows):
        first = {}
        for pos, it in enumerate(its):
            if it in toks and it not in first:
                first[it] = pos
        for (a, b), j in pidx.items():
            if a in first and b in first:
                X[i, j] = 1.0 if first[a] < first[b] else 0.0
    return X


def _auc(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return 0.5
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1])


def _entity_splits(sub):
    ents = sub["entity_id"].unique()
    for seed in range(NSPLIT):
        rng = np.random.default_rng(seed)
        e = ents.copy(); rng.shuffle(e)
        cut = int(len(e) * 0.7); tre = set(e[:cut].tolist())
        tr = sub[sub["entity_id"].isin(tre)]; te = sub[~sub["entity_id"].isin(tre)]
        if tr["is_failure"].nunique() < 2 or te["is_failure"].nunique() < 2:
            continue
        yield seed, tr, te


def _order_values(sub, inject=None):
    """Return per-representation order_value (real - shuffled) over NSPLIT splits.
    inject=(a,b,frac) optionally enforces a-before-b in `frac` of positive windows
    (counts preserved) to test detector power."""
    ovals = {"multiset": [], "bigram": [], "precedence": []}
    for seed, tr, te in _entity_splits(sub):
        rng = np.random.default_rng(1000 + seed)
        tri = [_items(r) for _, r in tr.iterrows()]
        tei = [_items(r) for _, r in te.iterrows()]
        ytr = tr["is_failure"].astype(int).to_numpy()
        yte = te["is_failure"].astype(int).to_numpy()

        if inject is not None:
            a, b, frac = inject
            def enforce(rows, y):
                out = []
                for its, lab in zip(rows, y):
                    if lab == 1 and (a in its) and (b in its) and rng.random() < frac:
                        others = [x for x in its if x not in (a, b)]
                        na = its.count(a); nb = its.count(b)
                        its = [a] * na + [b] * nb + others  # a's before b's, counts kept
                    out.append(its)
                return out
            tri = enforce(tri, ytr); tei = enforce(tei, yte)

        # count-preserving shuffles (order destroyed, multiset preserved)
        tri_s = [_shuffle(x, rng) for x in tri]
        tei_s = [_shuffle(x, rng) for x in tei]

        vocab = sorted({it for its in tri for it in its})
        # vocab for reps fixed on train (real) so real/shuffled share feature space
        bc = Counter(bg for its in tri for bg in _bigrams(its))
        bvocab = [b for b, _ in bc.most_common(200)]
        tc = Counter(it for its in tri for it in its)
        toks = [t for t, _ in tc.most_common(TOPK_PREC)]

        # multiset (order-invariant sanity): real vs shuffled
        aucs = {}
        aucs["multiset_real"] = _auc(_multiset(tri, vocab), ytr, _multiset(tei, vocab), yte)
        aucs["multiset_shuf"] = _auc(_multiset(tri_s, vocab), ytr, _multiset(tei_s, vocab), yte)
        aucs["bigram_real"] = _auc(_bigram_feats(tri, bvocab), ytr, _bigram_feats(tei, bvocab), yte)
        aucs["bigram_shuf"] = _auc(_bigram_feats(tri_s, bvocab), ytr, _bigram_feats(tei_s, bvocab), yte)
        aucs["prec_real"] = _auc(_precedence_feats(tri, toks), ytr, _precedence_feats(tei, toks), yte)
        aucs["prec_shuf"] = _auc(_precedence_feats(tri_s, toks), ytr, _precedence_feats(tei_s, toks), yte)

        ovals["multiset"].append(aucs["multiset_real"] - aucs["multiset_shuf"])
        ovals["bigram"].append(aucs["bigram_real"] - aucs["bigram_shuf"])
        ovals["precedence"].append(aucs["prec_real"] - aucs["prec_shuf"])
    return {k: (round(float(np.mean(v)), 3), round(float(np.min(v)), 3),
               round(float(np.max(v)), 3), round(float(np.mean(np.array(v) > 0)), 2))
            for k, v in ovals.items() if v}


def main():
    report = {}
    for name, path, hz in TRACES:
        fp = ROOT / "data/processed" / path
        if not fp.exists():
            print(f"[{name}] MISSING {path}, skip", flush=True); continue
        w = pd.read_parquet(fp)
        sub = w[w["horizon"] == hz]
        if sub.empty:
            print(f"[{name}] no {hz} windows, skip", flush=True); continue
        r = _order_values(sub)
        report[name] = {"horizon": hz, "n_fail": int(sub["is_failure"].sum()),
                        "n_ctrl": int((~sub["is_failure"]).sum()), "order_values": r}
        print(f"[{name} {hz}] order_value (mean,min,max,frac>0):", flush=True)
        for rep in ("multiset", "bigram", "precedence"):
            if rep in r:
                print(f"    {rep:11s}: {r[rep]}", flush=True)

    # ---- V3 power calibration on a currently-null real trace (Kelmarsh) ----
    print("\n[Power calibration on Kelmarsh last5: inject A-before-B into positive windows]", flush=True)
    w = pd.read_parquet(ROOT / "data/processed/kelmarsh_windows.parquet")
    sub = w[w["horizon"] == "last5"]
    tc = Counter(it for _, r in sub.iterrows() for it in _items(r))
    top2 = [t for t, _ in tc.most_common(2)]
    power = {}
    if len(top2) == 2:
        a, b = top2
        for frac in [0.0, 0.1, 0.2, 0.4]:
            ov = _order_values(sub, inject=(a, b, frac))
            power[str(frac)] = ov.get("precedence")
            print(f"    inject frac={frac}: precedence order_value={ov.get('precedence')}", flush=True)
        report["_power_calibration"] = {"trace": "Kelmarsh", "pair": [a, b], "by_frac": power}

    (ROOT / "results/patterns/order_capacity_ladder.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote results/patterns/order_capacity_ladder.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
