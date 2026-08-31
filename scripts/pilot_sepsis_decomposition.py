"""Feasibility pilot: presence / multiplicity / order decomposition on the
PhysioNet/CinC 2019 sepsis cohort.

The six operational datasets showed pure PRESENCE signal: which codes appear
before failure carries the risk, while multiplicity and order add nothing (their
increments' CIs include zero). This pilot asks whether a genuinely SLOW,
pre-outcome clinical deterioration behaves differently, i.e. whether the ordered
trajectory of organ dysfunction (order) or its persistence (multiplicity) adds
predictive signal above mere presence.

Representations (built from the same windows, vocab fixed on the training split,
identical definitions to scripts/representation_experiment.py):
  event_count : [n_events]                        degenerate last-K baseline
  presence    : binary unigram (abnormality present)          -> SET
  multiset    : unigram counts (abnormal-hour multiplicity)   -> +MULTIPLICITY
  bigram      : binary adjacent 2-grams (top-K on train)      -> +ORDER

Decomposition (held-out AUROC, patient-level bootstrap 95% CI):
  presence_effect        = AUROC(presence) - AUROC(event_count)
  multiplicity_increment = AUROC(multiset) - AUROC(presence)
  order_increment        = AUROC(bigram)   - AUROC(multiset)

This is the SPMF-free core of the decomposition, so it runs on the dense
hourly-repeating sepsis windows where PrefixSpan sequence mining is intractable.
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
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
SEED = 20260828
B = 300
TOP_BIGRAMS = 200
MAX_WINDOWS = 12000


def _items(row):
    return [f"{t}:{s}" for t, s in zip(row["event_type_seq"], row["event_subtype_seq"])]


def _entity_split(sub, frac=0.70):
    """Entity-disjoint split: no patient appears in both train and test.
    Patients are ordered by id (arbitrary in the challenge files, so this is a
    random 70/30 patient split), then all windows of the first 70% of patients
    form the training set and the rest the held-out set."""
    ents = np.sort(sub["entity_id"].unique())
    cut = int(len(ents) * frac)
    tr_ents = set(ents[:cut].tolist())
    tr = sub[sub["entity_id"].isin(tr_ents)]
    te = sub[~sub["entity_id"].isin(tr_ents)]
    return tr, te


def _presence(rows_items, vocab):
    X = np.zeros((len(rows_items), len(vocab)))
    idx = {v: j for j, v in enumerate(vocab)}
    for i, its in enumerate(rows_items):
        for it in set(its):
            if it in idx:
                X[i, idx[it]] = 1.0
    return X


def _multiset(rows_items, vocab):
    X = np.zeros((len(rows_items), len(vocab)))
    idx = {v: j for j, v in enumerate(vocab)}
    for i, its in enumerate(rows_items):
        for it in its:
            if it in idx:
                X[i, idx[it]] += 1.0
    return X


def _bigrams(its):
    return [f"{a}>{b}" for a, b in zip(its[:-1], its[1:])]


def _bigram_feats(rows_items, bvocab):
    X = np.zeros((len(rows_items), len(bvocab)))
    idx = {v: j for j, v in enumerate(bvocab)}
    for i, its in enumerate(rows_items):
        for bg in set(_bigrams(its)):
            if bg in idx:
                X[i, idx[bg]] = 1.0
    return X


def _auroc_lr(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return None, None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return p, roc_auc_score(yte, p)


def _auroc_lgb(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2:
        return None, None
    d = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                           verbose=-1, random_state=SEED)
    d.fit(Xtr, ytr)
    p = d.predict_proba(Xte)[:, 1]
    return p, roc_auc_score(yte, p)


def _boot_ci(yte, scores, ent, paired=None):
    rng = np.random.default_rng(SEED)
    ents = np.unique(ent)
    ent_to_idx = {e: np.where(ent == e)[0] for e in ents}
    aurocs, deltas = [], []
    for _ in range(B):
        samp = rng.choice(ents, size=len(ents), replace=True)
        idx = np.concatenate([ent_to_idx[e] for e in samp])
        yb = yte[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(roc_auc_score(yb, scores[idx]))
        if paired is not None:
            deltas.append(roc_auc_score(yb, scores[idx]) - roc_auc_score(yb, paired[idx]))
    ci = [round(float(np.percentile(aurocs, 2.5)), 3),
          round(float(np.percentile(aurocs, 97.5)), 3)] if aurocs else [None, None]
    dci = None
    if paired is not None and deltas:
        dci = [round(float(np.percentile(deltas, 2.5)), 3),
               round(float(np.percentile(deltas, 97.5)), 3)]
    return ci, dci


def run_horizon(w, horizon):
    sub = w[w["horizon"] == horizon].copy()
    if len(sub) > MAX_WINDOWS:
        sub = sub.sample(MAX_WINDOWS, random_state=SEED)
    tr, te = _entity_split(sub)
    if len(tr) == 0 or len(te) == 0 or te["is_failure"].nunique() < 2:
        return None
    tr_items = [_items(r) for _, r in tr.iterrows()]
    te_items = [_items(r) for _, r in te.iterrows()]
    ytr = tr["is_failure"].astype(int).to_numpy()
    yte = te["is_failure"].astype(int).to_numpy()
    ent = te["entity_id"].to_numpy()

    vocab = sorted({it for its in tr_items for it in its})
    bcount = Counter(bg for its in tr_items for bg in _bigrams(its))
    bvocab = [b for b, _ in bcount.most_common(TOP_BIGRAMS)]

    reps = {
        "event_count": (tr["n_events"].to_numpy(float).reshape(-1, 1),
                        te["n_events"].to_numpy(float).reshape(-1, 1), "lr"),
        "presence":    (_presence(tr_items, vocab), _presence(te_items, vocab), "lr"),
        "multiset":    (_multiset(tr_items, vocab), _multiset(te_items, vocab), "lr"),
        "bigram":      (_bigram_feats(tr_items, bvocab), _bigram_feats(te_items, bvocab), "lr"),
        "lgbm_multiset": (_multiset(tr_items, vocab), _multiset(te_items, vocab), "lgb"),
    }
    scores, aurocs, cis = {}, {}, {}
    for rn, (Xtr, Xte, model) in reps.items():
        p, a = (_auroc_lgb if model == "lgb" else _auroc_lr)(Xtr, ytr, Xte, yte)
        scores[rn] = p
        aurocs[rn] = round(a, 3) if a is not None else None
    for rn in ("event_count", "presence", "multiset", "bigram"):
        if scores.get(rn) is not None:
            cis[rn], _ = _boot_ci(yte, scores[rn], ent)

    def paired(hi, lo):
        if scores.get(hi) is None or scores.get(lo) is None:
            return None, None
        d = round(aurocs[hi] - aurocs[lo], 3)
        _, dci = _boot_ci(yte, scores[hi], ent, paired=scores[lo])
        return d, dci

    pres_eff, pres_ci = paired("presence", "event_count")
    mult_inc, mult_ci = paired("multiset", "presence")
    order_inc, order_ci = paired("bigram", "multiset")

    n_ev = sub["n_events"]
    return {
        "trace": "Sepsis", "horizon": horizon,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_test_entities": int(len(np.unique(ent))),
        "median_events_per_window": float(n_ev.median()),
        "auroc": aurocs, "auroc_ci95": cis,
        "decomposition": {
            "presence_effect": {"delta": pres_eff, "ci95": pres_ci},
            "multiplicity_increment": {"delta": mult_inc, "ci95": mult_ci},
            "order_increment": {"delta": order_inc, "ci95": order_ci},
        },
    }


def main() -> int:
    w = pd.read_parquet(ROOT / "data/processed/sepsis_windows.parquet")
    avail = set(w["horizon"].unique())
    horizons = [h for h in ["24h", "6h", "1h", "last10", "last5"] if h in avail]
    out = []
    for h in horizons:
        r = run_horizon(w, h)
        if r is None:
            print(f"[Sepsis {h}] skipped (degenerate)"); continue
        out.append(r)
        d = r["decomposition"]
        print(f"[Sepsis {h}] n_tr={r['n_train']} n_te={r['n_test']} ent={r['n_test_entities']} "
              f"med_ev/win={r['median_events_per_window']}", flush=True)
        print(f"    AUROC: count={r['auroc']['event_count']} presence={r['auroc']['presence']} "
              f"multiset={r['auroc']['multiset']} bigram={r['auroc']['bigram']} "
              f"lgbm={r['auroc']['lgbm_multiset']}", flush=True)
        print(f"    presence_eff={d['presence_effect']['delta']} (CI {d['presence_effect']['ci95']})  "
              f"mult+={d['multiplicity_increment']['delta']} (CI {d['multiplicity_increment']['ci95']})  "
              f"order+={d['order_increment']['delta']} (CI {d['order_increment']['ci95']})", flush=True)
    (ROOT / "results/patterns/pilot_sepsis_decomposition.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results/patterns/pilot_sepsis_decomposition.json ({len(out)} horizons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
