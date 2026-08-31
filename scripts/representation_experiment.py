"""W3 + W4 + W15: baselines, presence/multiplicity/order decomposition,
and entity-bootstrap confidence intervals, on the four temporally-split
traces (Azure, Alibaba, BGL, SCANIA).

Representations (all built from the same windows, vocab fixed on train):
  event_count   : [n_events]                     (degenerate last-K baseline)
  presence      : binary unigram (event type present)              -> SET
  multiset      : unigram counts (event type multiplicity)         -> +MULTIPLICITY
  bigram        : binary adjacent 2-grams (top-K on train)
  recency       : n_events + window time span + last-event gap
  mined_itemset : binary presence of train-mined significant itemsets
  mined_sequence: binary presence of train-mined significant sequences (ORDER)
  combined      : count + itemset + sequence

Models: L2 logistic regression for every representation; LightGBM on the
multiset (count) representation as a strong shallow baseline (W3).

Decomposition (W4), all on the same temporal test set, AUROC:
  presence effect       = AUROC(presence)  - AUROC(event_count)
  multiplicity increment= AUROC(multiset)  - AUROC(presence)
  order increment       = AUROC(mined_sequence) - AUROC(mined_itemset)

Uncertainty (W15): entity-level bootstrap (resample test ENTITIES with
replacement, B=500) gives a 95% CI per AUROC and a paired CI for each
decomposition increment.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mine.itemsets import mine as mine_itemsets
from src.mine.sequences import mine as mine_sequences

warnings.filterwarnings("ignore")
SEED = 20260828
B = 200
TOP_BIGRAMS = 200
MAX_WINDOWS = 8000  # entity-stratified cap so bootstrap stays tractable

CONFIGS = [
    ("Azure",   "azure_windows.parquet",   "last5"),
    ("Alibaba", "alibaba_windows.parquet", "last3"),
    ("BGL",     "bgl_windows.parquet",     "last20"),
    ("SCANIA",  "scania_windows.parquet",  "last20"),
]


def _items(row):
    return [f"{t}:{s}" for t, s in zip(row["event_type_seq"], row["event_subtype_seq"])]


def _temporal_split(sub):
    sub = sub.sort_values("anchor")
    cut = sub["anchor"].quantile(0.70)
    tr = sub[sub["anchor"] < cut]
    te = sub[sub["anchor"] >= cut]
    return tr, te


def _presence(rows_items, vocab):
    X = np.zeros((len(rows_items), len(vocab)), dtype=float)
    idx = {v: j for j, v in enumerate(vocab)}
    for i, its in enumerate(rows_items):
        for it in set(its):
            if it in idx:
                X[i, idx[it]] = 1.0
    return X


def _multiset(rows_items, vocab):
    X = np.zeros((len(rows_items), len(vocab)), dtype=float)
    idx = {v: j for j, v in enumerate(vocab)}
    for i, its in enumerate(rows_items):
        for it in its:
            if it in idx:
                X[i, idx[it]] += 1.0
    return X


def _bigrams(its):
    return [f"{a}>{b}" for a, b in zip(its[:-1], its[1:])]


def _bigram_feats(rows_items, bvocab):
    X = np.zeros((len(rows_items), len(bvocab)), dtype=float)
    idx = {v: j for j, v in enumerate(bvocab)}
    for i, its in enumerate(rows_items):
        for bg in set(_bigrams(its)):
            if bg in idx:
                X[i, idx[bg]] = 1.0
    return X


def _recency(sub):
    span = (sub["last_event_ts"] - sub["window_start"]).dt.total_seconds().fillna(0).to_numpy()
    n = sub["n_events"].to_numpy(dtype=float)
    return np.column_stack([n, span, np.log1p(span)])


def _pattern_present_itemset(rows_items, patterns):
    if not patterns:
        return np.zeros((len(rows_items), 0))
    X = np.zeros((len(rows_items), len(patterns)), dtype=float)
    for i, its in enumerate(rows_items):
        s = set(its)
        for j, p in enumerate(patterns):
            if p.issubset(s):
                X[i, j] = 1.0
    return X


def _seq_in(seq, tx):
    i = 0
    for it in tx:
        if it == seq[i]:
            i += 1
            if i == len(seq):
                return True
    return False


def _pattern_present_seq(rows_items, patterns):
    if not patterns:
        return np.zeros((len(rows_items), 0))
    X = np.zeros((len(rows_items), len(patterns)), dtype=float)
    for i, its in enumerate(rows_items):
        for j, p in enumerate(patterns):
            if _seq_in(p, its):
                X[i, j] = 1.0
    return X


def _auroc_lr(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return None, None
    sc = StandardScaler(with_mean=True).fit(Xtr)
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


def _boot_ci(yte, scores, ent, B, paired=None):
    """Entity bootstrap: resample entities, recompute AUROC (and paired delta)."""
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


def run_trace(name, path, horizon):
    w = pd.read_parquet(ROOT / "data/processed" / path)
    sub = w[w["horizon"] == horizon].copy()
    if len(sub) > MAX_WINDOWS:
        sub = sub.sample(MAX_WINDOWS, random_state=SEED).sort_values("anchor")
    tr, te = _temporal_split(sub)
    if len(tr) == 0 or len(te) == 0 or te["is_failure"].nunique() < 2:
        return None
    tr_items = [_items(r) for _, r in tr.iterrows()]
    te_items = [_items(r) for _, r in te.iterrows()]
    ytr = tr["is_failure"].astype(int).to_numpy()
    yte = te["is_failure"].astype(int).to_numpy()
    ent = te["entity_id"].to_numpy()

    vocab = sorted({it for its in tr_items for it in its})
    from collections import Counter
    bcount = Counter(bg for its in tr_items for bg in _bigrams(its))
    bvocab = [b for b, _ in bcount.most_common(TOP_BIGRAMS)]

    # mined patterns on train
    it_tr, _ = mine_itemsets(tr, [horizon], min_support=0.05)
    sq_tr, _ = mine_sequences(tr, [horizon], min_support=0.05)
    it_pats = [frozenset(x) for x in it_tr[it_tr["survives_permutation_null"]]["itemset"]] if not it_tr.empty else []
    sq_pats = [list(x) for x in sq_tr[sq_tr["survives_shuffle_null"]]["sequence"]] if not sq_tr.empty else []

    reps = {
        "event_count":    (tr["n_events"].to_numpy(float).reshape(-1, 1), te["n_events"].to_numpy(float).reshape(-1, 1), "lr"),
        "presence":       (_presence(tr_items, vocab), _presence(te_items, vocab), "lr"),
        "multiset":       (_multiset(tr_items, vocab), _multiset(te_items, vocab), "lr"),
        "bigram":         (_bigram_feats(tr_items, bvocab), _bigram_feats(te_items, bvocab), "lr"),
        "recency":        (_recency(tr), _recency(te), "lr"),
        "lgbm_multiset":  (_multiset(tr_items, vocab), _multiset(te_items, vocab), "lgb"),
        "mined_itemset":  (_pattern_present_itemset(tr_items, it_pats), _pattern_present_itemset(te_items, it_pats), "lr"),
        "mined_sequence": (_pattern_present_seq(tr_items, sq_pats), _pattern_present_seq(te_items, sq_pats), "lr"),
    }
    scores, aurocs, cis = {}, {}, {}
    for rn, (Xtr, Xte, model) in reps.items():
        p, a = (_auroc_lgb if model == "lgb" else _auroc_lr)(Xtr, ytr, Xte, yte)
        scores[rn] = p
        aurocs[rn] = round(a, 3) if a is not None else None

    # combined
    Xtr_c = np.hstack([reps["event_count"][0], reps["mined_itemset"][0], reps["mined_sequence"][0]])
    Xte_c = np.hstack([reps["event_count"][1], reps["mined_itemset"][1], reps["mined_sequence"][1]])
    pc, ac = _auroc_lr(Xtr_c, ytr, Xte_c, yte)
    aurocs["combined"] = round(ac, 3) if ac is not None else None
    # bootstrap CIs only for the headline reps + combined (tractable)
    for rn in ("event_count", "presence", "multiset", "mined_itemset", "mined_sequence"):
        if scores.get(rn) is not None:
            cis[rn], _ = _boot_ci(yte, scores[rn], ent, B)
    if pc is not None:
        cis["combined"], _ = _boot_ci(yte, pc, ent, B)

    # decomposition with paired CIs
    def paired(rep_hi, rep_lo):
        if scores.get(rep_hi) is None or scores.get(rep_lo) is None:
            return None, None
        d = round(aurocs[rep_hi] - aurocs[rep_lo], 3)
        _, dci = _boot_ci(yte, scores[rep_hi], ent, B, paired=scores[rep_lo])
        return d, dci
    # clean nested decomposition: presence -> counts (multiplicity) -> bigram (order)
    pres_eff, pres_ci = paired("presence", "event_count")
    mult_inc, mult_ci = paired("multiset", "presence")
    order_inc, order_ci = paired("bigram", "multiset")

    return {
        "trace": name, "horizon": horizon,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_test_entities": int(len(np.unique(ent))),
        "auroc": aurocs, "auroc_ci95": cis,
        "decomposition": {
            "presence_effect": {"delta": pres_eff, "ci95": pres_ci},
            "multiplicity_increment": {"delta": mult_inc, "ci95": mult_ci},
            "order_increment": {"delta": order_inc, "ci95": order_ci},
        },
    }


def main() -> int:
    out = []
    for name, path, h in CONFIGS:
        r = run_trace(name, path, h)
        if r is None:
            print(f"[{name} {h}] skipped"); continue
        out.append(r)
        d = r["decomposition"]
        print(f"[{name} {h}] AUROC: count={r['auroc']['event_count']} "
              f"presence={r['auroc']['presence']} multiset={r['auroc']['multiset']} "
              f"bigram={r['auroc']['bigram']} lgbm={r['auroc']['lgbm_multiset']} "
              f"itemset={r['auroc']['mined_itemset']} seq={r['auroc']['mined_sequence']} "
              f"combined={r["auroc"]["combined"]}", flush=True)
        print(f"        decomposition: presence={d['presence_effect']['delta']} "
              f"(CI {d['presence_effect']['ci95']})  "
              f"mult+={d['multiplicity_increment']['delta']} (CI {d['multiplicity_increment']['ci95']})  "
              f"order+={d['order_increment']['delta']} (CI {d['order_increment']['ci95']})", flush=True)
    (ROOT / "results/patterns/representation_experiment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results/patterns/representation_experiment.json ({len(out)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
