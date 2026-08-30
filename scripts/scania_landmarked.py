"""W16: landmarked SCANIA prediction on the risk-set windows.

The 0.826 random-CV number in the diagnostic used full-history
aggregates and cross-validation, so it is not a valid prospective
ceiling. This runs a fair landmarked comparison: every feature is
computed only from the pre-anchor risk-set window (history strictly
before the case time), and models are trained and tested on
entity-disjoint vehicle splits. If richer representations still do not
clear ~0.75, the conclusion is limited prospective information at this
readout cadence rather than a representation limitation.

Representations (all landmarked, from the pre-anchor window):
  counts        : total token count + number of distinct codes
  token_counts  : per-code coun 397/158/... token multiplicity
  patterns      : presence of the significant counter_surprise itemsets
  lgbm_tokens   : LightGBM on token_counts
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

warnings.filterwarnings("ignore")
SEED = 20260828


def _items(row):
    return [f"{t}:{s}" for t, s in zip(row["event_type_seq"], row["event_subtype_seq"])]


def main() -> int:
    w = pd.read_parquet(ROOT / "results/patterns/scania_riskset_windows.parquet")
    ents = np.array(sorted(w["entity_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ents)
    cut = len(ents) // 2
    tr = w[w["entity_id"].isin(set(ents[:cut]))].copy()
    te = w[w["entity_id"].isin(set(ents[cut:]))].copy()
    ytr = tr["is_failure"].astype(int).to_numpy()
    yte = te["is_failure"].astype(int).to_numpy()
    tr_items = [_items(r) for _, r in tr.iterrows()]
    te_items = [_items(r) for _, r in te.iterrows()]

    vocab = sorted({it for its in tr_items for it in its})
    idx = {v: j for j, v in enumerate(vocab)}

    def counts(sub):
        n = sub["n_events"].to_numpy(float)
        ndist = np.array([len(set(_items(r))) for _, r in sub.iterrows()], float)
        return np.column_stack([n, ndist])

    def tokcount(rows_items):
        X = np.zeros((len(rows_items), len(vocab)))
        for i, its in enumerate(rows_items):
            for it in its:
                if it in idx:
                    X[i, idx[it]] += 1
        return X

    it_tr, _ = mine_itemsets(tr, ["last20"] if "last20" in tr["horizon"].unique() else [tr["horizon"].iloc[0]],
                             min_support=0.05)
    pats = [frozenset(x) for x in it_tr["itemset"]] if not it_tr.empty else []

    def patpres(rows_items):
        if not pats:
            return np.zeros((len(rows_items), 0))
        X = np.zeros((len(rows_items), len(pats)))
        for i, its in enumerate(rows_items):
            s = set(its)
            for j, p in enumerate(pats):
                if p.issubset(s):
                    X[i, j] = 1
        return X

    def lr(Xtr, Xte):
        if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
            return None
        sc = StandardScaler().fit(Xtr)
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return round(roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1]), 3)

    Xtr_tc, Xte_tc = tokcount(tr_items), tokcount(te_items)
    d = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                           verbose=-1, random_state=SEED).fit(Xtr_tc, ytr)
    res = {
        "counts": lr(counts(tr), counts(te)),
        "token_counts": lr(Xtr_tc, Xte_tc),
        "patterns": lr(patpres(tr_items), patpres(te_items)),
        "lgbm_tokens": round(roc_auc_score(yte, d.predict_proba(Xte_tc)[:, 1]), 3),
    }
    summary = {
        "design": "landmarked risk-set windows, entity-disjoint vehicle split",
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_train_vehicles": int(cut), "n_test_vehicles": int(len(ents) - cut),
        "landmarked_auroc": res,
        "reads_as": ("no representation clears 0.75 at this cadence"
                     if max(v for v in res.values() if v) < 0.75
                     else "a richer representation lifts prospective prediction"),
    }
    (ROOT / "results/patterns/scania_landmarked.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
