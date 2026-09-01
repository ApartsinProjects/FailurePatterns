"""Experiment 2 (autonomous loop): temporal-set / recency test.

Hypothesis (external strategist): failure precursors are PERMUTATION-INVARIANT but
TEMPORALLY LOCALIZED. Risk depends on WHICH precursor events occurred and HOW
RECENTLY each occurred, but not on their order. If true, real precursors are best
represented as a MARKED TEMPORAL SET (event x recency), a representation strictly
between a bag-of-events and a sequence, and this turns the paper's order-negative
into a positive structural discovery.

For each window and each event token e:
    present_e  in {0,1}
    recency_e  = log(1 + (anchor - most_recent_occurrence_of_e))   (0 if absent)
Representations (same L2 logistic, capacity differs only by the recency block):
    P   : presence only              (2 blocks: present)
    P+R : presence + recency

Capacity-matched temporal assay (analogue of the count-preserving order null):
    recency_value = AUROC(P+R, real recencies) - AUROC(P+R, recency-null)
The recency-null permutes each token's recency across the windows where it is
present, WITHIN strata of (n_events bucket, global-last-event-recency bucket), so
it preserves: which events occurred, how many, each token's marginal recency
distribution, generic proximity, and model capacity; it destroys ONLY whether a
particular token tends to occur at a particular distance from the anchor.

Confound guard: a generic-proximity baseline [n_events, window_span,
global_last_event_recency] must NOT reproduce the P+R gain.

Validation: reconstructed per-window event counts must equal the stored n_events.
Win condition (preregistered): recency_value > +0.03 with >=7/8 splits positive on
>= 2 real traces whose order_value is below the 10%-order-injection floor (0.082).
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

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
NSPLIT = 8

# trace -> (windows parquet, events parquet, horizon, K)
TRACES = [
    ("Azure",    "azure_windows.parquet",        "azure_events.parquet",        "last5", 5),
    ("Alibaba",  "alibaba_windows.parquet",      "alibaba_events.parquet",      "last3", 3),
    ("Kelmarsh", "kelmarsh_windows.parquet",     "kelmarsh_events.parquet",     "last5", 5),
    ("Sepsis",   "sepsis_trend_windows.parquet", "sepsis_trend_events.parquet", "last10", 10),
]


def _tok(t, s):
    return f"{t}:{s}"


def build_window_recency(win, ev, K):
    """For each last-K window reconstruct per-token most-recent timestamp and
    thus recency to the anchor. Returns list of dicts and validates event counts."""
    ev = ev.sort_values(["entity_id", "timestamp"])
    by_ent = {e: g for e, g in ev.groupby("entity_id", sort=False)}
    rows = []
    mism = 0
    for _, w in win.iterrows():
        ent = w["entity_id"]; anchor = w["anchor"]
        g = by_ent.get(ent)
        if g is None:
            rows.append({"present": {}, "recency": {}, "n": 0, "span": 0.0,
                         "glast": 0.0, "y": int(w["is_failure"]), "ent": ent})
            continue
        before = g[g["timestamp"] < anchor].tail(K)
        toks = [_tok(t, s) for t, s in zip(before["event_type"], before["event_subtype"])]
        ts = before["timestamp"].to_numpy()
        present, recency = {}, {}
        for tk, tstamp in zip(toks, ts):
            # most recent occurrence -> smallest (anchor - tstamp); keep max tstamp
            if tk not in recency or tstamp > recency[tk]:
                recency[tk] = tstamp
            present[tk] = 1
        rec_secs = {}
        a = np.datetime64(anchor)
        for tk, tstamp in recency.items():
            d = (a - np.datetime64(tstamp)) / np.timedelta64(1, "s")
            rec_secs[tk] = float(np.log1p(max(0.0, d)))
        span = 0.0; glast = 0.0
        if len(ts):
            span = float((a - np.datetime64(ts.min())) / np.timedelta64(1, "s"))
            glast = float((a - np.datetime64(ts.max())) / np.timedelta64(1, "s"))
        if len(before) != int(w["n_events"]):
            mism += 1
        rows.append({"present": present, "recency": rec_secs, "n": len(before),
                     "span": float(np.log1p(span)), "glast": float(np.log1p(glast)),
                     "y": int(w["is_failure"]), "ent": ent})
    return rows, mism


def _mat_P(rows, vocab):
    idx = {v: j for j, v in enumerate(vocab)}
    X = np.zeros((len(rows), len(vocab)))
    for i, r in enumerate(rows):
        for tk in r["present"]:
            if tk in idx:
                X[i, idx[tk]] = 1
    return X


def _mat_PR(rows, vocab, recency_override=None):
    idx = {v: j for j, v in enumerate(vocab)}
    n = len(vocab)
    X = np.zeros((len(rows), 2 * n))
    for i, r in enumerate(rows):
        rec = recency_override[i] if recency_override is not None else r["recency"]
        for tk in r["present"]:
            if tk in idx:
                X[i, idx[tk]] = 1
                X[i, n + idx[tk]] = rec.get(tk, 0.0)
    return X


def _mat_generic(rows):
    return np.array([[r["n"], r["span"], r["glast"]] for r in rows], float)


def _auc(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return 0.5
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1])


def _recency_null(rows, vocab, rng):
    """Per-token permutation of recency across present-windows, within strata of
    (n_events, global-recency quartile). Preserves marginal recency + proximity."""
    n = len(rows)
    # strata key per window
    glast = np.array([r["glast"] for r in rows])
    q = np.quantile(glast, [0.25, 0.5, 0.75]) if len(glast) else [0, 0, 0]
    def gbin(v):
        return int(np.searchsorted(q, v))
    strata = [(rows[i]["n"], gbin(rows[i]["glast"])) for i in range(n)]
    override = [dict() for _ in range(n)]
    for tk in vocab:
        # windows where tk present, grouped by stratum
        buckets = {}
        for i, r in enumerate(rows):
            if tk in r["present"]:
                buckets.setdefault(strata[i], []).append(i)
        for idxs in buckets.values():
            vals = [rows[i]["recency"].get(tk, 0.0) for i in idxs]
            perm = list(vals); rng.shuffle(perm)
            for i, v in zip(idxs, perm):
                override[i][tk] = v
    return override


def run_trace(name, wpath, epath, hz, K):
    win = pd.read_parquet(ROOT / "data/processed" / wpath)
    win = win[win["horizon"] == hz]
    ev = pd.read_parquet(ROOT / "data/processed" / epath)
    ev = ev[ev["event_type"] != "terminal_failure"]  # exclude the anchor marker
    rows, mism = build_window_recency(win, ev, K)
    vocab = sorted({tk for r in rows for tk in r["present"]})
    ents = np.array([r["ent"] for r in rows])
    y = np.array([r["y"] for r in rows])

    recvals, prgain, gengain = [], [], []
    for seed in range(NSPLIT):
        rng = np.random.default_rng(seed)
        ue = np.unique(ents); e = ue.copy(); rng.shuffle(e)
        cut = int(len(e) * 0.7); tre = set(e[:cut].tolist())
        trm = np.array([en in tre for en in ents]); tem = ~trm
        if len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        tr_rows = [rows[i] for i in np.where(trm)[0]]
        te_rows = [rows[i] for i in np.where(tem)[0]]
        ytr, yte = y[trm], y[tem]

        aP = _auc(_mat_P(tr_rows, vocab), ytr, _mat_P(te_rows, vocab), yte)
        aPR = _auc(_mat_PR(tr_rows, vocab), ytr, _mat_PR(te_rows, vocab), yte)
        # recency-null: permute recency (train and test independently)
        rng2 = np.random.default_rng(100 + seed)
        ov_tr = _recency_null(tr_rows, vocab, rng2)
        ov_te = _recency_null(te_rows, vocab, rng2)
        aPRnull = _auc(_mat_PR(tr_rows, vocab, ov_tr), ytr, _mat_PR(te_rows, vocab, ov_te), yte)
        aGen = _auc(_mat_generic(tr_rows), ytr, _mat_generic(te_rows), yte)

        recvals.append(aPR - aPRnull)
        prgain.append(aPR - aP)
        gengain.append(aGen - 0.5)

    def summ(x):
        x = np.array(x)
        return {"mean": round(float(x.mean()), 3), "min": round(float(x.min()), 3),
                "max": round(float(x.max()), 3), "frac_pos": round(float(np.mean(x > 0)), 2)}
    return {"trace": name, "horizon": hz, "n_windows": len(rows),
            "count_mismatch": int(mism),
            "recency_value": summ(recvals), "PR_minus_P": summ(prgain),
            "generic_proximity_auroc_minus_half": summ(gengain)}


def main():
    out = []
    for name, wp, ep, hz, K in TRACES:
        if not (ROOT / "data/processed" / wp).exists() or not (ROOT / "data/processed" / ep).exists():
            print(f"[{name}] missing files, skip", flush=True); continue
        r = run_trace(name, wp, ep, hz, K)
        out.append(r)
        print(f"[{name} {hz}] n={r['n_windows']} mismatch={r['count_mismatch']}", flush=True)
        print(f"    recency_value = {r['recency_value']}", flush=True)
        print(f"    P+R minus P   = {r['PR_minus_P']}", flush=True)
        print(f"    generic-proximity AUROC-0.5 = {r['generic_proximity_auroc_minus_half']}", flush=True)
    (ROOT / "results/patterns/recency_marked_set.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote results/patterns/recency_marked_set.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
