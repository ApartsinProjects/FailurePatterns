"""Corrected power calibration for the capacity-matched order test.

The first attempt injected order into two existing frequent tokens, but they
co-occur in only ~14% of positive windows, starving the test. Here we inject a
SYNTHETIC order-only signal that is guaranteed to co-occur and to be pure order:

  in a fraction `frac` of BOTH positive and control windows, append two new
  tokens INJ_A, INJ_B. In positives -> A before B; in controls -> B before A.

Presence and multiplicity are then IDENTICAL across classes (both carry one A and
one B), so only ORDER can separate them. The precedence feature (A-before-B) must
detect it, and order_value = AUROC(real) - AUROC(count-preserving-shuffle) must
RISE with frac. If it does, the detector provably has power, and a near-zero
order_value on real traces is a genuine ("power-calibrated") negative.
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
A, B = "INJ:A", "INJ:B"


def _items(r):
    return [f"{t}:{s}" for t, s in zip(r["event_type_seq"], r["event_subtype_seq"])]


def _precedence_feats(rows, toks):
    pairs = [(x, y) for i, x in enumerate(toks) for y in toks[i + 1:]]
    pidx = {p: j for j, p in enumerate(pairs)}
    X = np.zeros((len(rows), len(pairs)))
    for i, its in enumerate(rows):
        first = {}
        for pos, it in enumerate(its):
            if it in toks and it not in first:
                first[it] = pos
        for (x, y), j in pidx.items():
            if x in first and y in first:
                X[i, j] = 1.0 if first[x] < first[y] else 0.0
    return X


def _auc(Xtr, ytr, Xte, yte):
    if Xtr.shape[1] == 0 or len(np.unique(ytr)) < 2:
        return 0.5
    sc = StandardScaler().fit(Xtr)
    c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1])


def _run(sub, frac):
    ents = sub["entity_id"].unique()
    ovals = []
    for seed in range(NSPLIT):
        rng = np.random.default_rng(seed)
        e = ents.copy(); rng.shuffle(e)
        cut = int(len(e) * 0.7); tre = set(e[:cut].tolist())
        tr = sub[sub["entity_id"].isin(tre)]; te = sub[~sub["entity_id"].isin(tre)]
        if tr["is_failure"].nunique() < 2 or te["is_failure"].nunique() < 2:
            continue
        rj = np.random.default_rng(500 + seed)

        def build(df):
            rows, ys = [], []
            for _, r in df.iterrows():
                its = _items(r); lab = int(r["is_failure"])
                if rj.random() < frac:
                    its = list(its) + ([A, B] if lab == 1 else [B, A])
                rows.append(its); ys.append(lab)
            return rows, np.array(ys)

        tri, ytr = build(tr); tei, yte = build(te)
        # count-preserving shuffle (destroys order, keeps multiset incl. INJ tokens)
        tri_s = [list(x) for x in tri]; tei_s = [list(x) for x in tei]
        for x in tri_s: rj.shuffle(x)
        for x in tei_s: rj.shuffle(x)

        # precedence over top tokens + forced-in injected pair
        from collections import Counter
        tc = Counter(it for its in tri for it in its)
        toks = [t for t, _ in tc.most_common(20)]
        for t in (A, B):
            if t not in toks:
                toks.append(t)
        ar = _auc(_precedence_feats(tri, toks), ytr, _precedence_feats(tei, toks), yte)
        ash = _auc(_precedence_feats(tri_s, toks), ytr, _precedence_feats(tei_s, toks), yte)
        ovals.append(ar - ash)
    return (round(float(np.mean(ovals)), 3), round(float(np.min(ovals)), 3),
            round(float(np.max(ovals)), 3), round(float(np.mean(np.array(ovals) > 0)), 2))


def main():
    w = pd.read_parquet(ROOT / "data/processed/sepsis_trend_windows.parquet")
    sub = w[w["horizon"] == "last10"]
    print(f"[Power calibration on Sepsis last10, synthetic order-only injection]", flush=True)
    out = {}
    for frac in [0.0, 0.1, 0.2, 0.4, 0.8]:
        ov = _run(sub, frac)
        out[str(frac)] = {"precedence_order_value": list(ov)}
        print(f"  frac={frac}: precedence order_value (mean,min,max,frac>0) = {ov}", flush=True)
    (ROOT / "results/patterns/order_power_calibration.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    # validation verdict
    mono = out["0.8"]["precedence_order_value"][0] > out["0.0"]["precedence_order_value"][0] + 0.05
    print(f"\nDETECTOR HAS POWER: {mono} (order_value rises from frac=0 to frac=0.8)", flush=True)
    print("Wrote results/patterns/order_power_calibration.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
