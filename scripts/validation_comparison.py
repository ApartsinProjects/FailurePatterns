"""W18: what does the discovery/inference validation control that naive
same-sample testing does not?

On Azure (last5), for the SAME candidate set mined on the training data,
we compare three validation strategies and measure both the number of
discoveries on the REAL labels and the number of FALSE discoveries when
the labels are permuted (no true signal, so every discovery is false):

  1. naive same-sample BH   : mine and test on the same windows, BH q<0.05.
  2. discovery/inference split (ours): candidates from the discovery half,
     tested on the disjoint inference half, BH q<0.05.
  3. Westfall-Young minP (FWER): permutation-adjusted p-values on the same
     sample, family-wise error controlled at 0.05.

A validation strategy that controls error reports ~0 false discoveries on
permuted labels. The naive strategy does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.discovery_inference_split import (
    entity_disjoint_split, score_on_inference_half, bh_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets

SEED = 20260828
WY_PERM = 500
N_LABEL_PERM = 20  # independent label permutations to average false discoveries


def _items(row):
    return {f"{t}:{s}" for t, s in zip(row["event_type_seq"], row["event_subtype_seq"])}


def _hyper_p(hit_f, hit_c, n_f, n_c):
    K = hit_f + hit_c; N = n_f + n_c
    if K == 0 or K == N or n_f == 0:
        return 1.0
    return float(hypergeom.sf(hit_f - 1, N, K, n_f))


def naive_bh(windows, patterns):
    items = [_items(r) for _, r in windows.iterrows()]
    y = windows["is_failure"].to_numpy().astype(bool)
    n_f, n_c = int(y.sum()), int((~y).sum())
    pv = []
    for _, r in patterns.iterrows():
        pat = set(r["itemset"])
        pres = np.array([pat.issubset(s) for s in items])
        pv.append(_hyper_p(int((pres & y).sum()), int((pres & ~y).sum()), n_f, n_c))
    q = bh_qvalues(np.array(pv))
    return int((q < 0.05).sum())


def wy_minp(windows, patterns, rng):
    """Westfall-Young minP FWER-adjusted discovery count on this sample."""
    items = [_items(r) for _, r in windows.iterrows()]
    y = windows["is_failure"].to_numpy().astype(bool)
    n_f, n_c = int(y.sum()), int((~y).sum())
    pres = np.array([[set(r["itemset"]).issubset(s) for s in items]
                     for _, r in patterns.iterrows()])  # P x N
    obs_p = np.array([_hyper_p(int((pres[j] & y).sum()), int((pres[j] & ~y).sum()), n_f, n_c)
                      for j in range(len(patterns))])
    minp_null = []
    for _ in range(WY_PERM):
        yp = rng.permutation(y)
        nf = int(yp.sum())
        ps = [_hyper_p(int((pres[j] & yp).sum()), int((pres[j] & ~yp).sum()), nf, n_c)
              for j in range(len(patterns))]
        minp_null.append(min(ps))
    minp_null = np.array(minp_null)
    adj = np.array([(minp_null <= p).mean() for p in obs_p])
    return int((adj < 0.05).sum())


def main() -> int:
    wins = pd.read_parquet(ROOT / "data/processed/azure_windows.parquet")
    sub = wins[wins["horizon"] == "last5"]
    rng = np.random.default_rng(SEED)

    # candidate set mined on a discovery half (fixed)
    disc, inf = entity_disjoint_split(sub, discovery_frac=0.5, rng_seed=SEED)
    pats, _ = mine_itemsets(disc, ["last5"], min_support=0.05)
    n_cand = len(pats)

    # ---- REAL labels ----
    real = {
        "naive_bh": naive_bh(sub, pats),
        "split_bh": int((score_on_inference_half(inf, pats[["itemset"]]).pipe(
            lambda d: bh_qvalues(d["inf_p_value"].to_numpy())) < 0.05).sum()),
        "wy_minp": wy_minp(sub, pats, rng),
    }

    # ---- PERMUTED labels (no signal): average false discoveries ----
    false_disc = {"naive_bh": [], "split_bh": [], "wy_minp": []}
    for _ in range(N_LABEL_PERM):
        perm = sub.copy()
        perm["is_failure"] = rng.permutation(perm["is_failure"].to_numpy())
        d2, i2 = entity_disjoint_split(perm, discovery_frac=0.5, rng_seed=int(rng.integers(1e6)))
        p2, _ = mine_itemsets(d2, ["last5"], min_support=0.05)
        if p2.empty:
            continue
        false_disc["naive_bh"].append(naive_bh(perm, p2))
        sc = score_on_inference_half(i2, p2[["itemset"]])
        false_disc["split_bh"].append(int((bh_qvalues(sc["inf_p_value"].to_numpy()) < 0.05).sum()))
        false_disc["wy_minp"].append(wy_minp(perm, p2, rng))

    summary = {
        "trace": "Azure", "horizon": "last5",
        "n_candidate_patterns": n_cand,
        "n_label_permutations": N_LABEL_PERM,
        "discoveries_real_labels": real,
        "mean_false_discoveries_permuted_labels": {
            k: round(float(np.mean(v)), 2) if v else None for k, v in false_disc.items()},
        "max_false_discoveries_permuted_labels": {
            k: int(np.max(v)) if v else None for k, v in false_disc.items()},
    }
    (ROOT / "results/patterns/validation_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
