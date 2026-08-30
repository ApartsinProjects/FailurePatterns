"""W10: repeated entity-disjoint splits and signature stability.

For a paper about statistical validity, one seeded discovery/inference
split is not enough. This runs many splits and reports how stable the
validated catalog is.

  - Azure, Alibaba, BGL, SCANIA: 30 seeded entity-disjoint splits.
  - Kelmarsh (6 turbines): all balanced 3/3 turbine partitions.
  - Penmanshiel (9 turbines): 30 sampled balanced 4/5 turbine partitions.

Per split we mine itemsets on the discovery half, score on the inference
half, and take the BY-significant, precursor-only patterns (prior
terminal_failure tokens excluded). We report per trace:
  - the significant-fraction distribution across splits (median, IQR);
  - each headline signature's selection frequency (fraction of splits in
    which it is BY-significant);
  - mean pairwise Jaccard stability of the BY-significant catalog.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.discovery_inference_split import (
    entity_disjoint_split, score_on_inference_half, by_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets

N_SEED = 20


def _key(itemset):
    return " + ".join(sorted(itemset))


def _has_term(items):
    return any("terminal_failure" in x for x in items)


def _one_split(disc, inf, horizon):
    pats, _ = mine_itemsets(disc, [horizon], min_support=0.05)
    if pats.empty:
        return set(), 0, 0
    scored = score_on_inference_half(inf, pats[["itemset"]])
    scored = scored[~scored["itemset"].apply(_has_term)]
    if scored.empty:
        return set(), 0, 0
    q = by_qvalues(scored["inf_p_value"].to_numpy())
    sig = scored[q < 0.05]
    sig_keys = {_key(list(r["itemset"])) for _, r in sig.iterrows()}
    return sig_keys, int(len(sig)), int(len(scored))


def _jaccard(sets):
    if len(sets) < 2:
        return None
    js = []
    for a, b in combinations(sets, 2):
        u = len(a | b)
        js.append(len(a & b) / u if u else 1.0)
    return round(float(np.mean(js)), 3)


def stability(name, windows_path, horizon, splits):
    wins = pd.read_parquet(windows_path)
    sub = wins[wins["horizon"] == horizon]
    fracs, sig_sets = [], []
    counter = {}
    for disc, inf in splits(sub):
        keys, nsig, ntot = _one_split(disc, inf, horizon)
        if ntot == 0:
            continue
        fracs.append(nsig / ntot)
        sig_sets.append(keys)
        for k in keys:
            counter[k] = counter.get(k, 0) + 1
    n = len(sig_sets)
    top = sorted(counter.items(), key=lambda kv: -kv[1])[:10]
    return {
        "trace": name, "horizon": horizon, "n_splits": n,
        "sig_fraction_median": round(float(np.median(fracs)), 3) if fracs else None,
        "sig_fraction_iqr": [round(float(np.percentile(fracs, 25)), 3),
                             round(float(np.percentile(fracs, 75)), 3)] if fracs else None,
        "mean_pairwise_jaccard": _jaccard(sig_sets),
        "top_signatures_by_selection_frequency": [
            {"signature": k, "selected_in": f"{c}/{n}", "frequency": round(c / max(n, 1), 3)}
            for k, c in top
        ],
    }


def seeded_splits(n):
    def gen(sub):
        for s in range(n):
            yield entity_disjoint_split(sub, discovery_frac=0.5, rng_seed=1000 + s)
    return gen


def turbine_balanced_splits(all_balanced=True, sample=None, seed=0):
    def gen(sub):
        turbines = sorted(sub["entity_id"].unique())
        k = len(turbines) // 2
        combos = list(combinations(turbines, k))
        if not all_balanced and sample and len(combos) > sample:
            rng = np.random.default_rng(seed)
            combos = [combos[i] for i in rng.choice(len(combos), sample, replace=False)]
        for disc_t in combos:
            disc_t = set(disc_t)
            disc = sub[sub["entity_id"].isin(disc_t)]
            inf = sub[~sub["entity_id"].isin(disc_t)]
            if disc["is_failure"].nunique() < 2 or inf["is_failure"].nunique() < 2:
                continue
            yield disc, inf
    return gen


def main() -> int:
    out = []
    big = [
        ("Azure", ROOT / "data/processed/azure_windows.parquet", "last5"),
        ("Alibaba", ROOT / "data/processed/alibaba_windows.parquet", "last3"),
        ("BGL", ROOT / "data/processed/bgl_windows.parquet", "last20"),
        ("SCANIA", ROOT / "data/processed/scania_windows.parquet", "last5"),
    ]
    for name, path, h in big:
        r = stability(name, path, h, seeded_splits(N_SEED))
        out.append(r)
        print(f"[{name} {h}] {r['n_splits']} splits: sig-frac median={r['sig_fraction_median']} "
              f"IQR={r['sig_fraction_iqr']} Jaccard={r['mean_pairwise_jaccard']}", flush=True)

    r = stability("Kelmarsh", ROOT / "data/processed/kelmarsh_windows.parquet", "last5",
                  turbine_balanced_splits(all_balanced=True))
    out.append(r)
    print(f"[Kelmarsh last5] {r['n_splits']} balanced turbine splits: "
          f"sig-frac median={r['sig_fraction_median']} Jaccard={r['mean_pairwise_jaccard']}", flush=True)
    r = stability("Penmanshiel", ROOT / "data/processed/penmanshiel_windows.parquet", "last5",
                  turbine_balanced_splits(all_balanced=False, sample=30))
    out.append(r)
    print(f"[Penmanshiel last5] {r['n_splits']} balanced turbine splits: "
          f"sig-frac median={r['sig_fraction_median']} Jaccard={r['mean_pairwise_jaccard']}", flush=True)

    (ROOT / "results/patterns/repeated_split_stability.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results/patterns/repeated_split_stability.json ({len(out)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
