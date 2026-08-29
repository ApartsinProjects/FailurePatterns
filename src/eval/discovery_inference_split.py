"""Discovery / inference split infrastructure (W1 from DAMI review).

The blocking issue: mining patterns from failure windows and then
computing hypergeometric p-values on the SAME windows yields
post-selection-invalid p-values. BH/BY correction on invalid marginal
p-values does not fix them.

Fix: split each trace's training set into two disjoint halves.
- Discovery half: run FP-Growth / PrefixSpan; produce a candidate
  pattern universe C. Labels ARE used to select C.
- Inference half: for every pattern P in C, compute exact one-sided
  hypergeometric p-value on the inference half's case/control hit
  counts. These p-values are marginally valid because the candidate
  selection did not touch the inference half.

Then BH or BY corrects the family {p(P) : P in C} honestly.

Entity-disjoint splitting (by entity_id, not by window) is used so a
truck / machine / job / rack cannot appear in both halves — otherwise
the inference sample re-uses information from the discovery sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def entity_disjoint_split(
    windows: pd.DataFrame,
    rng_seed: int = 20260828,
    discovery_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (discovery_windows, inference_windows) with disjoint entity_id."""
    rng = np.random.default_rng(rng_seed)
    ent_ids = windows["entity_id"].unique()
    n_disc = int(len(ent_ids) * discovery_frac)
    idx = rng.permutation(len(ent_ids))
    disc = set(ent_ids[idx[:n_disc]])
    inf = set(ent_ids[idx[n_disc:]])
    return (
        windows[windows["entity_id"].isin(disc)].reset_index(drop=True),
        windows[windows["entity_id"].isin(inf)].reset_index(drop=True),
    )


def hypergeom_p_upper(hit_f: int, hit_c: int, n_f: int, n_c: int) -> float:
    K = hit_f + hit_c
    N = n_f + n_c
    if K == 0 or K == N or n_f == 0 or n_c == 0:
        return 1.0
    return float(hypergeom.sf(hit_f - 1, N, K, n_f))


def score_on_inference_half(
    inference_windows: pd.DataFrame,
    patterns: pd.DataFrame,
    itemset_col: str = "itemset",
) -> pd.DataFrame:
    """For each pattern from the discovery half, compute case/control hit
    counts and the exact hypergeometric p-value on the inference half."""
    fail_w = inference_windows[inference_windows["is_failure"]]
    ctrl_w = inference_windows[~inference_windows["is_failure"]]
    n_f = int(len(fail_w))
    n_c = int(len(ctrl_w))

    def _items(row) -> set:
        return {f"{t}:{s}" for t, s in
                zip(row["event_type_seq"], row["event_subtype_seq"])}

    fail_items = [_items(r) for _, r in fail_w.iterrows()]
    ctrl_items = [_items(r) for _, r in ctrl_w.iterrows()]

    out_rows = []
    for _, r in patterns.iterrows():
        pat = set(r[itemset_col])
        hit_f = sum(1 for s in fail_items if pat.issubset(s))
        hit_c = sum(1 for s in ctrl_items if pat.issubset(s))
        p = hypergeom_p_upper(hit_f, hit_c, n_f, n_c)
        supp_f = hit_f / n_f if n_f else 0.0
        supp_c = hit_c / n_c if n_c else 0.0
        pooled = (hit_f + hit_c) / (n_f + n_c) if (n_f + n_c) else 0.0
        lift = supp_f / pooled if pooled > 0 else float("nan")
        out_rows.append({
            "itemset": r[itemset_col],
            "inf_hit_f": hit_f, "inf_hit_c": hit_c,
            "inf_n_f": n_f, "inf_n_c": n_c,
            "inf_supp_f": supp_f, "inf_supp_c": supp_c,
            "inf_lift": lift,
            "inf_p_value": p,
        })
    return pd.DataFrame(out_rows)


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    if m == 0: return pvals
    order = np.argsort(pvals)
    q_raw = pvals[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q = np.empty_like(pvals, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def by_qvalues(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    if m == 0: return pvals
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(pvals)
    q_raw = pvals[order] * m * c_m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q = np.empty_like(pvals, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q
