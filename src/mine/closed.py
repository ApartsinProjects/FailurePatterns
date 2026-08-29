"""Closed-itemset post-filter + Benjamini-Yekutieli FDR correction.

Given a frequent-itemset mining output where every itemset carries a
support count, an itemset I is *closed* iff no strict superset J of I
has the same support. Post-filtering the FP-Growth output to closed
itemsets losslessly compresses the pattern set: every non-closed pattern
is a subset of a closed pattern with the same support, so lift /
odds-ratio / hypergeometric statistics reproduce exactly.

Benjamini-Yekutieli (2001) FDR correction is Benjamini-Hochberg
multiplied by the harmonic number of the number of tests; it is valid
under arbitrary dependence between p-values, which is the honest
assumption for pattern-mining output where nearby patterns share items.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def closed_filter(patterns_df: pd.DataFrame,
                  itemset_col: str = "itemset",
                  support_col: str = "support_failure",
                  tol: float = 1e-9) -> pd.DataFrame:
    """Return the closed subset of a mined patterns frame.

    An itemset row is dropped when it has a strict superset in the frame
    with support within ``tol`` of its own support.
    """
    if patterns_df.empty:
        return patterns_df.copy()
    # Normalise itemsets to frozensets and pre-sort by length.
    df = patterns_df.copy()
    df["_items"] = df[itemset_col].apply(lambda x: frozenset(x))
    df["_size"] = df["_items"].apply(len)
    df = df.sort_values("_size", ascending=False).reset_index(drop=True)

    # Bucket by support (rounded) then by size, so we compare only
    # candidates that could be equal-support subsets.
    keep = np.ones(len(df), dtype=bool)
    # Index rows by size for fast superset lookup.
    rows_by_size = {s: [] for s in df["_size"].unique()}
    for i, r in df.iterrows():
        rows_by_size[r["_size"]].append(i)

    supports = df[support_col].to_numpy()
    for i in range(len(df)):
        if not keep[i]:
            continue
        item_i = df.at[i, "_items"]
        supp_i = supports[i]
        size_i = df.at[i, "_size"]
        # Any strict subset of item_i in the frame with equal support -> drop that subset row
        # But since we go from large to small, we check smaller-size rows.
        # Actually: for CLOSED, we mark a smaller-size row j as non-closed if
        # there is a larger-size row i where item_i ⊃ item_j and supp_i == supp_j.
        # Iterate over rows with smaller size that have supp within tol.
        for smaller_size in range(size_i - 1, 0, -1):
            for j in rows_by_size.get(smaller_size, []):
                if not keep[j]:
                    continue
                if abs(supports[j] - supp_i) <= tol and df.at[j, "_items"].issubset(item_i):
                    keep[j] = False
    out = df[keep].drop(columns=["_items", "_size"]).reset_index(drop=True)
    return out


def by_correction(pvals: np.ndarray, alpha: float = 0.05
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Yekutieli FDR correction (valid under arbitrary dependence).
    Returns (q_values, reject_flags), same shape as pvals.
    """
    m = len(pvals)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, m + 1)
    c_m = float(np.sum(1.0 / ranks))          # harmonic number
    q_raw = ranked * m * c_m / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qvals = np.empty_like(pvals, dtype=float)
    qvals[order] = q_sorted
    return qvals, qvals < alpha


def score_bh_and_by(patterns_df: pd.DataFrame,
                    pvalue_col: str = "p_value",
                    alpha: float = 0.05) -> pd.DataFrame:
    """Add both BH and BY q-value columns."""
    if patterns_df.empty:
        return patterns_df.copy()
    pv = patterns_df[pvalue_col].to_numpy(dtype=float)

    # BH
    m = len(pv)
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1)
    q_raw = pv[order] * m / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_bh = np.empty_like(pv, dtype=float)
    q_bh[order] = q_sorted

    q_by, _ = by_correction(pv, alpha=alpha)

    out = patterns_df.copy()
    out["q_bh"] = q_bh
    out["q_by"] = q_by
    out["significant_bh_005"] = q_bh < alpha
    out["significant_by_005"] = q_by < alpha
    return out
