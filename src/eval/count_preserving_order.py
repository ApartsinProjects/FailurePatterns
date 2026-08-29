"""Count-preserving order comparator (W4 from DAMI review).

The paper's original `order_gain = sequence_lift - itemset_lift` compared a
sequence like `M M M` against its ITEMSET counterpart `{M}`, which
collapses three occurrences to one presence. That conflates order with
event MULTIPLICITY.

This module computes an order-only comparator: for each sequence pattern
S with observed sequence lift L(S), permute the ORDER of events within
each window while preserving the EXACT event multiset. Then re-score
S against the shuffled windows. The shuffled lift is the count-
preserving null; the residual (L(S) - shuffled_lift) is the pure order
effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _seq_supports(sequence: list[str], transactions: list[list[str]]) -> int:
    n = 0
    for tx in transactions:
        i = 0
        for item in tx:
            if item == sequence[i]:
                i += 1
                if i == len(sequence):
                    n += 1
                    break
    return n


def count_preserving_lift(
    sequence: list[str],
    fail_seqs: list[list[str]],
    ctrl_seqs: list[list[str]],
    rng: np.random.Generator,
    n_shuffles: int = 50,
) -> tuple[float, float, float]:
    """Return (real_lift, mean_count_preserving_lift, order_effect).

    Every window in fail and control is shuffled uniformly (permutation
    of the sequence indices), preserving the exact event multiset.
    Then the sequence's support is recounted on the shuffled corpora.
    Averaged over n_shuffles."""
    n_f = len(fail_seqs)
    n_c = len(ctrl_seqs)
    n_all = n_f + n_c
    real_hit_f = _seq_supports(sequence, fail_seqs)
    real_hit_c = _seq_supports(sequence, ctrl_seqs)
    real_pooled = (real_hit_f + real_hit_c) / n_all if n_all else 0
    real_lift = (real_hit_f / n_f) / real_pooled if real_pooled > 0 else float("nan")

    shuffled_lifts = []
    for _ in range(n_shuffles):
        # Multiset-preserving shuffle: numpy.random.permutation of each list.
        sh_f = [list(rng.permutation(np.array(s, dtype=object))) for s in fail_seqs]
        sh_c = [list(rng.permutation(np.array(s, dtype=object))) for s in ctrl_seqs]
        hit_f = _seq_supports(sequence, sh_f)
        hit_c = _seq_supports(sequence, sh_c)
        pooled = (hit_f + hit_c) / n_all if n_all else 0
        if pooled > 0:
            shuffled_lifts.append((hit_f / n_f) / pooled)
    mean_shuf = float(np.mean(shuffled_lifts)) if shuffled_lifts else float("nan")
    order_effect = real_lift - mean_shuf if not (np.isnan(real_lift) or np.isnan(mean_shuf)) else float("nan")
    return real_lift, mean_shuf, order_effect


def score_top_sequences(
    windows: pd.DataFrame,
    patterns_df: pd.DataFrame,
    top_k: int = 30,
    n_shuffles: int = 30,
    rng_seed: int = 20260828,
) -> pd.DataFrame:
    """Compute count-preserving order effect for the top-K sequences by lift."""
    rng = np.random.default_rng(rng_seed)
    fail_w = windows[windows["is_failure"]]
    ctrl_w = windows[~windows["is_failure"]]

    def _seq(row) -> list[str]:
        return [f"{t}:{s}" for t, s in
                zip(row["event_type_seq"], row["event_subtype_seq"])]
    fail_seqs = [_seq(r) for _, r in fail_w.iterrows()]
    ctrl_seqs = [_seq(r) for _, r in ctrl_w.iterrows()]

    top = patterns_df.sort_values("lift_failure", ascending=False).head(top_k)
    rows = []
    for _, r in top.iterrows():
        pat = list(r["sequence"])
        if len(pat) < 2:
            continue
        real, shuf, effect = count_preserving_lift(pat, fail_seqs, ctrl_seqs,
                                                   rng, n_shuffles=n_shuffles)
        rows.append({
            "sequence": pat,
            "sequence_length": len(pat),
            "real_lift": real,
            "count_preserving_shuffle_lift": shuf,
            "order_effect": effect,
            "itemset_lift_from_mine": float(r["itemset_lift_failure"])
                if "itemset_lift_failure" in r else float("nan"),
            "naive_order_gain": float(r["order_gain"]) if "order_gain" in r else float("nan"),
        })
    return pd.DataFrame(rows)
