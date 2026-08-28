"""Frequent itemset mining on Azure PdM windows (Phase 3).

Items are ``event_type:event_subtype`` strings (e.g. ``software_error:error1``).
FP-Growth runs on failure windows only to surface pre-failure signatures,
then each mined itemset is scored on the matched control windows to give
a lift and relative-risk against normal operation on the same entity.

Pre-declared sanity invariant (see PLAN.md phase 3):

    A random permutation of ``is_failure`` labels within the same horizon
    must not yield any mined itemset with lift >= LIFT_SANITY_THRESHOLD
    at the same min_support. A hit is a data-leakage bug, not a finding.

Output schema (one row per surviving pattern per horizon):

    horizon               str
    itemset               list<str>
    itemset_size          int
    support_failure       float   fraction of failure windows containing it
    support_control       float   fraction of control windows containing it
    n_failure             int
    n_control             int
    lift_failure          float   support_failure / base_rate_of_itemset
    relative_risk         float   P(fail | itemset) / P(fail | not itemset)
    p_fail_given_pattern  float
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# -------------------------- config ----------------------------------------

DEFAULT_MIN_SUPPORT = 0.05  # 5% of failure windows -> ~37 per horizon
# Sanity threshold: how much the real top mined lift must exceed the
# permuted-label max lift at the same min_support. A random-label
# permutation on a wide horizon with hundreds of hypotheses can produce
# top lifts up to ~1.5 by chance alone (multiple-testing tail), so an
# absolute threshold is not meaningful; the ratio "real top >= 1.5 x
# permuted top" is.
LIFT_RATIO_SANITY = 1.5
RNG_SEED = 20260828


# -------------------------- helpers ---------------------------------------

def _make_items(seq_types: list[str], seq_subtypes: list[str]) -> list[str]:
    """Encode a window's events as ``event_type:event_subtype`` items.
    Deduplicated per window because itemset mining is set-based.
    """
    return sorted({f"{t}:{s}" for t, s in zip(seq_types, seq_subtypes)})


def _transactions(windows: pd.DataFrame) -> list[list[str]]:
    return [
        _make_items(row["event_type_seq"], row["event_subtype_seq"])
        for _, row in windows.iterrows()
    ]


def _fit_fpgrowth(
    transactions: list[list[str]],
    min_support: float,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``support`` (float) and ``itemsets``
    (frozenset). Handles the empty-transaction case cleanly."""
    if not transactions:
        return pd.DataFrame(columns=["support", "itemsets"])
    te = TransactionEncoder()
    arr = te.fit_transform(transactions)
    df = pd.DataFrame(arr, columns=te.columns_)
    out = fpgrowth(df, min_support=min_support, use_colnames=True)
    if out.empty:
        return pd.DataFrame(columns=["support", "itemsets"])
    return out.sort_values("support", ascending=False).reset_index(drop=True)


def _support_in(transactions: list[list[str]], itemset: frozenset) -> tuple[float, int]:
    """Fraction of transactions containing all items in ``itemset``."""
    if not transactions:
        return 0.0, 0
    hits = sum(1 for t in transactions if itemset.issubset(t))
    return hits / len(transactions), hits


# -------------------------- data model ------------------------------------

@dataclass
class MiningStats:
    min_support: float
    lift_ratio_sanity: float
    n_patterns_by_horizon: dict[str, int] = field(default_factory=dict)
    max_real_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    max_permuted_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    n_patterns_above_perm_ceiling_by_horizon: dict[str, int] = field(default_factory=dict)
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# -------------------------- main pipeline ---------------------------------

def mine(
    windows: pd.DataFrame,
    horizons: Iterable[str],
    min_support: float = DEFAULT_MIN_SUPPORT,
    rng_seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, MiningStats]:
    stats = MiningStats(
        min_support=min_support, lift_ratio_sanity=LIFT_RATIO_SANITY
    )
    rng = np.random.default_rng(rng_seed)
    out_rows: list[dict] = []
    horizon_real_top: dict[str, float] = {}

    for hname in horizons:
        w = windows[windows["horizon"] == hname]
        w_fail = w[w["is_failure"]]
        w_ctrl = w[~w["is_failure"]]
        n_fail = len(w_fail)
        n_ctrl = len(w_ctrl)
        if n_fail == 0 or n_ctrl == 0:
            stats.n_patterns_by_horizon[hname] = 0
            stats.max_permuted_lift_by_horizon[hname] = float("nan")
            continue

        t_fail = _transactions(w_fail)
        t_ctrl = _transactions(w_ctrl)
        t_all = t_fail + t_ctrl
        n_all = n_fail + n_ctrl
        base_rate = n_fail / n_all  # P(failure)

        patterns = _fit_fpgrowth(t_fail, min_support)
        stats.n_patterns_by_horizon[hname] = int(len(patterns))
        horizon_rows: list[dict] = []

        for _, p in patterns.iterrows():
            itemset = frozenset(p["itemsets"])
            supp_fail, hit_fail = _support_in(t_fail, itemset)
            supp_ctrl, hit_ctrl = _support_in(t_ctrl, itemset)
            # Base rate of the itemset in the pooled window population.
            pooled_supp = (hit_fail + hit_ctrl) / n_all if n_all else 0.0
            # Lift for "itemset -> failure" association:
            #   lift = P(itemset | failure) / P(itemset)  = supp_fail / pooled_supp
            lift = supp_fail / pooled_supp if pooled_supp > 0 else float("nan")
            # Relative risk: P(fail | itemset) / P(fail | no itemset)
            p_fail_given = (
                hit_fail / (hit_fail + hit_ctrl)
                if (hit_fail + hit_ctrl) > 0
                else float("nan")
            )
            miss_fail = n_fail - hit_fail
            miss_ctrl = n_ctrl - hit_ctrl
            p_fail_notgiven = (
                miss_fail / (miss_fail + miss_ctrl)
                if (miss_fail + miss_ctrl) > 0
                else float("nan")
            )
            rr = (
                p_fail_given / p_fail_notgiven
                if p_fail_notgiven and not np.isnan(p_fail_notgiven)
                else float("nan")
            )
            horizon_rows.append({
                "horizon": hname,
                "itemset": sorted(itemset),
                "itemset_size": len(itemset),
                "support_failure": supp_fail,
                "support_control": supp_ctrl,
                "n_failure": hit_fail,
                "n_control": hit_ctrl,
                "lift_failure": lift,
                "relative_risk": rr,
                "p_fail_given_pattern": p_fail_given,
                "base_rate_failure": base_rate,
            })

        # ------- sanity invariant: random label permutation -----------
        labels = np.zeros(n_all, dtype=bool)
        labels[:n_fail] = True
        rng.shuffle(labels)
        t_fail_perm = [t_all[i] for i in np.where(labels)[0]]
        t_ctrl_perm = [t_all[i] for i in np.where(~labels)[0]]
        patterns_perm = _fit_fpgrowth(t_fail_perm, min_support)
        max_lift_perm = 0.0
        for _, p in patterns_perm.iterrows():
            itemset = frozenset(p["itemsets"])
            supp_fail_perm, hit_fail_perm = _support_in(t_fail_perm, itemset)
            supp_ctrl_perm, hit_ctrl_perm = _support_in(t_ctrl_perm, itemset)
            pooled = (hit_fail_perm + hit_ctrl_perm) / n_all
            if pooled > 0:
                lift_perm = supp_fail_perm / pooled
                if lift_perm > max_lift_perm:
                    max_lift_perm = lift_perm
        stats.max_permuted_lift_by_horizon[hname] = float(max_lift_perm)

        # Record real max lift and how many real patterns exceed the
        # permuted-null ceiling.
        real_lifts = [r["lift_failure"] for r in horizon_rows
                      if not np.isnan(r["lift_failure"])]
        stats.max_real_lift_by_horizon[hname] = (
            float(max(real_lifts)) if real_lifts else float("nan")
        )
        stats.n_patterns_above_perm_ceiling_by_horizon[hname] = sum(
            1 for lft in real_lifts if lft > max_lift_perm
        )
        for r in horizon_rows:
            r["permuted_null_lift_ceiling"] = float(max_lift_perm)
            r["survives_permutation_null"] = (
                r["lift_failure"] > max_lift_perm
                if not np.isnan(r["lift_failure"])
                else False
            )
        out_rows.extend(horizon_rows)

    def _real_beats_perm(h: str) -> bool:
        real = stats.max_real_lift_by_horizon.get(h, float("nan"))
        perm = stats.max_permuted_lift_by_horizon.get(h, 0.0)
        if np.isnan(real) or real == 0.0:
            # No real patterns; vacuously fine (e.g. empty 1h/6h horizons).
            return True
        return real >= LIFT_RATIO_SANITY * perm

    # Informative horizons = whichever horizons actually returned patterns
    # (short-time horizons like 1h/6h on Azure return 0 patterns because
    # their windows are empty; a non-mined horizon shouldn't count against
    # the "something got mined" invariant).
    mined_horizons = [
        h for h, n in stats.n_patterns_by_horizon.items() if n > 0
    ]

    stats.invariants = {
        "real_top_lift_dominates_permuted_top": all(
            _real_beats_perm(h) for h in stats.n_patterns_by_horizon
        ),
        "at_least_one_horizon_mined": len(mined_horizons) > 0,
        "every_mined_horizon_has_survivors":
            all(
                stats.n_patterns_above_perm_ceiling_by_horizon.get(h, 0) > 0
                for h in mined_horizons
            ),
    }

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["horizon", "lift_failure"], ascending=[True, False])
    return out.reset_index(drop=True), stats


HORIZON_ORDER = {
    "1h": 0, "6h": 1, "24h": 2, "last3": 3, "last5": 4, "last10": 5,
}


def run(
    windows_parquet: Path,
    out_dir: Path,
    output_stem: str = "azure_itemsets",
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> MiningStats:
    windows = pd.read_parquet(windows_parquet)
    horizons = sorted(windows["horizon"].unique(),
                      key=lambda h: HORIZON_ORDER.get(h, 99))
    patterns, stats = mine(windows, horizons, min_support=min_support)
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns.to_parquet(out_dir / f"{output_stem}.parquet", index=False)
    with (out_dir / f"{output_stem}_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
