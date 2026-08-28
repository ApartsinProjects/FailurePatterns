"""Phase 5: per-pattern p-values and BH correction.

For each mined pattern (itemset or sequence) we already recorded the
number of failure windows and control windows that CONTAIN the
pattern. Under the null of random label permutation with the pattern
hit-set fixed, the number of pattern-hits landing in the failure class
is hypergeometric with population size N_F + N_C, K = total hits,
n = N_F. The upper-tail probability of the observed hit count IS the
label-permutation p-value; we do not need Monte Carlo.

Benjamini-Hochberg correction is applied per (horizon x pattern-class)
to give FDR-controlled q-values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ALPHA = 0.05


def _one_sided_p_upper(hit_f: int, hit_c: int, n_f: int, n_c: int) -> float:
    """P(X >= hit_f) under H0 where X ~ Hypergeom(N=n_f+n_c, K=hit_f+hit_c, n=n_f)."""
    K = hit_f + hit_c
    N = n_f + n_c
    if K == 0 or K == N or n_f == 0 or n_c == 0:
        return 1.0
    # sf(k) = P(X > k); we want P(X >= hit_f) = sf(hit_f - 1)
    return float(hypergeom.sf(hit_f - 1, N, K, n_f))


def _bh(pvals: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (qvals, reject) both same-length as pvals."""
    m = len(pvals)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, m + 1)
    q_raw = ranked * m / ranks
    # Enforce monotonicity (running min from the right).
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qvals = np.empty_like(pvals, dtype=float)
    qvals[order] = q_sorted
    reject = qvals < alpha
    return qvals, reject


def _class_counts_from_stats(patterns_path: Path, stats_path: Path
                             ) -> dict[str, tuple[int, int]]:
    """Return per-horizon (n_failure_windows, n_control_windows)."""
    # We look these up from the windows parquet since the mining stats
    # JSON does not record raw class sizes per horizon. Simplest: pass
    # the windows parquet path.
    raise NotImplementedError


def score_patterns(
    patterns_df: pd.DataFrame,
    class_counts: dict[str, tuple[int, int]],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Add p-value, q-value, and significance flag columns to a
    patterns DataFrame.

    ``class_counts`` maps horizon -> (n_failure_windows, n_control_windows).
    """
    if patterns_df.empty:
        return patterns_df.assign(
            p_value=[], q_value=[], significant_at_q_005=[],
        )
    out = patterns_df.copy()
    p_col = np.ones(len(out), dtype=float)
    for h in out["horizon"].unique():
        mask = out["horizon"].values == h
        n_f, n_c = class_counts.get(h, (0, 0))
        hit_f = out.loc[mask, "n_failure"].to_numpy()
        hit_c = out.loc[mask, "n_control"].to_numpy()
        p_col[mask] = np.array([
            _one_sided_p_upper(int(hf), int(hc), n_f, n_c)
            for hf, hc in zip(hit_f, hit_c)
        ])
    out["p_value"] = p_col

    # BH per horizon x class (pattern kind is uniform inside one call).
    q_col = np.ones(len(out), dtype=float)
    sig_col = np.zeros(len(out), dtype=bool)
    for h in out["horizon"].unique():
        mask = out["horizon"].values == h
        q, r = _bh(p_col[mask], alpha=alpha)
        q_col[mask] = q
        sig_col[mask] = r
    out["q_value"] = q_col
    out["significant_at_q_005"] = sig_col
    return out


@dataclass
class SignificanceSummary:
    dataset: str
    alpha: float
    itemsets_by_horizon: dict[str, dict] = field(default_factory=dict)
    sequences_by_horizon: dict[str, dict] = field(default_factory=dict)


def summarize(kind: str, scored: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for h, g in scored.groupby("horizon"):
        out[h] = {
            "n_patterns": int(len(g)),
            "n_significant": int(g["significant_at_q_005"].sum()),
            "min_q": float(g["q_value"].min()) if len(g) else float("nan"),
            "median_q": float(g["q_value"].median()) if len(g) else float("nan"),
        }
    return out


def run(
    dataset: str,
    windows_parquet: Path,
    itemsets_parquet: Path,
    sequences_parquet: Path,
    out_dir: Path,
) -> SignificanceSummary:
    windows = pd.read_parquet(windows_parquet)
    class_counts: dict[str, tuple[int, int]] = {}
    for h, g in windows.groupby("horizon"):
        n_f = int(g["is_failure"].sum())
        n_c = int(len(g) - n_f)
        class_counts[h] = (n_f, n_c)

    itemsets = pd.read_parquet(itemsets_parquet)
    sequences = pd.read_parquet(sequences_parquet)

    itemsets_scored = score_patterns(itemsets, class_counts)
    sequences_scored = score_patterns(sequences, class_counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    itemsets_scored.to_parquet(
        out_dir / f"{dataset}_itemsets_significance.parquet", index=False,
    )
    sequences_scored.to_parquet(
        out_dir / f"{dataset}_sequences_significance.parquet", index=False,
    )

    summary = SignificanceSummary(
        dataset=dataset, alpha=ALPHA,
        itemsets_by_horizon=summarize("itemset", itemsets_scored),
        sequences_by_horizon=summarize("sequence", sequences_scored),
    )
    with (out_dir / f"{dataset}_significance_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2, default=str)
    return summary
