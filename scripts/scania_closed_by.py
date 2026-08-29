"""Rerun SCANIA risk-set mining with closed-itemset post-filter and BY.

Loads the existing scania_riskset_patterns.parquet, filters to closed
itemsets (lossless dedup), computes exact one-sided hypergeometric p on
the case/control counts already present, applies BOTH BH and BY FDR
corrections. Reports the true predictive fraction after de-redundancy.
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

from src.mine.closed import closed_filter, score_bh_and_by


def hypergeom_p_upper(hit_f, hit_c, n_f, n_c):
    K = hit_f + hit_c
    N = n_f + n_c
    if K == 0 or K == N or n_f == 0 or n_c == 0:
        return 1.0
    return float(hypergeom.sf(hit_f - 1, N, K, n_f))


def main() -> int:
    patterns = pd.read_parquet(ROOT / "results" / "patterns" / "scania_riskset_patterns.parquet")
    windows = pd.read_parquet(ROOT / "results" / "patterns" / "scania_riskset_windows.parquet")
    n_cases = int(windows["is_failure"].sum())
    n_controls = int((~windows["is_failure"]).sum())

    print(f"Input: {len(patterns)} patterns, n_cases={n_cases}, n_controls={n_controls}")

    # Add hypergeometric p-values
    p = np.array([
        hypergeom_p_upper(int(r["n_failure"]), int(r["n_control"]), n_cases, n_controls)
        for _, r in patterns.iterrows()
    ])
    patterns["p_value"] = p
    print(f"Median p: {np.median(p):.3e}   min p: {np.min(p):.3e}")

    # Closed filter (lossless dedup)
    closed = closed_filter(patterns, itemset_col="itemset",
                           support_col="support_failure")
    print(f"Closed itemsets: {len(closed)} / {len(patterns)} "
          f"({100*len(closed)/len(patterns):.1f}%)")

    # BH + BY on closed set
    closed_scored = score_bh_and_by(closed, pvalue_col="p_value", alpha=0.05)
    n_bh = int(closed_scored["significant_bh_005"].sum())
    n_by = int(closed_scored["significant_by_005"].sum())
    print(f"Closed patterns significant at q_BH < 0.05: {n_bh}")
    print(f"Closed patterns significant at q_BY < 0.05: {n_by}")
    print()

    # Original (non-closed) BH+BY for comparison
    orig_scored = score_bh_and_by(patterns, pvalue_col="p_value", alpha=0.05)
    print(f"BEFORE closed filter: BH sig = {int(orig_scored['significant_bh_005'].sum())}, "
          f"BY sig = {int(orig_scored['significant_by_005'].sum())}")

    # Save
    out_dir = ROOT / "results" / "patterns"
    closed_scored.to_parquet(out_dir / "scania_riskset_closed_patterns.parquet",
                             index=False)

    summary = {
        "n_input_patterns": int(len(patterns)),
        "n_closed_patterns": int(len(closed)),
        "compression_ratio": float(len(closed) / len(patterns)),
        "n_closed_significant_bh_005": n_bh,
        "n_closed_significant_by_005": n_by,
        "n_orig_significant_bh_005": int(orig_scored["significant_bh_005"].sum()),
        "n_orig_significant_by_005": int(orig_scored["significant_by_005"].sum()),
        "top10_by_mh_or_closed": closed_scored.sort_values("mh_or", ascending=False).head(10)[
            ["itemset", "mh_or", "mh_or_ci_low", "mh_or_ci_high",
             "n_failure", "n_control", "p_value", "q_bh", "q_by",
             "significant_bh_005", "significant_by_005"]
        ].to_dict(orient="records"),
    }
    with (out_dir / "scania_closed_by_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("Wrote scania_riskset_closed_patterns.parquet + scania_closed_by_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
