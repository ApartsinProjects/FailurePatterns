"""SCANIA matched conditional logistic (W2 from DAMI review).

Replaces the pooled 2x2 MH-OR with a proper matched-set estimator:
statsmodels.ConditionalLogit stratified by match_id. Recovers the
per-pattern hazard ratio under Prentice-Breslow / incidence-density
sampling.

For time budget we score the top-200 patterns by n_case_hits
(equivalently, MH-OR after ranking). Full 42k pattern space is
tractable but overkill; the top-200 by hits dominate the story.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.matched_hazard import score_patterns_matched


def main() -> int:
    windows = pd.read_parquet(ROOT / "results/patterns/scania_riskset_windows.parquet")
    patterns = pd.read_parquet(ROOT / "results/patterns/scania_riskset_patterns.parquet")

    print(f"Risk-set windows: {len(windows)}  "
          f"(cases={int(windows['is_failure'].sum())}, "
          f"controls={int((~windows['is_failure']).sum())}, "
          f"matched sets={windows['match_id'].nunique()})")
    print(f"Input patterns: {len(patterns)}")

    top = patterns.sort_values("n_failure", ascending=False).head(200).copy()
    print(f"Analyzing top-200 patterns by n_case_hits...")

    scored = score_patterns_matched(windows, top)
    scored = scored.sort_values("hazard_ratio", ascending=False)

    n_sig = int(scored["significant_005"].sum())
    print(f"Matched conditional-logistic significant (HR CI excludes 1 AND p<0.05): "
          f"{n_sig} / {len(scored)}")
    print(f"Top 10 by hazard ratio:")
    for _, r in scored.head(10).iterrows():
        marker = "**" if r["significant_005"] else "  "
        it = " + ".join(str(x) for x in list(r["itemset"])[:3])
        print(f"  {marker} HR={r['hazard_ratio']:.3f} CI=[{r['hr_ci_low']:.3f}, "
              f"{r['hr_ci_high']:.3f}] p={r['p_value']:.2e} "
              f"n_case_hits={r['n_case_hits']}: {it}")

    out_dir = ROOT / "results/patterns"
    scored.to_parquet(out_dir / "scania_matched_hazard.parquet", index=False)
    summary = {
        "n_patterns_scored": int(len(scored)),
        "n_significant_conditional_logistic_005": n_sig,
        "top10": scored.head(10)[
            ["itemset", "hazard_ratio", "hr_ci_low", "hr_ci_high",
             "p_value", "significant_005", "n_case_hits", "n_control_hits",
             "n_informative_strata"]
        ].to_dict(orient="records"),
    }
    (out_dir / "scania_matched_hazard_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nWrote {out_dir / 'scania_matched_hazard.parquet'} and summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
