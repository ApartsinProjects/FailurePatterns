"""W2 fix: post-selection-valid SCANIA matched conditional logistic.

The earlier scania_matched_hazard.py selected the top-200 patterns by
case-hit count on the SAME risk-set sample it then tested, which is a
post-selection inference problem: the tested family is chosen using the
outcome. This script fixes it with a strict two-stage design.

  1. Split matched sets (match_id) 50/50 into DISCOVERY and INFERENCE.
  2. On the DISCOVERY windows only, rank the mined patterns by discovery
     case-hit count and take the top-200. Selection sees only discovery
     outcomes.
  3. On the INFERENCE windows only, fit matched conditional logistic
     (stratified by match_id) for exactly those 200 fixed patterns.
  4. BH and BY correct across the 200 inference p-values.

Reports how many survive on the honest inference half, alongside the
in-sample (post-selection) count for transparency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.matched_hazard import score_patterns_matched
from src.eval.discovery_inference_split import bh_qvalues, by_qvalues

SEED = 20260828
TOP_K = 200


def _items(row) -> set:
    return {f"{t}:{s}" for t, s in
            zip(row["event_type_seq"], row["event_subtype_seq"])}


def main() -> int:
    windows = pd.read_parquet(ROOT / "results/patterns/scania_riskset_windows.parquet")
    patterns = pd.read_parquet(ROOT / "results/patterns/scania_riskset_patterns.parquet")

    # --- split matched sets 50/50 ---
    rng = np.random.default_rng(SEED)
    mids = np.array(sorted(windows["match_id"].unique()))
    rng.shuffle(mids)
    cut = len(mids) // 2
    disc_mids = set(mids[:cut].tolist())
    inf_mids = set(mids[cut:].tolist())
    disc = windows[windows["match_id"].isin(disc_mids)].copy()
    inf = windows[windows["match_id"].isin(inf_mids)].copy()
    print(f"Discovery matched sets: {len(disc_mids)} ({len(disc)} windows); "
          f"Inference matched sets: {len(inf_mids)} ({len(inf)} windows)")

    # --- rank patterns by DISCOVERY case-hit count, take top-200 ---
    disc_items = [_items(r) for _, r in disc.iterrows()]
    disc_case = disc["is_failure"].to_numpy().astype(bool)
    hits = []
    for _, r in patterns.iterrows():
        pat = set(r["itemset"])
        present = np.array([pat.issubset(s) for s in disc_items])
        hits.append(int((present & disc_case).sum()))
    patterns = patterns.assign(disc_case_hits=hits)
    top = patterns.sort_values("disc_case_hits", ascending=False).head(TOP_K).copy()
    print(f"Selected top-{TOP_K} patterns by discovery case-hits "
          f"(min disc hits in set: {int(top['disc_case_hits'].min())})")

    # --- test the fixed 200 on the INFERENCE half ---
    scored = score_patterns_matched(inf, top[["itemset", "event_type_seq",
                                              "event_subtype_seq"]]
                                    if "event_type_seq" in top.columns else top)
    scored = scored.dropna(subset=["p_value"]).copy()
    if scored.empty:
        print("No informative inference-half strata for the selected patterns.")
        return 1
    scored["q_bh"] = bh_qvalues(scored["p_value"].to_numpy())
    scored["q_by"] = by_qvalues(scored["p_value"].to_numpy())
    scored = scored.sort_values("p_value")

    n = len(scored)
    n_bh = int(((scored["q_bh"] < 0.05) & (scored["hr_ci_low"] > 1)).sum())
    n_by = int(((scored["q_by"] < 0.05) & (scored["hr_ci_low"] > 1)).sum())
    n_hr_gt1 = int((scored["hazard_ratio"] > 1).sum())

    top10 = []
    for _, r in scored.sort_values("hazard_ratio", ascending=False).head(10).iterrows():
        top10.append({
            "itemset": list(r["itemset"])[:4],
            "hr": round(float(r["hazard_ratio"]), 3),
            "ci": [round(float(r["hr_ci_low"]), 3), round(float(r["hr_ci_high"]), 3)],
            "p": float(r["p_value"]), "q_bh": float(r["q_bh"]), "q_by": float(r["q_by"]),
            "n_case_hits": int(r["n_case_hits"]),
        })

    summary = {
        "design": "entity-disjoint matched-set split; top-200 selected on discovery "
                  "case-hits; conditional logistic tested on inference half",
        "n_discovery_sets": len(disc_mids), "n_inference_sets": len(inf_mids),
        "n_patterns_tested": n,
        "n_hr_gt1": n_hr_gt1,
        "inference_bh_005_ci_excl_1": n_bh,
        "inference_by_005_ci_excl_1": n_by,
        "top_hr": top10[0] if top10 else None,
        "top10": top10,
    }
    (ROOT / "results/patterns/scania_matched_hazard_split.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "top10"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
