"""(2) Apply BH and BY FDR correction to SCANIA matched-hazard p-values.

Reads results/patterns/scania_matched_hazard.parquet, computes BH and BY
q-values on the p_value column (excluding NaN rows), and reports how many
patterns survive at q<0.05 (paper-facing) and q<0.01.

BY is the arbitrary-dependence FDR; since patterns share windows and thus
are strongly dependent, BY is the correct correction to report alongside BH.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.discovery_inference_split import bh_qvalues, by_qvalues


def main() -> int:
    path = ROOT / "results/patterns/scania_matched_hazard.parquet"
    df = pd.read_parquet(path)
    keep = df["p_value"].notna()
    pv = df.loc[keep, "p_value"].to_numpy(dtype=float)

    q_bh = bh_qvalues(pv)
    q_by = by_qvalues(pv)

    df["q_bh"] = np.nan
    df["q_by"] = np.nan
    df.loc[keep, "q_bh"] = q_bh
    df.loc[keep, "q_by"] = q_by

    df.to_parquet(path, index=False)

    n = len(df)
    n_p = int(keep.sum())
    n_hr_gt1 = int(((df["hazard_ratio"] > 1) & keep).sum())
    n_bh_005 = int((df["q_bh"] < 0.05).sum())
    n_bh_001 = int((df["q_bh"] < 0.01).sum())
    n_by_005 = int((df["q_by"] < 0.05).sum())
    n_by_001 = int((df["q_by"] < 0.01).sum())
    n_bh_005_hr = int(((df["q_bh"] < 0.05) & (df["hazard_ratio"] > 1) &
                       (df["hr_ci_low"] > 1.0)).sum())
    n_by_005_hr = int(((df["q_by"] < 0.05) & (df["hazard_ratio"] > 1) &
                       (df["hr_ci_low"] > 1.0)).sum())

    top = df.dropna(subset=["p_value"]).sort_values("p_value").head(10)
    top_records = []
    for _, r in top.iterrows():
        top_records.append({
            "itemset": list(r["itemset"])[:4],
            "hr": round(float(r["hazard_ratio"]), 3),
            "ci": [round(float(r["hr_ci_low"]), 3), round(float(r["hr_ci_high"]), 3)],
            "p": float(r["p_value"]),
            "q_bh": float(r["q_bh"]),
            "q_by": float(r["q_by"]),
        })

    summary = {
        "n_patterns_total": n,
        "n_with_pvalue": n_p,
        "n_hr_gt1_with_p": n_hr_gt1,
        "bh_significant_005": n_bh_005,
        "bh_significant_001": n_bh_001,
        "by_significant_005": n_by_005,
        "by_significant_001": n_by_001,
        "bh_005_and_hr_gt1_ci_excl_1": n_bh_005_hr,
        "by_005_and_hr_gt1_ci_excl_1": n_by_005_hr,
        "top10_by_pvalue": top_records,
    }
    (ROOT / "results/patterns/scania_matched_hazard_fdr.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
