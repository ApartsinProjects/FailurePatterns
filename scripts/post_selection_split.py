"""Rerun significance testing on all four traces with entity-disjoint
discovery / inference split (W1 from DAMI review).

For each (trace, horizon), split the windows entity-disjoint into
50%/50% discovery/inference. Mine FP-Growth on discovery. Score every
mined pattern's hypergeometric p on the inference half. Apply BH and
BY correction on the resulting family. Report the number of patterns
significant under each correction.

Emits results/patterns/post_selection_significance.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.discovery_inference_split import (
    entity_disjoint_split, score_on_inference_half, bh_qvalues, by_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets

CONFIGS = [
    ("Azure",   ROOT / "data/processed/azure_windows.parquet",   ["24h", "last5", "last10"]),
    ("Alibaba", ROOT / "data/processed/alibaba_windows.parquet", ["last3", "last5", "last10"]),
    ("BGL",     ROOT / "data/processed/bgl_windows.parquet",     ["last5", "last10", "last20"]),
    ("SCANIA",  ROOT / "data/processed/scania_windows.parquet",  ["last5", "last10", "last20"]),
]


def main() -> int:
    out = []
    for name, path, horizons in CONFIGS:
        wins = pd.read_parquet(path)
        for h in horizons:
            sub = wins[wins["horizon"] == h]
            if sub.empty or int(sub["is_failure"].sum()) < 5:
                continue
            disc, inf = entity_disjoint_split(sub, discovery_frac=0.5)
            # Mine on discovery only
            patterns, _ = mine_itemsets(disc, [h], min_support=0.05)
            if patterns.empty:
                out.append({"trace": name, "horizon": h,
                            "n_disc_entities": int(disc["entity_id"].nunique()),
                            "n_inf_entities": int(inf["entity_id"].nunique()),
                            "n_disc_windows": int(len(disc)),
                            "n_inf_windows": int(len(inf)),
                            "n_patterns_mined_on_discovery": 0,
                            "n_significant_bh_005": 0,
                            "n_significant_by_005": 0})
                continue
            scored = score_on_inference_half(inf, patterns[["itemset"]])
            p = scored["inf_p_value"].to_numpy(dtype=float)
            q_bh = bh_qvalues(p)
            q_by = by_qvalues(p)
            n_bh = int((q_bh < 0.05).sum())
            n_by = int((q_by < 0.05).sum())
            out.append({
                "trace": name, "horizon": h,
                "n_disc_entities": int(disc["entity_id"].nunique()),
                "n_inf_entities": int(inf["entity_id"].nunique()),
                "n_disc_windows": int(len(disc)),
                "n_inf_windows": int(len(inf)),
                "n_patterns_mined_on_discovery": int(len(patterns)),
                "n_significant_bh_005": n_bh,
                "n_significant_by_005": n_by,
                "fraction_bh": round(n_bh / max(1, len(patterns)), 3),
                "fraction_by": round(n_by / max(1, len(patterns)), 3),
            })
            print(f"[{name} {h}] disc={len(disc)}w inf={len(inf)}w mined={len(patterns)} "
                  f"sig_BH={n_bh} sig_BY={n_by}", flush=True)

    (ROOT / "results/patterns/post_selection_significance.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nWrote results/patterns/post_selection_significance.json ({len(out)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
