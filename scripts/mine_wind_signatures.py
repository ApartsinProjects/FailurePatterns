"""Extract signature catalog for a wind-farm windows parquet
(Kelmarsh or Penmanshiel).

Discovery/inference-half split at 50/50 entity-disjoint (entity = turbine),
FP-Growth on discovery half at 5% min-support, score each pattern on the
inference half via exact hypergeometric, BH and BY corrections. Report per
horizon:
  - number of discovery / inference windows
  - patterns mined on the discovery half
  - patterns significant at q_bh<0.05 and q_by<0.05
  - top-10 patterns by inference-half lift
Writes results/patterns/<stem>_signatures.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.discovery_inference_split import (
    entity_disjoint_split, score_on_inference_half,
    bh_qvalues, by_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets


def run_for(wins_path: Path, out_path: Path, horizons: list[str],
            min_support: float = 0.05, top_k: int = 10) -> dict:
    wins = pd.read_parquet(wins_path)
    out: dict = {}
    for h in horizons:
        sub = wins[wins["horizon"] == h]
        if sub.empty or int(sub["is_failure"].sum()) < 5:
            continue
        disc, inf = entity_disjoint_split(sub, discovery_frac=0.5)
        pats, _ = mine_itemsets(disc, [h], min_support=min_support)
        if pats.empty:
            out[h] = {"n_disc_windows": int(len(disc)),
                      "n_inf_windows": int(len(inf)),
                      "n_mined": 0, "n_sig_bh": 0, "n_sig_by": 0,
                      "top_patterns": []}
            continue
        scored = score_on_inference_half(inf, pats[["itemset"]])
        pv = scored["inf_p_value"].to_numpy(dtype=float)
        scored["q_bh"] = bh_qvalues(pv)
        scored["q_by"] = by_qvalues(pv)
        top = scored.sort_values("inf_lift", ascending=False).head(top_k)
        top_records = []
        for _, r in top.iterrows():
            top_records.append({
                "itemset": list(r["itemset"]),
                "inf_hit_f": int(r["inf_hit_f"]),
                "inf_hit_c": int(r["inf_hit_c"]),
                "inf_n_f": int(r["inf_n_f"]),
                "inf_n_c": int(r["inf_n_c"]),
                "inf_supp_f": float(r["inf_supp_f"]),
                "inf_supp_c": float(r["inf_supp_c"]),
                "inf_lift": float(r["inf_lift"]),
                "inf_p_value": float(r["inf_p_value"]),
                "q_bh": float(r["q_bh"]),
                "q_by": float(r["q_by"]),
            })
        out[h] = {
            "n_disc_windows": int(len(disc)),
            "n_inf_windows": int(len(inf)),
            "n_mined": int(len(pats)),
            "n_sig_bh": int((scored["q_bh"] < 0.05).sum()),
            "n_sig_by": int((scored["q_by"] < 0.05).sum()),
            "top_patterns": top_records,
        }
        print(f"[{wins_path.stem} {h}] mined={len(pats)} "
              f"sig_bh={out[h]['n_sig_bh']} sig_by={out[h]['n_sig_by']}",
              flush=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", nargs="+",
                    default=["1h", "6h", "24h", "last5", "last10"])
    a = ap.parse_args()
    run_for(Path(a.windows), Path(a.out), list(a.horizons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
