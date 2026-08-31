"""Highest-value next experiment: cross-farm signature transfer.

The wind-farm result is a cascade replicated across two Senvion farms,
but each farm's rule is learned and tested within-site. This script
learns the precursor rule on ALL turbines of one farm and replays it,
unchanged, on every turbine of the OTHER farm (both directions), using
the shared Senvion code vocabulary. It reports the same four numbers as
the within-farm replay, plus which learned precursor codes are shared
between the two farms.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prospective_alarm_replay import (  # reuse the exact machinery
    learn_precursors, replay_turbine, H,
)

FARMS = {
    "Kelmarsh": ROOT / "data/processed/kelmarsh_events.parquet",
    "Penmanshiel": ROOT / "data/processed/penmanshiel_events.parquet",
}


def transfer(src_name, src_path, dst_name, dst_path) -> dict:
    src = pd.read_parquet(src_path)
    dst = pd.read_parquet(dst_path)
    precursors = learn_precursors(src)               # learn on ALL of source farm
    agg = {"n_alarms": 0, "tp": 0, "fp": 0, "n_outages": 0, "detected": 0,
           "leads": [], "turbine_months": 0.0}
    for t in sorted(dst["entity_id"].unique()):       # replay per destination turbine
        r = replay_turbine(dst[dst["entity_id"] == t], precursors)
        for k in ("n_alarms", "tp", "fp", "n_outages", "detected"):
            agg[k] += r[k]
        agg["leads"].extend(r["leads"])
        agg["turbine_months"] += r["span_days"] / 30.0
    leads = np.array(agg["leads"], dtype=float)
    return {
        "direction": f"{src_name}->{dst_name}",
        "n_learned_precursor_codes": len(precursors),
        "precision": round(agg["tp"] / agg["n_alarms"], 3) if agg["n_alarms"] else None,
        "recall": round(agg["detected"] / agg["n_outages"], 3) if agg["n_outages"] else None,
        "n_outages": agg["n_outages"], "outages_detected": agg["detected"],
        "false_alarms_per_turbine_month": round(agg["fp"] / max(agg["turbine_months"], 1e-9), 3),
        "lead_median_min": round(float(np.median(leads) / 60), 1) if len(leads) else None,
        "frac_lead_ge_1h": round(float((leads >= 3600).mean()), 3) if len(leads) else None,
        "frac_lead_ge_6h": round(float((leads >= 6 * 3600).mean()), 3) if len(leads) else None,
        "_precursors": sorted(precursors),
    }


def main() -> int:
    ke = learn_precursors(pd.read_parquet(FARMS["Kelmarsh"]))
    pe = learn_precursors(pd.read_parquet(FARMS["Penmanshiel"]))
    shared = sorted(ke & pe)
    rows = [
        transfer("Kelmarsh", FARMS["Kelmarsh"], "Penmanshiel", FARMS["Penmanshiel"]),
        transfer("Penmanshiel", FARMS["Penmanshiel"], "Kelmarsh", FARMS["Kelmarsh"]),
    ]
    out = {
        "kelmarsh_precursor_codes": sorted(ke),
        "penmanshiel_precursor_codes": sorted(pe),
        "shared_precursor_codes": shared,
        "transfer": rows,
    }
    (ROOT / "results/patterns/cross_farm_transfer.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        print(r["direction"], "recall", r["recall"], "prec", r["precision"],
              "lead_med_min", r["lead_median_min"], "FA/tm",
              r["false_alarms_per_turbine_month"], flush=True)
    print("shared precursor codes:", shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
