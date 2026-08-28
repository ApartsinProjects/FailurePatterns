"""Emit a compact lead-time markdown per dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"

for ds, split_note in [
    ("azure",   "temporal split at 2015-09-01, entity = machine"),
    ("alibaba", "temporal split at 2018-01-07, entity = job"),
]:
    r = pd.read_parquet(TAB / f"{ds}_predictive.parquet")
    r = r.copy()
    r["median_lead_min"] = r["median_lead_seconds"] / 60.0
    r["median_lead_h"] = r["median_lead_seconds"] / 3600.0
    tbl = r[[
        "horizon", "feature_set", "n_tp",
        "median_lead_min", "median_lead_h",
        "p25_lead_seconds", "p75_lead_seconds",
    ]].round(2)
    lines = []
    lines.append(f"# {ds.title()} lead time on true positives\n")
    lines.append(f"{split_note}. TPs are test-set failure windows the classifier "
                 "labels positive at threshold 0.5. Lead time = "
                 "anchor - last_event_ts.\n")
    lines.append(tbl.to_markdown(index=False))
    lines.append("")
    (TAB / f"{ds}_leadtime.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {TAB / f'{ds}_leadtime.md'}")
