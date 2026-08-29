"""Kelmarsh wind-farm status-event loader (physical cascade dataset).

Plumley et al. 2022 Zenodo release. Six Senvion MM92 turbines, four+
years of alarm/event logs. Each event carries:
- Timestamp start / end
- Status: Stop | Warning | Informational | Communication
- Code (numeric fault ID)
- Message (human-readable)
- IEC category: Forced outage | Full Performance | Out of Env Spec | ...

We normalise to shared vocabulary:
- event_type = one of {system_warning, system_stop, system_info, system_comm}
- event_subtype = str(Code)  (2550, 5720, etc.)
- Forced-outage events additionally emit a `terminal_failure` marker
  at the same timestamp for anchor purposes.

Entity is the turbine id (1-6). Failure = terminal_failure.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STATUS_MAP = {
    "Warning":       "system_warning",
    "Stop":          "system_stop",
    "Informational": "system_info",
    "Communication": "system_comm",
}
SHARED_VOCAB = set(STATUS_MAP.values()) | {"terminal_failure"}


@dataclass
class LoadStats:
    n_files: int
    n_events_raw: int
    n_events_kept: int
    n_turbines: int
    n_forced_outages: int
    time_min: str
    time_max: str
    events_by_type: dict[str, int]
    events_by_subtype_top10: dict[str, int]
    invariants: dict[str, bool]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _turbine_from_name(path: str) -> int:
    m = re.search(r"Status_Kelmarsh_(\d+)_", Path(path).name)
    return int(m.group(1)) if m else -1


def load_and_normalize(status_dir: Path) -> tuple[pd.DataFrame, LoadStats]:
    files = sorted(glob.glob(str(status_dir / "Status_Kelmarsh_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(
            f, skiprows=9,
            parse_dates=["Timestamp start", "Timestamp end"],
            low_memory=False,
        )
        df["turbine"] = _turbine_from_name(f)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    n_raw = len(raw)

    # Base event rows (from all statuses we recognise)
    keep = raw[raw["Status"].isin(STATUS_MAP.keys())].copy()
    keep["event_type"] = keep["Status"].map(STATUS_MAP)
    keep["event_subtype"] = keep["Code"].astype(str)
    keep["timestamp"] = keep["Timestamp start"]
    keep["entity_id"] = keep["turbine"].astype(str).apply(lambda x: f"T{x}")

    base = keep[["entity_id", "timestamp", "event_type", "event_subtype"]].copy()

    # Add terminal_failure markers at each Forced outage timestamp
    fo = raw[raw["IEC category"] == "Forced outage"]
    fo_rows = pd.DataFrame({
        "entity_id": fo["turbine"].astype(str).apply(lambda x: f"T{x}"),
        "timestamp": fo["Timestamp start"],
        "event_type": "terminal_failure",
        "event_subtype": fo["Code"].astype(str),
    })
    events = pd.concat([base, fo_rows], ignore_index=True)
    events = events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    n_fo = int(len(fo_rows))
    stats = LoadStats(
        n_files=len(files),
        n_events_raw=n_raw,
        n_events_kept=int(len(events)),
        n_turbines=int(events["entity_id"].nunique()),
        n_forced_outages=n_fo,
        time_min=str(events["timestamp"].min()),
        time_max=str(events["timestamp"].max()),
        events_by_type={
            k: int(v) for k, v in events["event_type"].value_counts().items()
        },
        events_by_subtype_top10={
            k: int(v) for k, v in events["event_subtype"].value_counts().head(10).items()
        },
        invariants={
            "vocab_subset_of_shared": set(events["event_type"].unique()).issubset(SHARED_VOCAB),
            "at_least_20_forced_outages": n_fo >= 20,
            "at_least_2_turbines": int(events["entity_id"].nunique()) >= 2,
        },
    )
    return events, stats


def run(status_dir: Path, out_dir: Path) -> LoadStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, stats = load_and_normalize(status_dir)
    events.to_parquet(out_dir / "kelmarsh_events.parquet", index=False)
    with (out_dir / "kelmarsh_load_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
