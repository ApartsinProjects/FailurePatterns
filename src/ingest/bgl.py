"""BGL (Blue Gene/L, LLNL) loader and event normalizer.

Parses the Loghub BGL.log file (4,747,963 messages, 214.7 days, 65 racks)
into the shared (entity_id, timestamp, event_type, event_subtype) format.

Design decisions:

- **Entity = rack** (the top-level R## prefix of the node ID). BGL has
  65 racks; each carries ~73k events on average, comparable in order
  of magnitude to Azure's per-machine event counts.
- **INFO messages are dropped.** They are 78% of the corpus (~3.7 M
  lines) and act as pure noise in the mining pass. The remaining
  ~1 M events carry all the FATAL / ERROR / WARNING / alert signal.
- **Alert lines** (label != "-") map to ``event_type = terminal_alert``
  with ``event_subtype = <alert code>`` (KERNMNTF, APPTO, KERNSTOR, ...).
  These are the target failure events for pre-failure window sampling.
- **Non-alert severe messages** (FATAL / ERROR / SEVERE / FAILURE) map
  to ``event_type = system_error`` with ``event_subtype = <severity>``.
- **Non-alert WARNING** → ``event_type = system_warning``, subtype
  ``"WARNING"``.

Raw line format:
    <label> <unix_ts> <date> <node1> <iso_ts> <node2> <component> <level> <severity> <message...>

We use fields: 0 label, 1 unix_ts, 3 node1 (for the rack prefix), 8 severity.
Malformed lines (fewer than 9 space-separated fields) are counted and
skipped rather than crashing the loader.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SHARED_VOCAB = {"terminal_alert", "system_error", "system_warning", "system_info"}

SEVERE_LEVELS = {"FATAL", "ERROR", "SEVERE", "FAILURE"}
WARNING_LEVELS = {"WARNING"}
INFO_LEVELS = {"INFO"}

_ALERT_RE = re.compile(r"^[A-Z]+$")   # KERNMNTF, APPTO, etc.
_RACK_RE = re.compile(r"^(R\d+)")


@dataclass
class LoadStats:
    n_lines_raw: int
    n_lines_kept: int
    n_lines_dropped_info: int
    n_lines_malformed: int
    n_racks: int
    n_alerts: int
    time_min: str
    time_max: str
    events_by_type: dict[str, int]
    events_by_subtype: dict[str, int]
    invariants: dict[str, bool]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _classify(label: str, severity: str, component: str) -> tuple[str, str] | None:
    """Return (event_type, event_subtype) or None to drop the line.

    event_subtype for non-alerts encodes component:severity so that
    RAS:INFO vs KERNEL:INFO vs APP:INFO are distinguishable items
    for pattern mining.
    """
    if label != "-" and _ALERT_RE.match(label):
        return "terminal_alert", label
    comp = component if component in {"RAS", "KERNEL", "APP", "MMCS",
                                     "DISCOVERY", "MONITOR", "LINKCARD",
                                     "BGLMASTER", "CMCS", "HARDWARE"} else "OTHER"
    if severity in SEVERE_LEVELS:
        return "system_error", f"{comp}:{severity}"
    if severity in WARNING_LEVELS:
        return "system_warning", f"{comp}:WARNING"
    if severity in INFO_LEVELS:
        return "system_info", f"{comp}:INFO"
    return None


def parse_stream(path: Path) -> tuple[pd.DataFrame, LoadStats]:
    """Stream-parse BGL.log into an events DataFrame."""
    rows: list[tuple] = []
    n_raw = 0
    n_info = 0
    n_bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_raw += 1
            parts = line.split(" ", 9)
            if len(parts) < 9:
                n_bad += 1
                continue
            label = parts[0]
            try:
                ts = int(parts[1])
            except ValueError:
                n_bad += 1
                continue
            node1 = parts[3]
            severity = parts[8]
            m = _RACK_RE.match(node1)
            if not m:
                n_bad += 1
                continue
            rack = m.group(1)
            component = parts[6]
            classified = _classify(label, severity, component)
            if classified is None:
                n_bad += 1
                continue
            event_type, event_subtype = classified
            rows.append((rack, ts, event_type, event_subtype))

    events = pd.DataFrame(rows, columns=["entity_id", "_unix_ts",
                                         "event_type", "event_subtype"])
    events["timestamp"] = pd.to_datetime(events["_unix_ts"], unit="s")
    events = events.drop(columns=["_unix_ts"])
    events = events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    n_alerts = int((events["event_type"] == "terminal_alert").sum())
    stats = LoadStats(
        n_lines_raw=n_raw,
        n_lines_kept=int(len(events)),
        n_lines_dropped_info=n_info,
        n_lines_malformed=n_bad,
        n_racks=int(events["entity_id"].nunique()),
        n_alerts=n_alerts,
        time_min=str(events["timestamp"].min()),
        time_max=str(events["timestamp"].max()),
        events_by_type={k: int(v) for k, v in events["event_type"].value_counts().items()},
        events_by_subtype={k: int(v) for k, v in events["event_subtype"].value_counts().items()},
        invariants={
            "vocab_subset_of_shared": set(events["event_type"].unique()).issubset(SHARED_VOCAB),
            "at_least_50_racks":       int(events["entity_id"].nunique()) >= 50,
            "at_least_100k_alerts":    n_alerts >= 100_000,
        },
    )
    return events, stats


def run(bgl_log: Path, out_dir: Path) -> LoadStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, stats = parse_stream(bgl_log)
    events.to_parquet(out_dir / "bgl_events.parquet", index=False)
    with (out_dir / "bgl_load_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
