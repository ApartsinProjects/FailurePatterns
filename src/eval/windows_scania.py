"""SCANIA per-vehicle window sampler.

Mirrors windows_alibaba: failure windows are anchored on the vehicle's
terminal_repair; control windows come from non-failure vehicles at their
last surprise event + one time step.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_HORIZONS = {"last5": 5, "last10": 10, "last20": 20}
CONTROLS_PER_FAILURE = 3
MIN_VEHICLE_LEN_FOR_CTRL = 5
RNG_SEED = 20260828
FAILURE_EVENT_TYPE = "terminal_repair"


@dataclass
class WindowStats:
    n_failure_windows: dict[str, int] = field(default_factory=dict)
    n_control_windows: dict[str, int] = field(default_factory=dict)
    n_failure_vehicles_eligible: int = 0
    n_control_vehicles_pool: int = 0
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _row(entity_id, horizon, anchor, ws, we, last_ts, slc, is_failure, target):
    slc = slc.sort_values("timestamp")
    ts = slc["event_type"].tolist()
    st = slc["event_subtype"].tolist()
    return {
        "entity_id": entity_id, "horizon": horizon, "anchor": anchor,
        "window_start": ws, "window_end": we,
        "last_event_ts": last_ts if last_ts is not None else pd.NaT,
        "is_failure": is_failure, "target_failure_type": target,
        "n_events": len(slc),
        "event_type_seq": ts, "event_subtype_seq": st,
        "event_type_set": sorted(set(ts)), "event_subtype_set": sorted(set(st)),
    }


def build_windows(events: pd.DataFrame) -> tuple[pd.DataFrame, WindowStats]:
    rng = np.random.default_rng(RNG_SEED)
    stats = WindowStats()

    repairs = events[events["event_type"] == FAILURE_EVENT_TYPE]
    first_repair = repairs.sort_values("timestamp").drop_duplicates(
        "entity_id", keep="first").set_index("entity_id")
    failure_vehicles = set(first_repair.index)

    per_vehicle_len = events.groupby("entity_id", sort=False).size()

    # ------ failure windows ---------------------------------------------
    fail_events = events[events["entity_id"].isin(failure_vehicles)
                         & (events["event_type"] != FAILURE_EVENT_TYPE)]
    fail_events = fail_events.merge(
        first_repair["timestamp"].rename("_anchor_ts"),
        left_on="entity_id", right_index=True,
    )
    before_fail = fail_events[fail_events["timestamp"] < fail_events["_anchor_ts"]]
    before_fail = before_fail.sort_values(["entity_id", "timestamp"])

    rows: list[dict] = []
    eligible = 0
    for vid, ev in before_fail.groupby("entity_id", sort=False):
        anchor = first_repair.loc[vid, "timestamp"]
        eligible += 1
        for hname, k in COUNT_HORIZONS.items():
            slc = ev.tail(k)
            ws = slc["timestamp"].min()
            last_ts = slc["timestamp"].max()
            rows.append(_row(vid, hname, anchor, ws, anchor, last_ts,
                             slc, True, "repair"))
    stats.n_failure_vehicles_eligible = eligible

    # ------ control windows ---------------------------------------------
    n_wanted = eligible * CONTROLS_PER_FAILURE
    ctrl_candidates = per_vehicle_len[
        ~per_vehicle_len.index.isin(failure_vehicles)
        & (per_vehicle_len >= MIN_VEHICLE_LEN_FOR_CTRL)
    ].index.to_numpy()
    stats.n_control_vehicles_pool = int(len(ctrl_candidates))

    if n_wanted < len(ctrl_candidates):
        idx = rng.choice(len(ctrl_candidates), size=n_wanted, replace=False)
        chosen = set(ctrl_candidates[idx])
    else:
        chosen = set(ctrl_candidates)

    ctrl_events = events[events["entity_id"].isin(chosen)
                         & (events["event_type"] != FAILURE_EVENT_TYPE)]
    ctrl_events = ctrl_events.sort_values(["entity_id", "timestamp"])
    for vid, ev in ctrl_events.groupby("entity_id", sort=False):
        anchor = ev["timestamp"].max() + pd.Timedelta(hours=1)
        for hname, k in COUNT_HORIZONS.items():
            slc = ev.tail(k)
            ws = slc["timestamp"].min()
            last_ts = slc["timestamp"].max()
            rows.append(_row(vid, hname, anchor, ws, anchor, last_ts,
                             slc, False, None))

    windows = pd.DataFrame(rows)

    for hname in COUNT_HORIZONS:
        m = windows["horizon"] == hname
        stats.n_failure_windows[hname] = int((m & windows["is_failure"]).sum())
        stats.n_control_windows[hname] = int((m & ~windows["is_failure"]).sum())

    stats.invariants = {
        "at_least_one_failure_per_horizon": all(v > 0 for v in stats.n_failure_windows.values()),
        "at_least_one_control_per_horizon": all(v > 0 for v in stats.n_control_windows.values()),
        "no_terminal_repair_in_windows": all(
            FAILURE_EVENT_TYPE not in r["event_type_seq"]
            for _, r in windows.iterrows()
        ),
    }
    return windows, stats


def run(events_parquet: Path, out_dir: Path) -> WindowStats:
    events = pd.read_parquet(events_parquet)
    windows, stats = build_windows(events)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "scania_windows.parquet", index=False)
    with (out_dir / "scania_windows_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
