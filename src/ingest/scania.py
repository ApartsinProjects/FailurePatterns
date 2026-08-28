"""SCANIA Component X loader with defensible binning.

Turns per-vehicle numeric readouts (counters, histograms) into a
discrete event stream compatible with the mining pipeline.

Binning design (declared up front, defended in the paper):

1. Sort readouts within each vehicle by ``time_step``.
2. For each of the 105 numeric feature columns, compute the
   inter-readout DELTA (absolute value). A large positive delta on a
   counter feature is a "surprise" that could mark a fault event.
3. For each (vehicle, feature) pair, compute the PER-VEHICLE
   ``QUANTILE`` threshold of its absolute deltas. Per-vehicle
   thresholds normalise for baseline usage patterns (a heavily-utilised
   truck accumulates counters faster than a light-duty one).
4. At each readout, emit a token ``event_type = counter_surprise``,
   ``event_subtype = <feature_name>`` for every feature whose
   |delta| exceeds ITS OWN vehicle-specific threshold.
5. The first readout per vehicle has no predecessor, so it emits no
   surprise events (skipped).

Entity = ``vehicle_id``. Timestamp = a pd.Timestamp built from
``time_step`` at a fixed synthetic epoch (2019-01-01 + time_step days;
time_step is a normalised study clock, not calendar time, so absolute
epoch is arbitrary but stable across runs).

Failure label = ``train_tte.in_study_repair``. The target failure event
is a synthetic ``terminal_repair`` event placed at each failure
vehicle's LAST readout timestamp.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SHARED_VOCAB = {"counter_surprise", "terminal_repair"}
QUANTILE = 0.90
EPOCH = pd.Timestamp("2019-01-01 00:00:00")


@dataclass
class LoadStats:
    n_vehicles: int
    n_failure_vehicles: int
    n_readouts: int
    n_features: int
    quantile: float
    n_surprise_events: int
    n_terminal_repair_events: int
    time_min: str
    time_max: str
    invariants: dict[str, bool]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_and_normalize(
    readouts_csv: Path,
    tte_csv: Path,
) -> tuple[pd.DataFrame, LoadStats]:
    tte = pd.read_csv(tte_csv)
    readouts = pd.read_csv(readouts_csv)
    readouts = readouts.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

    feature_cols = [c for c in readouts.columns if c not in {"vehicle_id", "time_step"}]

    # per-vehicle delta per feature
    grouped = readouts.groupby("vehicle_id", sort=False)
    deltas = grouped[feature_cols].diff().abs()

    # PER-VEHICLE quantile threshold per feature. Each vehicle's own
    # 90th-percentile of a feature's delta is that vehicle's "surprise"
    # bar. Broadcast the resulting (n_vehicles x n_features) frame back
    # to the row-level so we can gt-compare cell-wise.
    thresh_by_veh = (
        deltas.assign(vehicle_id=readouts["vehicle_id"].values)
        .groupby("vehicle_id", sort=False)[feature_cols]
        .quantile(QUANTILE)
    )
    per_row_threshold = thresh_by_veh.loc[readouts["vehicle_id"].values].reset_index(drop=True)

    mask = deltas.gt(per_row_threshold)

    # convert to long format: (row_index, feature) pairs where mask is True
    stacked = mask.stack()
    stacked = stacked[stacked].reset_index()
    # stacked now has 3 cols: original index name, column name (feature), the True value
    stacked.columns = ["row_idx", "feature", "_val"]
    where = stacked[["row_idx", "feature"]]

    long_df = where.assign(
        vehicle_id=readouts.loc[where["row_idx"], "vehicle_id"].values,
        time_step=readouts.loc[where["row_idx"], "time_step"].values,
    )
    long_df["event_type"] = "counter_surprise"
    long_df["event_subtype"] = long_df["feature"].astype(str)

    events = long_df[["vehicle_id", "time_step", "event_type", "event_subtype"]].copy()
    events = events.rename(columns={"vehicle_id": "entity_id"})

    # Add terminal_repair events at each failure vehicle's last readout
    failure_vehicles = set(tte.loc[tte["in_study_repair"] == 1, "vehicle_id"])
    last_readout = readouts.groupby("vehicle_id")["time_step"].max().reset_index()
    last_readout = last_readout[last_readout["vehicle_id"].isin(failure_vehicles)]
    repair_rows = pd.DataFrame({
        "entity_id": last_readout["vehicle_id"].values,
        "time_step": last_readout["time_step"].values,
        "event_type": "terminal_repair",
        "event_subtype": "repair",
    })
    events = pd.concat([events, repair_rows], ignore_index=True)

    events["timestamp"] = EPOCH + pd.to_timedelta(events["time_step"], unit="D")
    events = events[["entity_id", "timestamp", "event_type", "event_subtype"]]
    events = events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    stats = LoadStats(
        n_vehicles=int(readouts["vehicle_id"].nunique()),
        n_failure_vehicles=len(failure_vehicles),
        n_readouts=int(len(readouts)),
        n_features=len(feature_cols),
        quantile=QUANTILE,
        n_surprise_events=int((events["event_type"] == "counter_surprise").sum()),
        n_terminal_repair_events=int((events["event_type"] == "terminal_repair").sum()),
        time_min=str(events["timestamp"].min()),
        time_max=str(events["timestamp"].max()),
        invariants={
            "vocab_subset_of_shared":  set(events["event_type"].unique()).issubset(SHARED_VOCAB),
            "at_least_1000_failure_vehicles": len(failure_vehicles) >= 1000,
            "at_least_100k_surprise_events": int((events["event_type"] == "counter_surprise").sum()) >= 100_000,
        },
    )
    return events, stats


def run(readouts_csv: Path, tte_csv: Path, out_dir: Path) -> LoadStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, stats = load_and_normalize(readouts_csv, tte_csv)
    events.to_parquet(out_dir / "scania_events.parquet", index=False)
    with (out_dir / "scania_load_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
