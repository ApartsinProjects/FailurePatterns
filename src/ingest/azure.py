"""Azure PdM loader and event normalizer.

Reads the five Kaggle CSVs from ``data/raw/azure/`` and emits a single
``(entity_id, timestamp, event_type, event_subtype)`` parquet at
``data/processed/azure_events.parquet``.

Normalization rules (shared vocabulary, see PLAN.md phase 1):

- ``PdM_errors.errorID``      -> ``software_error`` (Azure errors are non-fatal)
- ``PdM_maint.comp``          -> ``maintenance`` when NOT joined to a failure,
                                 ``component_replacement`` when the same
                                 (machineID, datetime, comp) appears in
                                 ``PdM_failures``.
- ``PdM_failures.failure``    -> ``terminal_failure`` (target label).

``event_subtype`` carries the source-specific code (``error1``..``error5``,
``comp1``..``comp4``) for downstream analysis; ``event_type`` is the shared
vocab.

Telemetry is NOT emitted as events (it is continuous, not discrete). It is
preserved separately in ``data/processed/azure_telemetry.parquet`` for
potential Phase 6 feature engineering.

Sanity checks live in ``scripts/ingest_azure.py`` (the CLI entry point).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

SHARED_VOCAB = {
    "software_error",        # non-fatal errors
    "maintenance",           # scheduled / preventive
    "component_replacement", # maintenance coinciding with a recorded failure
    "terminal_failure",      # PdM_failures event (target label)
}

RAW_FILES = {
    "errors":    "PdM_errors.csv",
    "failures":  "PdM_failures.csv",
    "machines":  "PdM_machines.csv",
    "maint":     "PdM_maint.csv",
    "telemetry": "PdM_telemetry.csv",
}


SEED_FAILURE_TS = pd.Timestamp("2015-01-02 03:00:00")
"""Discovered 2026-08-28: the Azure PdM synthetic generator plants a batch
of 18 ``terminal_failure`` records at exactly 2015-01-02 03:00:00 without
any matching ``PdM_maint`` row. Every failure AFTER this timestamp joins
cleanly to maint. Treated as legitimate failure events; the
``failures_all_matched_to_maint`` invariant is scoped to non-seed failures."""


@dataclasses.dataclass
class LoadStats:
    """Sanity statistics for a load pass. Serialized to JSON alongside the
    parquet output so the run is auditable against ``docs/data-inventory.md``.
    """

    n_machines: int
    time_min: str
    time_max: str
    events_by_type: dict[str, int]
    events_by_subtype: dict[str, int]
    telemetry_rows: int
    maint_before_telemetry: int  # gotcha from scout: pre-2015 maint
    failure_maint_join_rate: float  # fraction of PdM_failures matched to maint
    n_seed_failures: int  # unmatched failures at SEED_FAILURE_TS
    invariants: dict[str, bool]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the five source CSVs. Parses datetimes at read time."""
    out: dict[str, pd.DataFrame] = {}
    for key, fname in RAW_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing Azure PdM file: {path}")
        if key == "machines":
            out[key] = pd.read_csv(path)
        else:
            out[key] = pd.read_csv(path, parse_dates=["datetime"])
    return out


def build_events(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Normalize the three event tables into a single event stream.

    Distinguishes ``maintenance`` from ``component_replacement`` by joining
    (machineID, datetime, comp) between ``PdM_maint`` and ``PdM_failures``.
    """
    errors = raw["errors"].rename(columns={"errorID": "event_subtype"})
    errors["event_type"] = "software_error"

    maint = raw["maint"].rename(columns={"comp": "event_subtype"}).copy()
    failures = raw["failures"].rename(columns={"failure": "event_subtype"}).copy()

    # Build the (machine, timestamp, comp) key for the failure-join.
    fail_key = failures.assign(_is_replace=True)[
        ["datetime", "machineID", "event_subtype", "_is_replace"]
    ]
    maint = maint.merge(
        fail_key, on=["datetime", "machineID", "event_subtype"], how="left"
    )
    maint["event_type"] = maint["_is_replace"].where(
        maint["_is_replace"].isna(), other="component_replacement"
    ).fillna("maintenance")
    maint = maint.drop(columns=["_is_replace"])

    failures["event_type"] = "terminal_failure"

    events = pd.concat(
        [
            errors[["datetime", "machineID", "event_type", "event_subtype"]],
            maint[["datetime", "machineID", "event_type", "event_subtype"]],
            failures[["datetime", "machineID", "event_type", "event_subtype"]],
        ],
        ignore_index=True,
    )
    events = events.rename(
        columns={"machineID": "entity_id", "datetime": "timestamp"}
    )
    events = events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    unknown = set(events["event_type"].unique()) - SHARED_VOCAB
    if unknown:
        raise ValueError(f"Emitted event_type outside SHARED_VOCAB: {unknown}")
    return events


def compute_stats(raw: dict[str, pd.DataFrame], events: pd.DataFrame) -> LoadStats:
    telemetry_min = raw["telemetry"]["datetime"].min()
    maint_before_telemetry = int(
        (raw["maint"]["datetime"] < telemetry_min).sum()
    )

    # Failure-to-maint join rate: how many PdM_failures rows got matched.
    failures = raw["failures"]
    maint_key = raw["maint"].set_index(
        ["datetime", "machineID", "comp"]
    ).index
    matched_mask = failures.apply(
        lambda r: (r["datetime"], r["machineID"], r["failure"]) in maint_key,
        axis=1,
    )
    matched = int(matched_mask.sum())
    fmj = float(matched) / float(len(failures)) if len(failures) else 0.0

    unmatched = failures[~matched_mask]
    n_seed = int((unmatched["datetime"] == SEED_FAILURE_TS).sum())
    non_seed_unmatched = len(unmatched) - n_seed

    invariants = {
        "n_machines_is_100": int(events["entity_id"].nunique()) == 100,
        "vocab_subset_of_shared": set(events["event_type"].unique()).issubset(
            SHARED_VOCAB
        ),
        "monotonic_per_entity": bool(
            events.groupby("entity_id")["timestamp"].apply(
                lambda s: s.is_monotonic_increasing
            ).all()
        ),
        # Scope: every failure OUTSIDE the seed batch at 2015-01-02 03:00
        # must join to a maint record on the same (machine, timestamp, comp).
        "non_seed_failures_all_matched_to_maint": non_seed_unmatched == 0,
    }

    return LoadStats(
        n_machines=int(events["entity_id"].nunique()),
        time_min=str(events["timestamp"].min()),
        time_max=str(events["timestamp"].max()),
        events_by_type={
            k: int(v) for k, v in events["event_type"].value_counts().items()
        },
        events_by_subtype={
            k: int(v) for k, v in events["event_subtype"].value_counts().items()
        },
        telemetry_rows=int(len(raw["telemetry"])),
        maint_before_telemetry=maint_before_telemetry,
        failure_maint_join_rate=fmj,
        n_seed_failures=n_seed,
        invariants=invariants,
    )


def run(raw_dir: Path, out_dir: Path) -> LoadStats:
    """End-to-end load. Writes events parquet + telemetry parquet + stats JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_dir)
    events = build_events(raw)
    stats = compute_stats(raw, events)

    events.to_parquet(out_dir / "azure_events.parquet", index=False)
    raw["telemetry"].rename(
        columns={"machineID": "entity_id", "datetime": "timestamp"}
    ).to_parquet(out_dir / "azure_telemetry.parquet", index=False)

    with (out_dir / "azure_load_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)

    return stats
