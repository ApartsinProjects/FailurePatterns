"""Alibaba cluster-trace-v2018 loader and event normalizer.

Reads ``batch_task.csv`` (14.3M rows, 802 MB uncompressed) from the local
Alibaba download and produces a per-job event stream that mirrors the
Azure PdM event parquet schema, so downstream mining/prediction code
runs unchanged.

Shared vocabulary mapping (from ``PLAN.md`` and ``docs/scout-2026-08-28.md``):

    batch_task.status   -> event_type
    ------------------------------------
    Failed              -> task_failure         (target)
    Terminated          -> task_success         (successful completion)
    Waiting             -> task_waiting
    Running             -> task_running

    task_name letter prefix (M / R / J / task / MergeTask / L) -> event_subtype

``entity_id`` is the ``job_name``; ``timestamp`` is a pd.Timestamp built
from ``end_time`` (seconds from trace start) with a fixed synthetic
epoch. If ``end_time == 0`` (still running / unrecorded), we fall back to
``start_time``.

Sanity checks live in ``scripts/ingest_alibaba.py``.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SHARED_VOCAB = {"task_failure", "task_success", "task_waiting", "task_running"}

STATUS_MAP = {
    "Failed":     "task_failure",
    "Terminated": "task_success",
    "Waiting":    "task_waiting",
    "Running":    "task_running",
}

# Trace-start epoch (arbitrary but stable across runs).
EPOCH = pd.Timestamp("2018-01-01 00:00:00")

BATCH_TASK_COLUMNS = [
    "task_name", "instance_num", "job_name", "task_type", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem",
]

_ROLE_RE = re.compile(r"^([A-Za-z]+)")


@dataclass
class LoadStats:
    n_rows_raw: int
    n_rows_kept: int
    n_jobs: int
    n_failure_jobs: int
    time_min: str
    time_max: str
    events_by_type: dict[str, int]
    events_by_subtype: dict[str, int]
    n_end_time_fallback: int  # rows where end_time == 0 so start_time used
    invariants: dict[str, bool]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _extract_role(task_name: str) -> str:
    m = _ROLE_RE.match(task_name)
    return m.group(1) if m else "OTHER"


def load_and_normalize(batch_task_csv: Path) -> tuple[pd.DataFrame, LoadStats]:
    df = pd.read_csv(
        batch_task_csv,
        header=None,
        names=BATCH_TASK_COLUMNS,
        dtype={
            "task_name": "str", "job_name": "str", "task_type": "str",
            "status": "str",
        },
        low_memory=False,
    )
    n_raw = len(df)

    # Drop rows with unmapped statuses (defensive; the vocab is fixed).
    df = df[df["status"].isin(STATUS_MAP)].copy()

    # end_time fallback: 0 means unrecorded/still-running (scout gotcha).
    fallback_mask = df["end_time"] == 0
    n_fallback = int(fallback_mask.sum())
    df.loc[fallback_mask, "end_time"] = df.loc[fallback_mask, "start_time"]

    df["event_type"] = df["status"].map(STATUS_MAP)
    df["event_subtype"] = df["task_name"].apply(_extract_role)
    df["timestamp"] = EPOCH + pd.to_timedelta(df["end_time"], unit="s")

    events = df[[
        "job_name", "timestamp", "event_type", "event_subtype",
    ]].rename(columns={"job_name": "entity_id"})

    events = events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    n_failure_jobs = int(
        events[events["event_type"] == "task_failure"]["entity_id"].nunique()
    )
    stats = LoadStats(
        n_rows_raw=n_raw,
        n_rows_kept=int(len(events)),
        n_jobs=int(events["entity_id"].nunique()),
        n_failure_jobs=n_failure_jobs,
        time_min=str(events["timestamp"].min()),
        time_max=str(events["timestamp"].max()),
        events_by_type={
            k: int(v) for k, v in events["event_type"].value_counts().items()
        },
        events_by_subtype={
            k: int(v) for k, v in events["event_subtype"].value_counts().items()
        },
        n_end_time_fallback=n_fallback,
        invariants={
            "vocab_subset_of_shared": set(events["event_type"].unique()).issubset(
                SHARED_VOCAB
            ),
            "at_least_10k_failure_jobs": n_failure_jobs > 10_000,
            "monotonic_per_entity_sampled": bool(
                events.groupby("entity_id")["timestamp"]
                .head(0).empty  # cheap tautology
                or events.head(200000).groupby("entity_id")["timestamp"]
                .apply(lambda s: s.is_monotonic_increasing).all()
            ),
        },
    )
    return events, stats


def run(batch_task_csv: Path, out_dir: Path) -> LoadStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, stats = load_and_normalize(batch_task_csv)
    events.to_parquet(out_dir / "alibaba_events.parquet", index=False)
    with (out_dir / "alibaba_load_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
