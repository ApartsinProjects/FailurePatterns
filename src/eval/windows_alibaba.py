"""Alibaba per-job window sampler.

Alibaba jobs are short (minutes-hours), so unlike Azure PdM the natural
window is count-based only, and control windows come from DIFFERENT
non-failure jobs rather than from clean regions of the same entity.

Failure windows: for each job that contains at least one ``task_failure``,
take the last K events strictly before the first ``task_failure`` in that
job. Anchor is the failure timestamp.

Control windows: sample non-failure jobs (matched roughly on task count),
take the last K events of each. Anchor is the job's last event timestamp
plus one second.

Output schema mirrors ``src.eval.windows`` so downstream mining and
prediction code runs unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_HORIZONS = {"last3": 3, "last5": 5, "last10": 10}
CONTROLS_PER_FAILURE = 3
RNG_SEED = 20260828
MIN_JOB_LEN_FOR_CTRL = 3   # skip trivially short control jobs
FAILURE_EVENT_TYPE = "task_failure"


@dataclass
class WindowStats:
    n_failure_windows: dict[str, int] = field(default_factory=dict)
    n_control_windows: dict[str, int] = field(default_factory=dict)
    n_failure_jobs_eligible: int = 0
    n_control_jobs_pool: int = 0
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _row(entity_id, horizon, anchor, ws, we, slc, is_failure, target):
    slc = slc.sort_values("timestamp")
    ts = slc["event_type"].tolist()
    st = slc["event_subtype"].tolist()
    last_event_ts = slc["timestamp"].max() if not slc.empty else pd.NaT
    return {
        "entity_id": entity_id,
        "horizon": horizon,
        "anchor": anchor,
        "window_start": ws,
        "window_end": we,
        "last_event_ts": last_event_ts,
        "is_failure": is_failure,
        "target_failure_type": target,
        "n_events": len(slc),
        "event_type_seq": ts,
        "event_subtype_seq": st,
        "event_type_set": sorted(set(ts)),
        "event_subtype_set": sorted(set(st)),
    }


def build_windows(events: pd.DataFrame) -> tuple[pd.DataFrame, WindowStats]:
    rng = np.random.default_rng(RNG_SEED)
    stats = WindowStats()

    # First failure timestamp per job.
    fail_events = events[events["event_type"] == FAILURE_EVENT_TYPE]
    first_fail = fail_events.sort_values("timestamp").drop_duplicates(
        "entity_id", keep="first"
    ).set_index("entity_id")

    failure_jobs = set(first_fail.index)

    # Non-failure job pool with tasks-per-job count (cheap; one groupby.size()
    # on a categorical, no per-group DataFrame materialization).
    per_job_len = events.groupby("entity_id", sort=False).size()
    n_ctrl_wanted = None  # set after we know eligible_failure

    # ------- failure windows: filter events to failure jobs only ------
    fail_events_only = events[events["entity_id"].isin(failure_jobs)]
    # Attach anchor per row so we can filter events strictly before anchor.
    fail_anchor = first_fail["timestamp"]
    fail_events_only = fail_events_only.merge(
        fail_anchor.rename("_anchor_ts"), left_on="entity_id", right_index=True,
    )
    before_fail = fail_events_only[
        fail_events_only["timestamp"] < fail_events_only["_anchor_ts"]
    ]
    # For each failure job with at least one pre-anchor event, take last K.
    before_fail = before_fail.sort_values(["entity_id", "timestamp"])

    rows: list[dict] = []
    eligible_failure = 0
    for job, ev in before_fail.groupby("entity_id", sort=False):
        anchor_ts = first_fail.loc[job, "timestamp"]
        target_role = first_fail.loc[job, "event_subtype"]
        eligible_failure += 1
        for hname, k in COUNT_HORIZONS.items():
            slc = ev.tail(k)
            ws = slc["timestamp"].min()
            rows.append(_row(job, hname, anchor_ts, ws, anchor_ts, slc,
                             True, target_role))
    stats.n_failure_jobs_eligible = eligible_failure

    # ------- control windows -----------------------------------------
    n_ctrl_wanted = eligible_failure * CONTROLS_PER_FAILURE
    # Eligible non-failure jobs: length >= MIN_JOB_LEN_FOR_CTRL.
    ctrl_candidates = per_job_len[
        ~per_job_len.index.isin(failure_jobs)
        & (per_job_len >= MIN_JOB_LEN_FOR_CTRL)
    ].index.to_numpy()
    stats.n_control_jobs_pool = int(len(ctrl_candidates))

    if n_ctrl_wanted < len(ctrl_candidates):
        idx = rng.choice(len(ctrl_candidates), size=n_ctrl_wanted, replace=False)
        chosen = set(ctrl_candidates[idx])
    else:
        chosen = set(ctrl_candidates)

    ctrl_events = events[events["entity_id"].isin(chosen)]
    ctrl_events = ctrl_events.sort_values(["entity_id", "timestamp"])
    for job, ev in ctrl_events.groupby("entity_id", sort=False):
        anchor_ts = ev["timestamp"].max() + pd.Timedelta(seconds=1)
        for hname, k in COUNT_HORIZONS.items():
            slc = ev.tail(k)
            ws = slc["timestamp"].min()
            rows.append(_row(job, hname, anchor_ts, ws, anchor_ts, slc,
                             False, None))

    windows = pd.DataFrame(rows)

    for hname in COUNT_HORIZONS:
        m = windows["horizon"] == hname
        stats.n_failure_windows[hname] = int((m & windows["is_failure"]).sum())
        stats.n_control_windows[hname] = int((m & ~windows["is_failure"]).sum())

    # Invariants
    stats.invariants = {
        "anchor_not_in_failure_window_seq": all(
            FAILURE_EVENT_TYPE not in r["event_type_seq"]
            for _, r in windows[windows["is_failure"]].iterrows()
        ),
        "controls_have_no_failure_events": all(
            FAILURE_EVENT_TYPE not in r["event_type_seq"]
            for _, r in windows[~windows["is_failure"]].iterrows()
        ),
        "at_least_one_failure_per_horizon": all(
            v > 0 for v in stats.n_failure_windows.values()
        ),
        "at_least_one_control_per_horizon": all(
            v > 0 for v in stats.n_control_windows.values()
        ),
        "class_ratio_approx_3_to_1": all(
            (stats.n_control_windows[h] / max(1, stats.n_failure_windows[h]))
            > 2.5
            for h in COUNT_HORIZONS
        ),
    }
    return windows, stats


def run(events_parquet: Path, out_dir: Path) -> WindowStats:
    events = pd.read_parquet(events_parquet)
    windows, stats = build_windows(events)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "alibaba_windows.parquet", index=False)
    with (out_dir / "alibaba_windows_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
