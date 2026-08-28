"""BGL per-rack window sampler.

BGL alerts cluster into dense cascades (kernel FATAL storms, APP timeouts,
etc.). Anchoring on every one of the 348k alerts would produce 8 M
windows with massive overlap and destroy any matched-control property.
We instead group alerts on a rack into EPISODES separated by an
``EPISODE_GAP`` inter-arrival threshold, and anchor on the first alert
of each episode. This gives ~10-30 k anchors across 64 racks with clean
inter-episode intervals from which controls can be sampled.

Output schema mirrors ``src.eval.windows`` and
``src.eval.windows_alibaba`` so downstream mining and prediction code
runs unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_HORIZONS = {"last5": 5, "last10": 10, "last20": 20}
EPISODE_GAP = pd.Timedelta(hours=1)
CONTROL_CLEAN_MARGIN = pd.Timedelta(hours=1)
CONTROLS_PER_EPISODE = 3
CTRL_ATTEMPTS = 40
RNG_SEED = 20260828
FAILURE_EVENT_TYPE = "terminal_alert"


@dataclass
class WindowStats:
    n_failure_windows: dict[str, int] = field(default_factory=dict)
    n_control_windows: dict[str, int] = field(default_factory=dict)
    n_episodes: int = 0
    n_alerts_total: int = 0
    episode_gap_seconds: int = int(EPISODE_GAP.total_seconds())
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _row(entity_id, horizon, anchor, ws, we, last_ts, slc, is_failure, target):
    slc = slc.sort_values("timestamp")
    ts = slc["event_type"].tolist()
    st = slc["event_subtype"].tolist()
    return {
        "entity_id": entity_id,
        "horizon": horizon,
        "anchor": anchor,
        "window_start": ws,
        "window_end": we,
        "last_event_ts": last_ts if last_ts is not None else pd.NaT,
        "is_failure": is_failure,
        "target_failure_type": target,
        "n_events": len(slc),
        "event_type_seq": ts,
        "event_subtype_seq": st,
        "event_type_set": sorted(set(ts)),
        "event_subtype_set": sorted(set(st)),
    }


def _episodes(alert_times: pd.Series) -> pd.DataFrame:
    """Group consecutive alerts into episodes separated by EPISODE_GAP.
    Return a frame with one row per episode: first_ts (anchor) + first alert's subtype."""
    if alert_times.empty:
        return pd.DataFrame(columns=["anchor", "first_subtype"])
    alert_times = alert_times.sort_values("timestamp").reset_index(drop=True)
    gap = alert_times["timestamp"].diff()
    episode_id = (gap > EPISODE_GAP).cumsum()
    grouped = alert_times.assign(_eid=episode_id).groupby("_eid", sort=False)
    firsts = grouped.first()
    return firsts.rename(columns={"timestamp": "anchor",
                                  "event_subtype": "first_subtype"})[["anchor", "first_subtype"]]


def build_windows(events: pd.DataFrame) -> tuple[pd.DataFrame, WindowStats]:
    rng = np.random.default_rng(RNG_SEED)
    stats = WindowStats()

    alerts = events[events["event_type"] == FAILURE_EVENT_TYPE][
        ["entity_id", "timestamp", "event_subtype"]
    ]
    stats.n_alerts_total = int(len(alerts))

    # Filter alerts out of the STREAM used to build windows. Alerts are
    # the target; including them as "pre-alert events" reduces the
    # mining question to "alerts follow alerts", which is trivial.
    # The mining question is: which non-alert signals (system_error,
    # system_warning) precede an alert episode?
    non_alert_events = events[events["event_type"] != FAILURE_EVENT_TYPE]

    rows: list[dict] = []
    total_episodes = 0

    for rack, ent in non_alert_events.groupby("entity_id", sort=False):
        ent = ent.sort_values("timestamp").reset_index(drop=True)
        rack_alerts = alerts[alerts["entity_id"] == rack][
            ["timestamp", "event_subtype"]
        ]
        eps = _episodes(rack_alerts)
        total_episodes += len(eps)

        # -- failure windows: one per episode per horizon --
        for _, ep in eps.iterrows():
            anchor = ep["anchor"]
            target = ep["first_subtype"]
            before = ent[ent["timestamp"] < anchor]
            if before.empty:
                continue
            for hname, k in COUNT_HORIZONS.items():
                slc = before.tail(k)
                ws = slc["timestamp"].min()
                last_ts = slc["timestamp"].max()
                rows.append(_row(rack, hname, anchor, ws, anchor, last_ts,
                                 slc, True, target))

        # -- control windows: sample clean anchors on the same rack --
        n_wanted = len(eps) * CONTROLS_PER_EPISODE
        if n_wanted == 0 or ent.empty:
            continue
        alert_times = rack_alerts["timestamp"].values.astype("datetime64[ns]")
        margin_ns = np.timedelta64(int(CONTROL_CLEAN_MARGIN.total_seconds() * 1e9), "ns")
        t_min = ent["timestamp"].min() + CONTROL_CLEAN_MARGIN
        t_max = ent["timestamp"].max() - CONTROL_CLEAN_MARGIN
        span_sec = (t_max - t_min).total_seconds()
        if span_sec <= 0:
            continue
        got = 0
        tries = 0
        while got < n_wanted and tries < n_wanted * CTRL_ATTEMPTS:
            tries += 1
            cand = t_min + pd.Timedelta(seconds=float(rng.uniform(0, span_sec)))
            cand64 = np.datetime64(cand.value, "ns")
            if len(alert_times) and np.min(np.abs(alert_times - cand64)) <= margin_ns:
                continue
            before = ent[ent["timestamp"] < cand]
            if before.empty:
                continue
            for hname, k in COUNT_HORIZONS.items():
                slc = before.tail(k)
                ws = slc["timestamp"].min()
                last_ts = slc["timestamp"].max()
                rows.append(_row(rack, hname, cand, ws, cand, last_ts,
                                 slc, False, None))
            got += 1

    stats.n_episodes = total_episodes
    windows = pd.DataFrame(rows)

    for hname in COUNT_HORIZONS:
        m = windows["horizon"] == hname
        stats.n_failure_windows[hname] = int((m & windows["is_failure"]).sum())
        stats.n_control_windows[hname] = int((m & ~windows["is_failure"]).sum())

    stats.invariants = {
        "at_least_one_failure_per_horizon": all(v > 0 for v in stats.n_failure_windows.values()),
        "at_least_one_control_per_horizon": all(v > 0 for v in stats.n_control_windows.values()),
        "class_ratio_at_least_2_to_1": all(
            (stats.n_control_windows[h] / max(1, stats.n_failure_windows[h])) >= 2.0
            for h in COUNT_HORIZONS
        ),
        "episodes_much_fewer_than_alerts": stats.n_episodes < stats.n_alerts_total // 3,
    }
    return windows, stats


def run(events_parquet: Path, out_dir: Path) -> WindowStats:
    events = pd.read_parquet(events_parquet)
    windows, stats = build_windows(events)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "bgl_windows.parquet", index=False)
    with (out_dir / "bgl_windows_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
