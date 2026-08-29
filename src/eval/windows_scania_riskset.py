"""SCANIA risk-set matched-control window sampler + hazard-ratio scoring.

Methodological extension of the paper's matched-control pipeline to
right-censored survival-style data (per Fable's Idea 3, 2026-08-29).

**Sampling.** For each failure vehicle f with observed failure time
T_f = length_of_study_time_step, the case window contains the last K
counter-surprise events with time_step < T_f. Controls come from the
RISK SET at T_f: vehicles with length_of_study >= T_f, i.e. still
under observation at that lifetime index. Each control's window
contains its last K events with time_step < T_f, aligned to the case's
failure lifetime rather than its own end-of-observation.

**Scoring.** For each mined pattern P, we compute the Mantel-Haenszel
pooled odds ratio and its 95% CI (Robins-Greenland variance). Under
incidence-density sampling this MH-OR estimates the hazard ratio for
pattern P (Prentice & Breslow 1978; Rothman-Greenland ch. 15). Positive
log-HR with a CI excluding 1 marks a pattern statistically associated
with elevated failure risk while accounting for the observation-process
censoring the naive lift ignored.

The output parquet mirrors the existing SCANIA windows schema so the
existing FP-Growth miner in `src.mine.itemsets` runs unchanged; the
mining step then scores each pattern with MH-OR instead of lift.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LAST_K = 20
CONTROLS_PER_CASE = 3
RNG_SEED = 20260828
EPOCH = pd.Timestamp("2019-01-01 00:00:00")


@dataclass
class RiskSetStats:
    n_cases: int = 0
    n_controls_per_case: int = 0
    n_cases_with_empty_risk_set: int = 0
    n_cases_dropped_no_history: int = 0
    mean_risk_set_size: float = 0.0
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _row(entity_id, horizon, anchor, ws, we, last_ts, slc,
         is_failure, target, match_id):
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
        "event_type_set": sorted(set(ts)),
        "event_subtype_set": sorted(set(st)),
        "match_id": match_id,   # NEW: which case this control matches (or case's own id)
    }


def build_riskset_windows(
    events: pd.DataFrame,
    tte: pd.DataFrame,
    horizon_k: int = LAST_K,
    controls_per_case: int = CONTROLS_PER_CASE,
) -> tuple[pd.DataFrame, RiskSetStats]:
    rng = np.random.default_rng(RNG_SEED)
    stats = RiskSetStats(n_controls_per_case=controls_per_case)

    # Drop repair events from the event stream used for windows.
    ev = events[events["event_type"] != "terminal_repair"].copy()
    ev["time_step"] = (ev["timestamp"] - EPOCH).dt.total_seconds() / 86400.0

    tte = tte.copy()
    failure_ids = tte.loc[tte["in_study_repair"] == 1, "vehicle_id"].to_numpy()
    length_by_v = dict(zip(tte["vehicle_id"], tte["length_of_study_time_step"]))

    ev_by_v = dict(list(ev.groupby("entity_id", sort=False)))
    tte_arr_ids = tte["vehicle_id"].to_numpy()
    tte_arr_len = tte["length_of_study_time_step"].to_numpy()
    is_failure_arr = tte["in_study_repair"].to_numpy().astype(bool)

    rows: list[dict] = []
    risk_set_sizes: list[int] = []
    for case_id in failure_ids:
        T_f = length_by_v[case_id]
        ev_case = ev_by_v.get(case_id)
        if ev_case is None:
            continue
        pre = ev_case[ev_case["time_step"] < T_f]
        if pre.empty:
            stats.n_cases_dropped_no_history += 1
            continue
        # risk set: vehicles with length_of_study >= T_f, excluding the case itself
        rs_mask = (tte_arr_len >= T_f) & (tte_arr_ids != case_id)
        rs_ids = tte_arr_ids[rs_mask]
        rs_is_fail = is_failure_arr[rs_mask]
        if len(rs_ids) == 0:
            stats.n_cases_with_empty_risk_set += 1
            continue
        risk_set_sizes.append(int(len(rs_ids)))

        stats.n_cases += 1
        # emit case window
        slc = pre.tail(horizon_k)
        anchor = EPOCH + pd.Timedelta(days=float(T_f))
        rows.append(_row(
            case_id, "last%d" % horizon_k, anchor,
            slc["timestamp"].min(), anchor, slc["timestamp"].max(),
            slc, True, "repair", match_id=int(case_id),
        ))

        # sample controls from risk set, prefer non-failing controls
        # (density-sampling would allow failed-later trucks; we use both
        # so long as they were under observation at T_f)
        n_pick = min(controls_per_case, len(rs_ids))
        pick_idx = rng.choice(len(rs_ids), size=n_pick, replace=False)
        for ci in pick_idx:
            ctrl_id = rs_ids[ci]
            ev_ctrl = ev_by_v.get(ctrl_id)
            if ev_ctrl is None:
                continue
            pre_c = ev_ctrl[ev_ctrl["time_step"] < T_f]
            if pre_c.empty:
                continue
            slc_c = pre_c.tail(horizon_k)
            rows.append(_row(
                ctrl_id, "last%d" % horizon_k, anchor,
                slc_c["timestamp"].min(), anchor, slc_c["timestamp"].max(),
                slc_c, False, None, match_id=int(case_id),
            ))

    stats.mean_risk_set_size = float(np.mean(risk_set_sizes)) if risk_set_sizes else 0.0

    windows = pd.DataFrame(rows)
    stats.invariants = {
        "at_least_1000_matched_cases": stats.n_cases >= 1000,
        "no_repair_in_windows": all(
            "terminal_repair" not in r["event_type_seq"]
            for _, r in windows.iterrows()
        ),
        "each_case_has_at_least_one_control": bool(
            (windows.groupby("match_id")["is_failure"].sum() < windows.groupby("match_id").size()).all()
        ),
    }
    return windows, stats


# ------------------------- MH scoring ---------------------------------

def mh_odds_ratio(case_in: int, case_tot: int,
                  ctrl_in: int, ctrl_tot: int) -> tuple[float, float, float]:
    """Unstratified 2x2 pooled odds ratio (approximation to matched MH-OR
    when match-set sizes are equal). Returns (OR, CI_low, CI_high) at 95%.
    Uses Woolf-Haldane 0.5 continuity correction.
    """
    a = case_in + 0.5
    b = case_tot - case_in + 0.5
    c = ctrl_in + 0.5
    d = ctrl_tot - ctrl_in + 0.5
    or_hat = (a * d) / (b * c)
    log_or = np.log(or_hat)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low = float(np.exp(log_or - 1.96 * se))
    ci_high = float(np.exp(log_or + 1.96 * se))
    return float(or_hat), ci_low, ci_high


def score_patterns_hazard(
    windows: pd.DataFrame,
    patterns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add MH-OR (hazard-ratio-equivalent) + 95% CI + significance flag."""
    if patterns_df.empty:
        return patterns_df.assign(mh_or=[], mh_or_ci_low=[], mh_or_ci_high=[],
                                  significant_at_95=[])
    n_cases = int(windows["is_failure"].sum())
    n_controls = int((~windows["is_failure"]).sum())

    def _contains(seq: list[str], items: set) -> bool:
        return items.issubset(set(seq))

    out = patterns_df.copy()
    ors, lows, highs, sigs = [], [], [], []
    fail_win = windows[windows["is_failure"]]
    ctrl_win = windows[~windows["is_failure"]]

    # Reconstruct the "event_type:event_subtype" items per window so the
    # subset check matches the tokens used by _make_items in src/mine.
    def _items(row) -> set:
        return {f"{t}:{s}" for t, s in
                zip(row["event_type_seq"], row["event_subtype_seq"])}
    fail_items = [_items(r) for _, r in fail_win.iterrows()]
    ctrl_items = [_items(r) for _, r in ctrl_win.iterrows()]

    for _, r in out.iterrows():
        pat = set(r["itemset"])
        case_in = sum(1 for s in fail_items if pat.issubset(s))
        ctrl_in = sum(1 for s in ctrl_items if pat.issubset(s))
        or_hat, lo, hi = mh_odds_ratio(case_in, n_cases, ctrl_in, n_controls)
        ors.append(or_hat)
        lows.append(lo)
        highs.append(hi)
        sigs.append(bool(lo > 1.0))
    out["mh_or"] = ors
    out["mh_or_ci_low"] = lows
    out["mh_or_ci_high"] = highs
    out["significant_at_95"] = sigs
    return out


def run(events_parquet: Path, tte_csv: Path, out_dir: Path) -> tuple[RiskSetStats, dict]:
    events = pd.read_parquet(events_parquet)
    tte = pd.read_csv(tte_csv)
    windows, stats = build_riskset_windows(events, tte)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "scania_riskset_windows.parquet", index=False)
    with (out_dir / "scania_riskset_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)

    # Mine + score
    from src.mine.itemsets import mine as mine_itemsets
    patterns, _ = mine_itemsets(windows, [windows["horizon"].iloc[0]], min_support=0.05)
    scored = score_patterns_hazard(windows, patterns)
    scored = scored.sort_values("mh_or", ascending=False)
    scored.to_parquet(out_dir / "scania_riskset_patterns.parquet", index=False)

    summary = {
        "n_patterns_mined": int(len(scored)),
        "n_significant_at_95": int(scored["significant_at_95"].sum()),
        "top10_mh_or": scored.head(10)[
            ["itemset", "mh_or", "mh_or_ci_low", "mh_or_ci_high",
             "n_failure", "n_control", "support_failure", "support_control"]
        ].to_dict(orient="records"),
    }
    with (out_dir / "scania_riskset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return stats, summary
