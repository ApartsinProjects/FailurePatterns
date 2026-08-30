"""W6 / W7: prospective full-timeline alarm replay on the wind farms.

This replaces the assumed-prior PPV with a measured prospective
evaluation. For each turbine (held out in turn), an alarm rule learned
ONLY from the other turbines is replayed chronologically over the held-
out turbine's real event stream, with an alarm cooldown, and scored
against the actual forced outages.

Learned rule (per leave-one-turbine-out fold): on the training turbines,
a non-terminal code c is a learned precursor if, within the horizon H
before a forced outage, its firing is followed by a forced outage more
than TAU times the per-firing base rate, with at least MIN_FIRE
supporting firings. On the held-out turbine we raise an alarm whenever a
learned precursor code fires and no alarm fired in the preceding COOLDOWN.

Reported per farm (pooled over held-out turbines):
  precision, recall, false alarms per turbine-month, alarms per outage,
  lead-time median/IQR, and the fraction of outages caught at least
  10 min / 1 h / 6 h / 24 h early.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

H = pd.Timedelta(hours=24)       # prediction horizon
COOLDOWN = pd.Timedelta(hours=24)
TAU = 3.0                        # min lift over base rate
MIN_FIRE = 5


def learn_precursors(train: pd.DataFrame) -> set:
    """Codes whose firing is followed by a forced outage within H at
    > TAU x the base rate, on the training turbines."""
    fo = train[train["event_type"] == "terminal_failure"]
    non = train[train["event_type"] != "terminal_failure"].copy()
    if fo.empty or non.empty:
        return set()
    # base rate = fraction of all non-terminal firings followed by an outage in H
    base_hits = 0
    fo_by_ent = {e: g["timestamp"].sort_values().to_numpy() for e, g in fo.groupby("entity_id")}
    def followed(ent, ts):
        arr = fo_by_ent.get(ent)
        if arr is None:
            return False
        return bool(((arr > np.datetime64(ts)) & (arr <= np.datetime64(ts + H))).any())
    non["code"] = non["event_type"] + ":" + non["event_subtype"].astype(str)
    non["hit"] = [followed(e, t) for e, t in zip(non["entity_id"], non["timestamp"])]
    base = non["hit"].mean()
    if base <= 0:
        return set()
    codes = set()
    for code, g in non.groupby("code"):
        if len(g) >= MIN_FIRE and g["hit"].mean() >= TAU * base:
            codes.add(code)
    return codes


def replay_turbine(test: pd.DataFrame, precursors: set) -> dict:
    non = test[test["event_type"] != "terminal_failure"].copy()
    non["code"] = non["event_type"] + ":" + non["event_subtype"].astype(str)
    non = non.sort_values("timestamp")
    fo = test[test["event_type"] == "terminal_failure"]["timestamp"].sort_values().to_numpy()
    span_days = ((test["timestamp"].max() - test["timestamp"].min()).total_seconds()
                 / 86400.0) if len(test) else 0.0

    alarms = []          # timestamps of raised alarms
    last_alarm = None
    for _, r in non.iterrows():
        if r["code"] not in precursors:
            continue
        t = r["timestamp"]
        if last_alarm is not None and (t - last_alarm) < COOLDOWN:
            continue
        alarms.append(t)
        last_alarm = t

    # score alarms
    tp, fp, leads = 0, 0, []
    for t in alarms:
        nxt = fo[(fo > np.datetime64(t)) & (fo <= np.datetime64(t + H))]
        if len(nxt):
            tp += 1
            leads.append((pd.Timestamp(nxt[0]) - t).total_seconds())
        else:
            fp += 1
    # recall: outages with >=1 alarm within H before
    detected = 0
    for o in fo:
        prior = [t for t in alarms if np.datetime64(t) <= o and o <= np.datetime64(t + H)]
        if prior:
            detected += 1
    return {"n_alarms": len(alarms), "tp": tp, "fp": fp,
            "n_outages": int(len(fo)), "detected": detected,
            "leads": leads, "span_days": span_days}


def evaluate(name: str, events_path: Path) -> dict:
    ev = pd.read_parquet(events_path)
    turbines = sorted(ev["entity_id"].unique())
    agg = {"n_alarms": 0, "tp": 0, "fp": 0, "n_outages": 0, "detected": 0,
           "leads": [], "turbine_months": 0.0}
    learned_union = set()
    for t in turbines:
        train = ev[ev["entity_id"] != t]
        test = ev[ev["entity_id"] == t]
        prec = learn_precursors(train)
        learned_union |= prec
        r = replay_turbine(test, prec)
        agg["n_alarms"] += r["n_alarms"]; agg["tp"] += r["tp"]; agg["fp"] += r["fp"]
        agg["n_outages"] += r["n_outages"]; agg["detected"] += r["detected"]
        agg["leads"].extend(r["leads"])
        agg["turbine_months"] += r["span_days"] / 30.0

    leads = np.array(agg["leads"], dtype=float)
    prec = agg["tp"] / agg["n_alarms"] if agg["n_alarms"] else float("nan")
    rec = agg["detected"] / agg["n_outages"] if agg["n_outages"] else float("nan")
    def frac_ge(sec): return round(float((leads >= sec).mean()), 3) if len(leads) else None
    return {
        "farm": name, "n_turbines": len(turbines),
        "n_learned_precursor_codes_union": len(learned_union),
        "horizon_hours": 24, "cooldown_hours": 24,
        "n_alarms": agg["n_alarms"], "tp_alarms": agg["tp"], "fp_alarms": agg["fp"],
        "n_outages": agg["n_outages"], "outages_detected": agg["detected"],
        "precision": round(prec, 3), "recall": round(rec, 3),
        "false_alarms_per_turbine_month": round(agg["fp"] / max(agg["turbine_months"], 1e-9), 3),
        "alarms_per_outage": round(agg["n_alarms"] / max(agg["n_outages"], 1), 3),
        "lead_median_min": round(float(np.median(leads) / 60), 1) if len(leads) else None,
        "lead_iqr_min": [round(float(np.percentile(leads, 25) / 60), 1),
                         round(float(np.percentile(leads, 75) / 60), 1)] if len(leads) else None,
        "frac_lead_ge_10min": frac_ge(600),
        "frac_lead_ge_1h": frac_ge(3600),
        "frac_lead_ge_6h": frac_ge(6 * 3600),
        "frac_lead_ge_24h": frac_ge(24 * 3600),
    }


def main() -> int:
    out = []
    for name, path in [
        ("Kelmarsh", ROOT / "data/processed/kelmarsh_events.parquet"),
        ("Penmanshiel", ROOT / "data/processed/penmanshiel_events.parquet"),
    ]:
        r = evaluate(name, path)
        out.append(r)
        print(f"[{name}] precision={r['precision']} recall={r['recall']} "
              f"FA/turbine-month={r['false_alarms_per_turbine_month']} "
              f"lead median={r['lead_median_min']}min "
              f">=1h={r['frac_lead_ge_1h']} >=6h={r['frac_lead_ge_6h']}", flush=True)
    (ROOT / "results/patterns/prospective_alarm_replay.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results/patterns/prospective_alarm_replay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
