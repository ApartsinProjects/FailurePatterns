"""Trend / severity discrete-event encoding of the PhysioNet sepsis cohort.

The static threshold-crossing encoding (scripts/ingest_sepsis.py) carries no
structural signal once window length is matched: the pre-onset predictability is
window truncation, not a mineable pattern. Slow physiological deterioration lives
in the DIRECTION and MAGNITUDE of change, so here we encode transition events:

  * direction transitions: ``Chan:rising`` / ``Chan:falling`` emitted at the hour
    a sustained rising/falling run BEGINS (change over the last W hours exceeds
    +/- DELTA population-IQR units, and the previous step was not already in that
    direction). Marks trend onsets rather than every repeated hour.
  * severity-band entry: ``Chan:high`` / ``Chan:low`` when a value first enters an
    abnormal band from normal, and ``Chan:severe`` when it enters an extreme band.
    Marks state changes rather than sustained presence.

Events fire only at ACTUAL observations (no forward-filled hour emits an event),
so the stream is sparse and event-like. The outcome (terminal_failure, sepsis)
is placed at the first SepsisLabel==1 hour, which already carries the challenge's
built-in six-hour lead. Timestamps and patient spacing follow ingest_sepsis.py.

Usage:
    python scripts/ingest_sepsis_trend.py --src <dir of .psv> [--limit N]
Writes data/processed/sepsis_trend_events.parquet (+ _load_stats.json).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_EPOCH = pd.Timestamp("2015-01-01 00:00:00")
PATIENT_SPACING = pd.Timedelta(hours=1)
W = 6          # hours look-back for direction
DELTA = 0.5    # population-IQR units for a rising/falling step

CHANNELS = ["HR", "Resp", "SBP", "MAP", "O2Sat", "Temp",
            "Lactate", "Creatinine", "WBC", "Platelets", "BUN", "pH"]

# severity bands: (abnormal test, severe test) as callables on a scalar value
def _band(chan, v):
    if np.isnan(v):
        return None
    T = {
        "HR":         (lambda x: x > 100 or x < 60,  lambda x: x > 130 or x < 40),
        "Resp":       (lambda x: x > 22,             lambda x: x > 30),
        "SBP":        (lambda x: x < 100,            lambda x: x < 90),
        "MAP":        (lambda x: x < 65,             lambda x: x < 55),
        "O2Sat":      (lambda x: x < 92,             lambda x: x < 88),
        "Temp":       (lambda x: x > 38 or x < 36,   lambda x: x > 39.5 or x < 35),
        "Lactate":    (lambda x: x > 2,              lambda x: x > 4),
        "Creatinine": (lambda x: x > 1.2,            lambda x: x > 2.0),
        "WBC":        (lambda x: x > 12 or x < 4,    lambda x: x > 20 or x < 2),
        "Platelets":  (lambda x: x < 150,            lambda x: x < 100),
        "BUN":        (lambda x: x > 20,             lambda x: x > 40),
        "pH":         (lambda x: x < 7.35,           lambda x: x < 7.25),
    }
    ab, sev = T[chan]
    if sev(v):
        return "severe"
    if ab(v):
        # direction of the abnormality for readability of high/low
        return "abn"
    return "normal"


def _pid(path):
    m = re.search(r"p(\d+)\.psv$", Path(path).name)
    return int(m.group(1)) if m else -1


def _iqr_scales(files, sample=400):
    acc = {c: [] for c in CHANNELS}
    for f in files[:sample]:
        try:
            d = pd.read_csv(f, sep="|")
        except Exception:
            continue
        for c in CHANNELS:
            if c in d.columns:
                acc[c] += list(d[c].dropna().values)
    scale = {}
    for c in CHANNELS:
        if acc[c]:
            iqr = np.subtract(*np.percentile(acc[c], [75, 25]))
            scale[c] = float(iqr) if iqr > 0 else 1.0
        else:
            scale[c] = 1.0
    return scale


def ingest(src: Path, limit):
    files = sorted(glob.glob(str(src / "p*.psv")))
    if limit:
        files = files[:limit]
    scale = _iqr_scales(files)
    recs = []
    n_patients = n_septic = 0
    for path in files:
        pid = _pid(path)
        try:
            d = pd.read_csv(path, sep="|")
        except Exception:
            continue
        if d.empty or "ICULOS" not in d.columns:
            continue
        n_patients += 1
        iculos = d["ICULOS"].to_numpy()
        base = BASE_EPOCH + pid * PATIENT_SPACING
        ts = base + pd.to_timedelta(iculos, unit="h")

        for c in CHANNELS:
            if c not in d.columns:
                continue
            v = d[c].to_numpy(dtype=float)
            obs = np.where(~np.isnan(v))[0]      # actual observations only
            if len(obs) == 0:
                continue
            prev_dir = 0
            prev_band = None
            for k, t in enumerate(obs):
                # --- direction transition (vs last obs at least W hours earlier) ---
                past = obs[obs <= t - W]          # obs indices are hour rows
                if len(past):
                    p = past[-1]
                    dv = (v[t] - v[p]) / scale[c]
                    d_dir = 1 if dv > DELTA else (-1 if dv < -DELTA else 0)
                    if d_dir != 0 and d_dir != prev_dir:
                        recs.append((ts[t], pid, c, "rising" if d_dir > 0 else "falling"))
                    prev_dir = d_dir
                # --- severity-band entry ---
                band = _band(c, v[t])
                if band in ("abn", "severe") and band != prev_band:
                    recs.append((ts[t], pid, c, "high" if band == "abn" else "severe"))
                prev_band = band

        if "SepsisLabel" in d.columns and (d["SepsisLabel"] == 1).any():
            first = int(np.argmax(d["SepsisLabel"].to_numpy() == 1))
            recs.append((ts[first], pid, "terminal_failure", "sepsis"))
            n_septic += 1

    ev = pd.DataFrame(recs, columns=["timestamp", "entity_id", "event_type", "event_subtype"])
    ev = ev.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    nonf = ev[ev["event_type"] != "terminal_failure"]
    stats = {
        "encoding": "trend+severity transition events",
        "n_patients": n_patients, "n_septic": n_septic,
        "sepsis_prevalence": round(n_septic / max(1, n_patients), 4),
        "n_events": int(len(ev)),
        "iqr_scale": {k: round(x, 3) for k, x in scale.items()},
        "subtype_counts": nonf["event_subtype"].value_counts().to_dict(),
        "token_counts": (nonf["event_type"] + ":" + nonf["event_subtype"])
            .value_counts().head(40).to_dict(),
    }
    return ev, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ev, stats = ingest(Path(a.src), a.limit)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out / "sepsis_trend_events.parquet", index=False)
    (out / "sepsis_trend_load_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
