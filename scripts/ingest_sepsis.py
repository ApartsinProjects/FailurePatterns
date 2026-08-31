"""Ingest the PhysioNet/CinC Challenge 2019 sepsis cohort into the shared
discrete-event schema (``timestamp, entity_id, event_type, event_subtype``).

Each patient file ``p######.psv`` is one hourly ICU record: 40 physiological
channels plus a per-hour ``SepsisLabel``. We convert the continuous stream into
a stream of clinical ABNORMALITY events using standard SIRS / qSOFA / SOFA
thresholds (the recognised precursors of clinical deterioration toward sepsis).
For every hour and every channel that crosses its threshold we emit one event
whose subtype is ``Channel:direction`` (e.g. ``Lactate:high``, ``SBP:low``).

The outcome is sepsis onset. The challenge's ``SepsisLabel`` is already shifted
to turn 1 six hours BEFORE the Sepsis-3 onset time, so the first ``SepsisLabel==1``
hour carries a built-in six-hour prediction lead. We place a single
``terminal_failure`` event (subtype ``sepsis``) at that hour. Patients who never
become septic contribute only clean regions (no failure), exactly as healthy
stretches of the six operational datasets do.

Timestamps are synthetic: ICULOS is an integer hour index, so we anchor each
patient at a distinct base time (epoch + patient_index * 60 days) and add ICULOS
hours. Grouping is by entity, so the base only serves to (a) space patients apart
so the downstream anchor-sorted temporal split becomes an entity-disjoint
patient split, and (b) keep the 1-hour spacing within a stay exact.

Usage:
    python scripts/ingest_sepsis.py --src <dir of .psv> [--limit N]

Writes ``data/processed/sepsis_events.parquet`` and ``sepsis_load_stats.json``.
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
# Patients are isolated by entity grouping during window building and by an
# entity-level (patient-id) train/test split downstream, so the base offset only
# needs to keep every patient's anchors distinct and within datetime64[ns] range;
# a 1-hour stride does both (max offset ~848 days across the 20k cohort).
PATIENT_SPACING = pd.Timedelta(hours=1)

# (channel, direction, threshold). direction 'high' -> value > thr; 'low' -> value < thr.
# Thresholds are standard SIRS / qSOFA / SOFA / sepsis-resuscitation cut-points.
RULES = [
    ("HR",              "high", 100.0),   # tachycardia (SIRS)
    ("HR",              "low",   60.0),
    ("Temp",            "high",  38.0),   # fever (SIRS)
    ("Temp",            "low",   36.0),   # hypothermia (SIRS)
    ("Resp",            "high",  22.0),   # tachypnoea (qSOFA >=22; SIRS >20)
    ("SBP",             "low",  100.0),   # qSOFA hypotension (SBP <=100)
    ("MAP",             "low",   65.0),   # sepsis MAP resuscitation target
    ("O2Sat",           "low",   92.0),   # hypoxaemia
    ("Lactate",         "high",   2.0),   # hyperlactataemia (key sepsis marker)
    ("WBC",             "high",  12.0),   # leukocytosis (SIRS)
    ("WBC",             "low",    4.0),   # leukopenia (SIRS)
    ("Creatinine",      "high",   1.2),   # renal dysfunction (SOFA)
    ("Bilirubin_total", "high",   1.2),   # hepatic dysfunction (SOFA)
    ("Platelets",       "low",  150.0),   # coagulation dysfunction (SOFA)
    ("BUN",             "high",  20.0),   # azotaemia
    ("pH",              "low",    7.35),  # acidosis
    ("HCO3",            "low",   22.0),   # metabolic acidosis
    ("FiO2",            "high",   0.5),   # rising oxygen requirement
]


def _pid(path: str) -> int:
    m = re.search(r"p(\d+)\.psv$", Path(path).name)
    return int(m.group(1)) if m else -1


def ingest(src: Path, limit: int | None) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(src / "p*.psv")))
    if limit:
        files = files[:limit]
    recs = []
    n_septic = 0
    n_patients = 0
    onset_iculos = []
    for path in files:
        pid = _pid(path)
        try:
            df = pd.read_csv(path, sep="|")
        except Exception:
            continue
        if df.empty or "ICULOS" not in df.columns:
            continue
        n_patients += 1
        base = BASE_EPOCH + pid * PATIENT_SPACING
        ts = base + pd.to_timedelta(df["ICULOS"].to_numpy(), unit="h")

        # abnormality events
        for chan, direction, thr in RULES:
            if chan not in df.columns:
                continue
            v = df[chan].to_numpy(dtype=float)
            if direction == "high":
                mask = v > thr
            else:
                mask = v < thr
            mask &= ~np.isnan(v)
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            for i in idx:
                recs.append((ts[i], pid, chan, direction))

        # sepsis onset -> terminal_failure at first SepsisLabel == 1 hour
        if "SepsisLabel" in df.columns and (df["SepsisLabel"] == 1).any():
            first = int(np.argmax(df["SepsisLabel"].to_numpy() == 1))
            recs.append((ts[first], pid, "terminal_failure", "sepsis"))
            n_septic += 1
            onset_iculos.append(int(df["ICULOS"].iloc[first]))

    ev = pd.DataFrame(recs, columns=["timestamp", "entity_id", "event_type", "event_subtype"])
    ev = ev.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    stats = {
        "n_patients": n_patients,
        "n_septic": n_septic,
        "sepsis_prevalence": round(n_septic / max(1, n_patients), 4),
        "n_events": int(len(ev)),
        "n_abnormality_events": int((ev["event_type"] != "terminal_failure").sum()),
        "n_terminal_failures": int((ev["event_type"] == "terminal_failure").sum()),
        "event_subtype_counts": ev[ev["event_type"] != "terminal_failure"]
            ["event_subtype"].value_counts().to_dict(),
        "channel_counts": ev[ev["event_type"] != "terminal_failure"]
            ["event_type"].value_counts().to_dict(),
        "onset_iculos_median": float(np.median(onset_iculos)) if onset_iculos else None,
        "onset_iculos_p25_p75": [float(np.percentile(onset_iculos, 25)),
                                 float(np.percentile(onset_iculos, 75))] if onset_iculos else None,
    }
    return ev, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of p######.psv files")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ev, stats = ingest(Path(args.src), args.limit)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out_dir / "sepsis_events.parquet", index=False)
    (out_dir / "sepsis_load_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
