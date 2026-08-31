"""Does slow fault development live in the CONTINUOUS SCADA channels that the
discrete alarm logs do not capture?

The discrete alarm codes we mine are threshold tripwires: they fire only after
a measured quantity crosses a limit. If a fault develops slowly, the signal
should appear earlier in a continuous channel. We test the most direct case:
generator-bearing temperature, load-controlled.

For each Kelmarsh turbine:
  1. Fit bearing_temp ~ power + wind + ambient on all data (the expected
     temperature for the current load and weather).
  2. residual = actual - expected  (how much hotter than it should be running).
  3. For each Forced outage, take the DEVELOPING window [outage-48h, outage-6h]
     (excluding the acute last 6h) and record the median residual and its slope.
  4. Draw matched control windows from clean regions (no outage within 72h).
  5. Report AUROC of the developing-window residual (and slope) discriminating
     pre-outage from control windows. AUROC well above 0.5 means slow
     degradation is present in the continuous channel; ~0.5 means it is not.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCADA_DIR = Path(r"E:/tmp/kelmarsh/turbdata")
GUARD = pd.Timedelta(days=14)

DATE = "# Date and time"
POWER = "Power (kW)"
WIND = "Wind speed (m/s)"
AMB = "Nacelle ambient temperature (°C)"
CHANNELS = [
    "Generator bearing front temperature (°C)",
    "Generator bearing rear temperature (°C)",
    "Gear oil temperature (°C)",
    "Rear bearing temperature (°C)",
    "Stator temperature 1 (°C)",
]
# (developing window low, high) pairs to sweep: 48h-6h and 14d-1d
WINDOWS = [(pd.Timedelta(hours=48), pd.Timedelta(hours=6), "48h-6h"),
           (pd.Timedelta(days=14), pd.Timedelta(days=1), "14d-1d")]


def _turbine_no(path):
    m = re.search(r"Turbine_Data_Kelmarsh_(\d+)_", Path(path).name)
    return int(m.group(1)) if m else -1


def analyse():
    ev = pd.read_parquet(ROOT / "data/processed/kelmarsh_events.parquet")
    fo = ev[ev["event_type"] == "terminal_failure"]
    # preload each turbine once with all channels
    frames = {}
    for path in sorted(glob.glob(str(SCADA_DIR / "Turbine_Data_Kelmarsh_*.csv"))):
        tno = _turbine_no(path)
        cols = [DATE, POWER, WIND, AMB] + [c for c in CHANNELS]
        df = pd.read_csv(path, skiprows=9, usecols=lambda c: c in cols,
                         parse_dates=[DATE], low_memory=False).sort_values(DATE)
        frames[f"T{tno}"] = df

    results = []
    for chan in CHANNELS:
        for lo_td, hi_td, wlabel in WINDOWS:
            res, slope, y = [], [], []
            for ent, df in frames.items():
                if chan not in df.columns:
                    continue
                d = df.dropna(subset=[POWER, WIND, AMB, chan])
                if len(d) < 500:
                    continue
                X = d[[POWER, WIND, AMB]].to_numpy()
                reg = LinearRegression().fit(X, d[chan].to_numpy())
                resid = d[chan].to_numpy() - reg.predict(X)
                ts = d[DATE].to_numpy()
                outs = fo[fo["entity_id"] == ent]["timestamp"].sort_values().to_numpy()

                def feat(anchor):
                    lo = anchor - np.timedelta64(int(lo_td.total_seconds()), "s")
                    hi = anchor - np.timedelta64(int(hi_td.total_seconds()), "s")
                    m = (ts >= lo) & (ts <= hi)
                    if m.sum() < 20:
                        return None
                    r = resid[m]
                    tt = (ts[m] - ts[m][0]) / np.timedelta64(1, "h")
                    return float(np.median(r)), float(np.polyfit(tt, r, 1)[0]) if len(r) > 2 else 0.0

                for a in outs:
                    f = feat(a)
                    if f:
                        res.append(f[0]); slope.append(f[1]); y.append(1)
                rng = np.random.default_rng(20260828)
                for a in rng.choice(ts, size=min(60, len(ts)), replace=False):
                    if len(outs) and ((outs >= a - np.timedelta64(14, "D")) &
                                      (outs <= a + np.timedelta64(14, "D"))).any():
                        continue
                    f = feat(a)
                    if f:
                        res.append(f[0]); slope.append(f[1]); y.append(0)
            y = np.array(y); res = np.array(res); slope = np.array(slope)
            if len(set(y.tolist())) < 2:
                continue
            results.append({
                "channel": chan, "window": wlabel,
                "n_outage": int((y == 1).sum()), "n_control": int((y == 0).sum()),
                "auroc_resid_level": round(float(roc_auc_score(y, res)), 3),
                "auroc_resid_slope": round(float(roc_auc_score(y, slope)), 3),
            })
            print(f"{chan[:34]:34s} {wlabel:6s} n={int((y==1).sum())}/{int((y==0).sum())} "
                  f"AUROC level {results[-1]['auroc_resid_level']} slope {results[-1]['auroc_resid_slope']}",
                  flush=True)
    best = max((r["auroc_resid_level"] for r in results), default=None)
    out = {"load_controlled_on": [POWER, WIND, AMB], "results": results,
           "best_auroc_level": best,
           "reads_as": ("no continuous channel shows load-controlled slow degradation "
                        "before Kelmarsh forced outages" if best and best < 0.6
                        else "at least one continuous channel shows slow degradation")}
    (ROOT / "results/patterns/scada_degradation_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nbest AUROC level:", best, "->", out["reads_as"])
    return 0


if __name__ == "__main__":
    raise SystemExit(analyse())
