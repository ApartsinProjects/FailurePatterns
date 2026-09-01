"""Power calibration for the recency (token-specific timing) assay - the check
Experiment 2 was missing. Analogue of the order power calibration.

Inject a TIMING-ONLY signal into sepsis windows: add a synthetic token T to a
fraction f of BOTH classes with identical presence and multiplicity (one T each)
and no order difference, but with different RECENCY: in positives T is recent
(small age), in controls T is old (large age). Everything else (n_events, span,
global recency, presence, multiplicity, order) is matched. Only token-specific
timing differs.

recency_value = AUROC(P+R real) - AUROC(P+R recency-null) must RISE with f. If it
does, the recency assay provably has power, so the near-zero recency_value on real
traces (Exp 2) is a genuine power-calibrated negative.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from scripts.recency_marked_set import (build_window_recency, _mat_PR, _auc, _recency_null)

T = "INJ:T"


def run(rows, vocab, frac, rng_seed):
    ents = np.array([r["ent"] for r in rows]); y = np.array([r["y"] for r in rows])
    vals = []
    for seed in range(8):
        rng = np.random.default_rng(seed); ue = np.unique(ents); e = ue.copy(); rng.shuffle(e)
        cut = int(len(e) * 0.7); tre = set(e[:cut].tolist())
        trm = np.array([en in tre for en in ents]); tem = ~trm
        if len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        rj = np.random.default_rng(rng_seed + seed)

        def inject(idxs):
            out = []
            for i in idxs:
                r = dict(present=dict(rows[i]["present"]), recency=dict(rows[i]["recency"]),
                         n=rows[i]["n"], span=rows[i]["span"], glast=rows[i]["glast"],
                         y=rows[i]["y"], ent=rows[i]["ent"])
                if rj.random() < frac:
                    r["present"][T] = 1
                    # recent for positives, old for controls; jitter so it is not a constant
                    r["recency"][T] = float(np.log1p(rj.uniform(1, 60))) if r["y"] == 1 \
                        else float(np.log1p(rj.uniform(3600, 36000)))
                out.append(r)
            return out
        tr = inject(np.where(trm)[0]); te = inject(np.where(tem)[0])
        v2 = sorted(set(vocab) | {T})
        ytr, yte = y[trm], y[tem]
        aPR = _auc(_mat_PR(tr, v2), ytr, _mat_PR(te, v2), yte)
        r2 = np.random.default_rng(200 + seed)
        aN = _auc(_mat_PR(tr, v2, _recency_null(tr, v2, r2)), ytr,
                  _mat_PR(te, v2, _recency_null(te, v2, r2)), yte)
        vals.append(aPR - aN)
    vals = np.array(vals)
    return (round(float(vals.mean()), 3), round(float(vals.min()), 3),
            round(float(vals.max()), 3), round(float(np.mean(vals > 0)), 2))


def main():
    win = pd.read_parquet(ROOT / "data/processed/sepsis_trend_windows.parquet")
    win = win[win["horizon"] == "last10"]
    ev = pd.read_parquet(ROOT / "data/processed/sepsis_trend_events.parquet")
    ev = ev[ev["event_type"] != "terminal_failure"]
    rows, _ = build_window_recency(win, ev, 10)
    rows = [r for r in rows if r["n"] == 10]   # length-matched substrate
    vocab = sorted({tk for r in rows for tk in r["present"]})
    print("[Timing power calibration on Sepsis last10, timing-only injection]", flush=True)
    out = {}
    for f in [0.0, 0.1, 0.2, 0.4, 0.8]:
        r = run(rows, vocab, f, 700)
        out[str(f)] = list(r)
        print(f"  frac={f}: recency_value (mean,min,max,frac>0) = {r}", flush=True)
    rises = out["0.8"][0] > out["0.0"][0] + 0.05
    print(f"\nRECENCY ASSAY HAS POWER: {rises}", flush=True)
    (ROOT / "results/patterns/timing_power_calibration.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote results/patterns/timing_power_calibration.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
