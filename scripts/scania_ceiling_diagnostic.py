"""SCANIA ceiling diagnostic (per Fable + GPTConsult consensus).

Build ~50 structured features per vehicle from the raw readouts using
histogram-aware descriptors (Wasserstein-1, signed centroid shift,
tail-mass change, entropy change, EMD slope) plus counter rates.
Train LightGBM on the SAME temporal split as the pattern-mining
pipeline. This decides whether SCANIA is a representation-loss case
(the pattern pipeline is fixable via histogram-aware tokenization)
or a real signal-absence case (SCANIA becomes a legitimate boundary).

Key design decisions:

- **Baseline is causal**: each vehicle's own EARLY history (first
  window_baseline_frac readouts) defines its healthy distribution.
- **Anchor per vehicle**: the LAST readout timestamp of that vehicle
  is the "prediction time" (mirrors what the pattern pipeline sees).
- **Window**: the last LAST_K readouts (or all available if shorter)
  strictly before the anchor.
- **Temporal split** on the anchor timestamp, matching the pattern
  pipeline's cutoff of 2020-01-01.

Emits results/tables/scania_ceiling_diagnostic.json with the
headline AUROC / AUPRC on the held-out test set.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
READOUTS = Path(r"E:/tmp/scania/train_operational_readouts.csv")
TTE = Path(r"E:/tmp/scania/train_tte.csv")
OUT = ROOT / "results" / "tables"

LAST_K = 20
BASELINE_FRAC = 0.30
CUTOFF = pd.Timestamp("2020-01-01")
EPOCH = pd.Timestamp("2019-01-01 00:00:00")
RNG_SEED = 20260828


def _feature_groups(cols: list[str]) -> dict[str, list[str]]:
    """Group columns by feature prefix (before underscore)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in cols:
        m = re.match(r"^(\d+)(?:_(\d+))?$", c)
        if not m:
            continue
        prefix = m.group(1)
        bin_idx = m.group(2)
        if bin_idx is not None:
            groups[prefix].append(c)
    # Sort columns within each histogram group by numeric bin index
    for p in groups:
        groups[p].sort(key=lambda c: int(c.split("_")[1]) if "_" in c else 0)
    return dict(groups)


def _wasserstein1_ordered(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 distance between two discrete distributions on an
    ordered bin axis. Both p, q sum to 1."""
    cp = np.cumsum(p)
    cq = np.cumsum(q)
    return float(np.abs(cp - cq).sum())


def _normalize(v: np.ndarray) -> np.ndarray | None:
    s = v.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return v / s


def _shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _histogram_features(
    baseline_hist: np.ndarray,
    window_hists: list[np.ndarray],
    prefix: str,
) -> dict[str, float]:
    """Descriptors of the window's histograms vs the vehicle's baseline."""
    n_bins = len(baseline_hist)
    axis = np.arange(n_bins, dtype=float)

    p0 = _normalize(baseline_hist)
    if p0 is None:
        return {}

    centroid0 = float((axis * p0).sum())
    entropy0 = _shannon_entropy(p0)
    tail0 = float(p0[n_bins // 2:].sum())

    dists, centroids, entropies, tails = [], [], [], []
    for h in window_hists:
        p = _normalize(h)
        if p is None:
            continue
        dists.append(_wasserstein1_ordered(p, p0))
        centroids.append((axis * p).sum() - centroid0)
        entropies.append(_shannon_entropy(p) - entropy0)
        tails.append(p[n_bins // 2:].sum() - tail0)
    if not dists:
        return {}
    dists = np.array(dists)
    centroids = np.array(centroids)
    entropies = np.array(entropies)
    tails = np.array(tails)

    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    return {
        f"h{prefix}_emd_last": float(dists[-1]),
        f"h{prefix}_emd_max": float(dists.max()),
        f"h{prefix}_emd_slope": _slope(dists),
        f"h{prefix}_centroid_shift_last": float(centroids[-1]),
        f"h{prefix}_centroid_shift_slope": _slope(centroids),
        f"h{prefix}_entropy_shift_last": float(entropies[-1]),
        f"h{prefix}_tail_shift_last": float(tails[-1]),
        f"h{prefix}_tail_shift_slope": _slope(tails),
    }


def _counter_features(
    baseline_series: np.ndarray, window_series: np.ndarray, name: str,
) -> dict[str, float]:
    """Rate + baseline-relative shift for a single counter feature."""
    if len(window_series) < 2:
        return {f"c{name}_rate": 0.0, f"c{name}_delta_vs_base": 0.0}
    rate = float(np.diff(window_series).mean())
    base_rate = float(np.diff(baseline_series).mean()) if len(baseline_series) >= 2 else 0.0
    return {
        f"c{name}_rate": rate,
        f"c{name}_delta_vs_base": rate - base_rate,
        f"c{name}_last_val": float(window_series[-1]),
    }


def build_vehicle_features() -> pd.DataFrame:
    print("Loading readouts + TTE...", flush=True)
    readouts = pd.read_csv(READOUTS)
    tte = pd.read_csv(TTE)
    readouts = readouts.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

    feature_cols = [c for c in readouts.columns if c not in ("vehicle_id", "time_step")]
    hist_groups = _feature_groups(feature_cols)
    counter_cols = [c for c in feature_cols if c not in {b for g in hist_groups.values() for b in g}]

    print(f"Histogram groups: {list(hist_groups.keys())}", flush=True)
    print(f"Counter columns: {counter_cols}", flush=True)

    rows: list[dict] = []
    labels = dict(zip(tte["vehicle_id"], tte["in_study_repair"]))
    for vid, ent in readouts.groupby("vehicle_id", sort=False):
        n = len(ent)
        if n < 2:
            continue
        n_baseline = max(1, int(n * BASELINE_FRAC))
        base = ent.iloc[:n_baseline]
        window = ent.iloc[-min(LAST_K, n):]
        # exclude window from baseline where possible
        if n > LAST_K + n_baseline:
            base = ent.iloc[:n_baseline]
        else:
            # small vehicle: baseline is first half of pre-window
            pre_window = ent.iloc[:-min(LAST_K, n)]
            if len(pre_window) >= 1:
                base = pre_window
        anchor_ts = EPOCH + pd.Timedelta(days=float(ent["time_step"].max()))

        feats: dict[str, float | str | int] = {
            "vehicle_id": int(vid),
            "anchor_ts": anchor_ts,
            "label": int(labels.get(vid, 0)),
            "n_readouts": n,
        }
        # histogram features
        for prefix, cols in hist_groups.items():
            base_h = base[cols].sum(axis=0).to_numpy(dtype=float)
            win_hists = [row.to_numpy(dtype=float) for _, row in window[cols].iterrows()]
            feats.update(_histogram_features(base_h, win_hists, prefix))
        # counter features
        for c in counter_cols:
            base_series = base[c].to_numpy(dtype=float)
            win_series = window[c].to_numpy(dtype=float)
            feats.update(_counter_features(base_series, win_series, c))
        rows.append(feats)

    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = build_vehicle_features()
    print(f"Feature frame: {feats.shape}", flush=True)
    print(f"Label distribution: {feats['label'].value_counts().to_dict()}", flush=True)

    train = feats[feats["anchor_ts"] < CUTOFF]
    test = feats[feats["anchor_ts"] >= CUTOFF]
    print(f"Train: {len(train)}  Test: {len(test)}", flush=True)

    feat_cols = [c for c in feats.columns
                 if c not in {"vehicle_id", "anchor_ts", "label"}]
    X_train = train[feat_cols].fillna(0.0).to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_test = test[feat_cols].fillna(0.0).to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=int)

    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    results = {"n_features": len(feat_cols),
               "n_train": int(len(train)), "n_test": int(len(test)),
               "train_rate": float(y_train.mean()),
               "test_rate": float(y_test.mean())}

    # LightGBM ceiling
    lgb = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         min_child_samples=20, random_state=RNG_SEED,
                         class_weight="balanced", verbose=-1)
    lgb.fit(X_train, y_train)
    p_test = lgb.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.5).astype(int)
    results["lgbm"] = {
        "auroc": float(roc_auc_score(y_test, p_test)),
        "auprc": float(average_precision_score(y_test, p_test)),
        "f1_at_0.5": float(f1_score(y_test, pred, zero_division=0)),
    }

    # LR baseline for comparison
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=RNG_SEED))])
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (p_lr >= 0.5).astype(int)
    results["lr_on_same_features"] = {
        "auroc": float(roc_auc_score(y_test, p_lr)),
        "auprc": float(average_precision_score(y_test, p_lr)),
        "f1_at_0.5": float(f1_score(y_test, pred_lr, zero_division=0)),
    }

    # feature importance (top 15)
    importances = pd.Series(lgb.feature_importances_, index=feat_cols)
    results["top_features"] = importances.sort_values(ascending=False).head(15).to_dict()

    (OUT / "scania_ceiling_diagnostic.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
