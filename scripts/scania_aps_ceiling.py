"""SCANIA APS Failure ceiling test.

APS Failure at Scania Trucks (UCI 421, IDA 2016 challenge). One row per
truck, 171 features including 7 histogram-encoded feature groups
(10 bins each) plus 100 single-value counters. Binary label
"class" = pos (APS system failure) / neg.

This is CROSS-SECTIONAL, not sequential, so the paper's frequent-
pattern-mining pipeline does not apply. We include APS here as an
INDEPENDENT ceiling test on SCANIA-family data: if a compact
LightGBM on the same anonymised histogram/counter format reaches
AUROC 0.9+ on this task, that reinforces the §7.2 boundary reading
of Component X: the low AUROC on Component X is a signal-availability
limit of the specific readout cadence in that release, not a
fundamental limit of SCANIA telemetry data.

Emits results/tables/scania_aps_ceiling_diagnostic.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(r"E:/tmp/scania_aps/aps_failure_training_set.csv")
TEST = Path(r"E:/tmp/scania_aps/aps_failure_test_set.csv")
OUT = ROOT / "results" / "tables"
RNG_SEED = 20260828


def load(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path, skiprows=20, na_values="na")
    y = (df["class"] == "pos").astype(int).to_numpy()
    X_df = df.drop(columns=["class"]).fillna(0.0)
    return X_df.to_numpy(dtype=float), y, list(X_df.columns)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, feat_cols = load(TRAIN)
    X_te, y_te, feat_cols_te = load(TEST)
    assert feat_cols == feat_cols_te, "train/test schema differ"

    print(f"Train {X_tr.shape}  test {X_te.shape}")
    print(f"Train pos rate {y_tr.mean():.4f}  test pos rate {y_te.mean():.4f}")

    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    results = {
        "dataset": "SCANIA APS Failure (UCI 421 / IDA 2016)",
        "n_features": len(feat_cols),
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "train_pos_rate": float(y_tr.mean()),
        "test_pos_rate":  float(y_te.mean()),
    }

    lgb = LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, random_state=RNG_SEED,
        class_weight="balanced", verbose=-1,
    )
    lgb.fit(X_tr, y_tr)
    p_te = lgb.predict_proba(X_te)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    results["lgbm"] = {
        "auroc":            float(roc_auc_score(y_te, p_te)),
        "auprc":            float(average_precision_score(y_te, p_te)),
        "f1_at_0.5":        float(f1_score(y_te, pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_te, pred, zero_division=0)),
        "recall_at_0.5":    float(recall_score(y_te, pred, zero_division=0)),
    }

    lr = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=RNG_SEED)),
    ])
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    pred_lr = (p_lr >= 0.5).astype(int)
    results["lr_on_same_features"] = {
        "auroc":            float(roc_auc_score(y_te, p_lr)),
        "auprc":            float(average_precision_score(y_te, p_lr)),
        "f1_at_0.5":        float(f1_score(y_te, pred_lr, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_te, pred_lr, zero_division=0)),
        "recall_at_0.5":    float(recall_score(y_te, pred_lr, zero_division=0)),
    }

    importances = pd.Series(lgb.feature_importances_, index=feat_cols)
    results["top_features"] = importances.sort_values(ascending=False).head(15).to_dict()

    (OUT / "scania_aps_ceiling_diagnostic.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
