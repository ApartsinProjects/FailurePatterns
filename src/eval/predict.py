"""Phase 6: head-to-head predictive evaluation on Azure PdM windows.

Four feature sets compared on the SAME temporally-held-out split:

    A. event_count      -- single feature: n_events in the window
    B. itemsets_only    -- binary presence of each train-mined itemset
    C. sequences_only   -- binary presence of each train-mined sequence
    D. combined         -- A + B + C

Mining runs ONLY on training windows (anchor timestamp before CUTOFF) so
no test-window information leaks into feature selection. All four models
are logistic regression fit on TRAIN and evaluated on TEST.

Per (horizon, feature-set) we report AUROC, AUPRC, F1@0.5, precision@0.5,
recall@0.5, n_train, n_test, n_features. All numbers land in a single
results DataFrame + JSON.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.mine.itemsets import mine as mine_itemsets
from src.mine.sequences import mine as mine_sequences

# -------------------------- config ----------------------------------------

CUTOFF = pd.Timestamp("2015-09-01")
HORIZONS = ("24h", "last5", "last10")
MIN_SUPPORT = 0.05
LR_C = 1.0  # inverse regularization; ridge-ish. Small dataset, keep steady.
LR_MAX_ITER = 2000
RNG_SEED = 20260828


# -------------------------- data model ------------------------------------

@dataclass
class SplitStats:
    n_train_windows: int
    n_test_windows: int
    train_failure_rate: float
    test_failure_rate: float


@dataclass
class HorizonEval:
    horizon: str
    n_train_patterns_itemset: int
    n_train_patterns_sequence: int
    n_train_patterns_itemset_survived: int
    n_train_patterns_sequence_survived: int
    results_by_feature_set: dict[str, dict] = field(default_factory=dict)


@dataclass
class PredictStats:
    cutoff: str
    split: SplitStats
    by_horizon: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# -------------------------- feature engineering ---------------------------

def _make_items(seq_types: list[str], seq_subtypes: list[str]) -> list[str]:
    return [f"{t}:{s}" for t, s in zip(seq_types, seq_subtypes)]


def _itemset_features(
    windows: pd.DataFrame, patterns: list[frozenset]
) -> np.ndarray:
    """One binary column per pattern. All-zero if pattern list is empty."""
    if not patterns:
        return np.zeros((len(windows), 0), dtype=np.int8)
    X = np.zeros((len(windows), len(patterns)), dtype=np.int8)
    for i, row in enumerate(windows.itertuples(index=False)):
        items = set(_make_items(row.event_type_seq, row.event_subtype_seq))
        for j, p in enumerate(patterns):
            if p.issubset(items):
                X[i, j] = 1
    return X


def _sequence_present(seq: list[str], transaction: list[str]) -> bool:
    i = 0
    for item in transaction:
        if item == seq[i]:
            i += 1
            if i == len(seq):
                return True
    return False


def _sequence_features(
    windows: pd.DataFrame, patterns: list[list[str]]
) -> np.ndarray:
    if not patterns:
        return np.zeros((len(windows), 0), dtype=np.int8)
    X = np.zeros((len(windows), len(patterns)), dtype=np.int8)
    for i, row in enumerate(windows.itertuples(index=False)):
        tx = _make_items(row.event_type_seq, row.event_subtype_seq)
        for j, p in enumerate(patterns):
            if _sequence_present(p, tx):
                X[i, j] = 1
    return X


def _count_features(windows: pd.DataFrame) -> np.ndarray:
    return windows["n_events"].to_numpy(dtype=float).reshape(-1, 1)


# -------------------------- eval ------------------------------------------

def _leadtime_stats(
    windows_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Lead time = anchor - last_event_ts for TP failure windows.

    Returns median / IQR / count in seconds. NaN if no TPs.
    """
    tp_mask = (y_test == 1) & (y_pred == 1)
    if not tp_mask.any():
        return {
            "n_tp": 0,
            "median_lead_seconds": float("nan"),
            "p25_lead_seconds": float("nan"),
            "p75_lead_seconds": float("nan"),
        }
    tp_rows = windows_test.iloc[np.where(tp_mask)[0]]
    lead = (tp_rows["anchor"] - tp_rows["last_event_ts"]).dt.total_seconds()
    return {
        "n_tp": int(len(tp_rows)),
        "median_lead_seconds": float(lead.median()),
        "p25_lead_seconds": float(lead.quantile(0.25)),
        "p75_lead_seconds": float(lead.quantile(0.75)),
    }


def _fit_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    windows_test: pd.DataFrame | None = None,
) -> dict:
    n_feat = X_train.shape[1]
    if n_feat == 0:
        return {
            "n_features": 0,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1_at_0.5": float("nan"),
            "precision_at_0.5": float("nan"),
            "recall_at_0.5": float("nan"),
            "n_tp": 0,
            "median_lead_seconds": float("nan"),
            "p25_lead_seconds": float("nan"),
            "p75_lead_seconds": float("nan"),
            "note": "no features",
        }
    model = Pipeline([
        ("sc", StandardScaler(with_mean=n_feat < 500)),
        ("lr", LogisticRegression(
            C=LR_C, max_iter=LR_MAX_ITER, random_state=RNG_SEED,
            solver="liblinear" if n_feat <= 200 else "lbfgs",
        )),
    ])
    # If TRAIN has only one class (degenerate), skip.
    if len(np.unique(y_train)) < 2:
        return {
            "n_features": n_feat,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1_at_0.5": float("nan"),
            "precision_at_0.5": float("nan"),
            "recall_at_0.5": float("nan"),
            "n_tp": 0,
            "median_lead_seconds": float("nan"),
            "p25_lead_seconds": float("nan"),
            "p75_lead_seconds": float("nan"),
            "note": "train has one class",
        }
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    lead = (
        _leadtime_stats(windows_test, y_test, y_pred)
        if windows_test is not None
        else {"n_tp": 0, "median_lead_seconds": float("nan"),
              "p25_lead_seconds": float("nan"), "p75_lead_seconds": float("nan")}
    )
    return {
        "n_features": n_feat,
        "auroc": float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "f1_at_0.5": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_test, y_pred, zero_division=0)),
        **lead,
    }


# -------------------------- main pipeline ---------------------------------

def evaluate(
    windows: pd.DataFrame,
    horizons: Iterable[str] = HORIZONS,
    cutoff: pd.Timestamp = CUTOFF,
    min_support: float = MIN_SUPPORT,
) -> tuple[pd.DataFrame, PredictStats]:
    stats = PredictStats(
        cutoff=str(cutoff),
        split=SplitStats(0, 0, 0.0, 0.0),
    )
    rows: list[dict] = []

    train_mask = windows["anchor"] < cutoff
    train_all = windows[train_mask]
    test_all = windows[~train_mask]
    stats.split = SplitStats(
        n_train_windows=int(len(train_all)),
        n_test_windows=int(len(test_all)),
        train_failure_rate=float(train_all["is_failure"].mean()) if len(train_all) else 0.0,
        test_failure_rate=float(test_all["is_failure"].mean()) if len(test_all) else 0.0,
    )

    for hname in horizons:
        w_train = train_all[train_all["horizon"] == hname]
        w_test = test_all[test_all["horizon"] == hname]
        if len(w_train) == 0 or len(w_test) == 0:
            continue

        # Mine on TRAIN only.
        it_train, it_stats = mine_itemsets(w_train, [hname], min_support=min_support)
        sq_train, sq_stats = mine_sequences(w_train, [hname], min_support=min_support)

        # Feature-select: use patterns that beat their own null on TRAIN.
        it_survivors = it_train[it_train["survives_permutation_null"]] if not it_train.empty else it_train
        sq_survivors = sq_train[sq_train["survives_shuffle_null"]] if not sq_train.empty else sq_train

        it_patterns = [frozenset(x) for x in it_survivors["itemset"]] if not it_survivors.empty else []
        sq_patterns = [list(x) for x in sq_survivors["sequence"]] if not sq_survivors.empty else []

        # Build feature matrices.
        Xc_tr = _count_features(w_train)
        Xc_te = _count_features(w_test)
        Xi_tr = _itemset_features(w_train, it_patterns)
        Xi_te = _itemset_features(w_test, it_patterns)
        Xs_tr = _sequence_features(w_train, sq_patterns)
        Xs_te = _sequence_features(w_test, sq_patterns)

        y_train = w_train["is_failure"].astype(int).to_numpy()
        y_test = w_test["is_failure"].astype(int).to_numpy()

        Xcomb_tr = np.hstack([Xc_tr, Xi_tr, Xs_tr])
        Xcomb_te = np.hstack([Xc_te, Xi_te, Xs_te])

        feat_sets = {
            "event_count":     (Xc_tr, Xc_te),
            "itemsets_only":   (Xi_tr, Xi_te),
            "sequences_only":  (Xs_tr, Xs_te),
            "combined":        (Xcomb_tr, Xcomb_te),
        }
        by_set: dict[str, dict] = {}
        for name, (Xtr, Xte) in feat_sets.items():
            res = _fit_eval(Xtr, y_train, Xte, y_test, windows_test=w_test)
            by_set[name] = res
            rows.append({
                "horizon": hname,
                "feature_set": name,
                "n_train_windows": len(w_train),
                "n_test_windows": len(w_test),
                "train_failure_rate": float(y_train.mean()),
                "test_failure_rate": float(y_test.mean()),
                "n_features": res["n_features"],
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "f1_at_0.5": res["f1_at_0.5"],
                "precision_at_0.5": res["precision_at_0.5"],
                "recall_at_0.5": res["recall_at_0.5"],
                "n_tp": res.get("n_tp", 0),
                "median_lead_seconds": res.get("median_lead_seconds"),
                "p25_lead_seconds": res.get("p25_lead_seconds"),
                "p75_lead_seconds": res.get("p75_lead_seconds"),
            })

        stats.by_horizon[hname] = {
            "n_itemset_mined": int(len(it_train)),
            "n_itemset_survived": int(len(it_survivors)),
            "n_sequence_mined": int(len(sq_train)),
            "n_sequence_survived": int(len(sq_survivors)),
            "results_by_feature_set": by_set,
        }

    return pd.DataFrame(rows), stats


def run(
    windows_parquet: Path,
    out_dir: Path,
    output_stem: str = "azure_predictive",
    horizons: Iterable[str] = HORIZONS,
    cutoff: pd.Timestamp = CUTOFF,
    min_support: float = MIN_SUPPORT,
) -> PredictStats:
    windows = pd.read_parquet(windows_parquet)
    results, stats = evaluate(
        windows, horizons=horizons, cutoff=cutoff, min_support=min_support,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir / f"{output_stem}.parquet", index=False)
    with (out_dir / f"{output_stem}_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
