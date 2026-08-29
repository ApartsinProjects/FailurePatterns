"""W7: strong baseline comparisons on Kelmarsh and Penmanshiel.

For each wind-farm trace, compare three predictors on the same
entity-disjoint discovery/inference split at horizon `last5`:

  (a) MOST-RECENT-EVENT indicator: label as "at risk" whenever the
      last observed event on the entity is any of the top-K
      forced-outage precursor codes seen on the discovery half. This
      is the simplest deployable rule.
  (b) MINED-ITEMSET rule: label as "at risk" whenever the window
      contains any BY-significant itemset mined on discovery.
  (c) UNSUPERVISED-BASELINE (window event count): predict "at risk"
      whenever the window's event count is above the median count of
      discovery-half failure windows.

Report per predictor on the INFERENCE half:
  - precision, recall, F1
  - PPV under the ACTUAL base rate estimated from the pipeline's
    case:control ratio (1:3, so prior P(fail) = 0.25).
  - PPV rescaled to a REALISTIC operational base rate of P(fail)=0.01
    via Bayes with the estimator's sensitivity/specificity.

The rescaling addresses DAMI-review blocker W8: a 3:1 case:control
sampling design fixes the posterior at 0.25 by construction, so it
cannot be reported as the operational P(fail | pattern). We invert
the empirical sensitivity/specificity through Bayes at the true
operational base rate to get an interpretable posterior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.discovery_inference_split import (
    entity_disjoint_split, score_on_inference_half,
    bh_qvalues, by_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets

TARGET_RATE = 0.01  # 1% realistic operational base rate


def _bayes_ppv(tp: int, fp: int, fn: int, tn: int, base_rate: float) -> float:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    num = sens * base_rate
    den = sens * base_rate + (1 - spec) * (1 - base_rate)
    return num / den if den > 0 else 0.0


def _classify(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "ppv_at_sample_baserate": round(prec, 3),
        "ppv_at_target_baserate": round(_bayes_ppv(tp, fp, fn, tn, TARGET_RATE), 4),
    }


def _items_of(row) -> set:
    return {f"{t}:{s}" for t, s in zip(row["event_type_seq"], row["event_subtype_seq"])}


def evaluate_trace(name: str, windows_path: Path, horizon: str = "last5") -> dict:
    wins = pd.read_parquet(windows_path)
    sub = wins[wins["horizon"] == horizon]
    if sub.empty:
        return {"trace": name, "horizon": horizon, "error": "no windows"}

    disc, inf = entity_disjoint_split(sub, discovery_frac=0.5)
    if disc.empty or inf.empty:
        return {"trace": name, "horizon": horizon, "error": "empty split"}

    # ------- inference half y_true
    y_true = inf["is_failure"].to_numpy().astype(int)

    # ------- baseline (a) MOST-RECENT-EVENT indicator
    fail_disc = disc[disc["is_failure"]]
    last_items_disc: list[str] = []
    for _, r in fail_disc.iterrows():
        types = list(r["event_type_seq"])
        subs = list(r["event_subtype_seq"])
        if types and subs:
            last_items_disc.append(f"{types[-1]}:{subs[-1]}")
    from collections import Counter
    top_last = [it for it, _ in Counter(last_items_disc).most_common(5)]

    def last_item(row) -> str:
        types = list(row["event_type_seq"])
        subs = list(row["event_subtype_seq"])
        return f"{types[-1]}:{subs[-1]}" if types and subs else ""

    inf_last = inf.apply(last_item, axis=1).to_numpy()
    y_pred_last = np.array([1 if it in top_last else 0 for it in inf_last])
    baseline_a = _classify(y_true, y_pred_last)
    baseline_a["rule"] = f"last event in top-5 disc-half fail-anchoring events: {top_last}"

    # ------- baseline (b) MINED-ITEMSET rule
    pats, _ = mine_itemsets(disc, [horizon], min_support=0.05)
    if pats.empty:
        baseline_b = {"skipped": "no patterns mined"}
    else:
        scored = score_on_inference_half(inf, pats[["itemset"]])
        pv = scored["inf_p_value"].to_numpy(dtype=float)
        scored["q_by"] = by_qvalues(pv)
        sig = scored[scored["q_by"] < 0.05]
        # Focus on non-terminal, deployable rules: exclude any pattern
        # containing "terminal_failure" (those are same-time markers).
        def has_terminal(items):
            return any("terminal_failure" in x for x in items)
        sig = sig[~sig["itemset"].apply(has_terminal)]
        sig_patterns = [set(row["itemset"]) for _, row in sig.iterrows()]
        inf_item_sets = [_items_of(r) for _, r in inf.iterrows()]
        y_pred_b = np.array([
            1 if any(p.issubset(items) for p in sig_patterns) else 0
            for items in inf_item_sets
        ])
        baseline_b = _classify(y_true, y_pred_b)
        baseline_b["n_significant_deployable_patterns"] = int(len(sig_patterns))

    # ------- baseline (c) event count
    counts_inf = inf["n_events"].to_numpy()
    fail_disc_counts = disc[disc["is_failure"]]["n_events"].to_numpy()
    thr = float(np.median(fail_disc_counts)) if len(fail_disc_counts) else 0
    y_pred_c = (counts_inf > thr).astype(int)
    baseline_c = _classify(y_true, y_pred_c)
    baseline_c["rule"] = f"n_events > median(disc fail) = {thr}"

    return {
        "trace": name, "horizon": horizon,
        "n_disc_windows": int(len(disc)),
        "n_inf_windows": int(len(inf)),
        "n_inf_fail": int(y_true.sum()),
        "n_inf_ctrl": int((y_true == 0).sum()),
        "target_operational_base_rate": TARGET_RATE,
        "baseline_a_most_recent_event": baseline_a,
        "baseline_b_mined_itemset": baseline_b,
        "baseline_c_event_count": baseline_c,
    }


def main() -> int:
    out = {}
    for name, path in [
        ("kelmarsh", ROOT / "data/processed/kelmarsh_windows.parquet"),
        ("penmanshiel", ROOT / "data/processed/penmanshiel_windows.parquet"),
    ]:
        for h in ("last5", "last10"):
            key = f"{name}_{h}"
            out[key] = evaluate_trace(name, path, h)
            print(f"[{key}] a.f1={out[key]['baseline_a_most_recent_event']['f1']} "
                  f"b.f1={out[key]['baseline_b_mined_itemset'].get('f1', 'n/a')} "
                  f"c.f1={out[key]['baseline_c_event_count']['f1']}",
                  flush=True)

    (ROOT / "results/patterns/baselines_wind.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nWrote results/patterns/baselines_wind.json ({len(out)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
