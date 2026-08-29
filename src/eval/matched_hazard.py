"""Matched conditional logistic regression for SCANIA risk-set patterns
(W2 from DAMI review).

Under incidence-density (risk-set) sampling, controls are drawn from
each case's risk set at the case's event time. The correct estimator
is conditional (matched) logistic regression stratified by matched
risk set — equivalently the sampled Cox partial likelihood restricted
to the pattern indicator — which recovers the hazard ratio.

The pooled 2x2 Woolf-Haldane OR previously called "MH-OR" discards
the matched structure and is not the Prentice-Breslow analysis.

This module fits a proper conditional logistic per pattern using
statsmodels ConditionalLogit stratified by ``match_id``, then reports
the pattern coefficient, its 95% CI, p-value, and hazard-ratio
interpretation.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd


def _fit_one_pattern(args) -> dict:
    """Worker: fit a matched conditional logistic for one pattern."""
    import statsmodels.api as sm  # local import for pickling in workers
    itemset, y, x, match_id = args
    df = pd.DataFrame({"y": y, "x": x, "s": match_id})
    # Drop strata with zero variation on x (uninformative)
    variation = df.groupby("s")["x"].std().rename("_v")
    df = df.join(variation, on="s")
    df = df[df["_v"] > 0]
    if df.empty or df["x"].sum() == 0 or (df["x"] == 0).all():
        return {"itemset": list(itemset), "n_informative_strata": 0,
                "hazard_ratio": float("nan"), "hr_ci_low": float("nan"),
                "hr_ci_high": float("nan"), "p_value": float("nan"),
                "significant_005": False, "n_case_hits": int((y * x).sum()),
                "n_control_hits": int((1 - y) * x).sum() if hasattr((1 - y) * x, "sum") else 0}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.ConditionalLogit(df["y"], df[["x"]], groups=df["s"])
            res = model.fit(method="bfgs", disp=False, maxiter=200)
        coef = float(res.params["x"])
        se = float(res.bse["x"])
        p = float(res.pvalues["x"])
        hr = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
        return {"itemset": list(itemset),
                "n_informative_strata": int(df["s"].nunique()),
                "hazard_ratio": hr, "hr_ci_low": ci_low, "hr_ci_high": ci_high,
                "p_value": p,
                "significant_005": bool(ci_low > 1.0 and p < 0.05),
                "coef": coef, "coef_se": se,
                "n_case_hits": int(((df["y"] == 1) & (df["x"] == 1)).sum()),
                "n_control_hits": int(((df["y"] == 0) & (df["x"] == 1)).sum())}
    except Exception as e:
        return {"itemset": list(itemset), "n_informative_strata": 0,
                "hazard_ratio": float("nan"), "hr_ci_low": float("nan"),
                "hr_ci_high": float("nan"), "p_value": float("nan"),
                "significant_005": False, "error": str(e)[:120],
                "n_case_hits": 0, "n_control_hits": 0}


def score_patterns_matched(
    riskset_windows: pd.DataFrame,
    patterns_df: pd.DataFrame,
    itemset_col: str = "itemset",
    n_workers: int | None = 1,
) -> pd.DataFrame:
    """Fit conditional logistic per pattern stratified by ``match_id``.
    Returns a DataFrame with per-pattern hazard ratio, CI, and p-value.
    """
    if patterns_df.empty:
        return patterns_df.copy()

    # Precompute per-window item sets
    def _items(row) -> set:
        return {f"{t}:{s}" for t, s in
                zip(row["event_type_seq"], row["event_subtype_seq"])}
    all_items = [_items(r) for _, r in riskset_windows.iterrows()]
    y = riskset_windows["is_failure"].to_numpy().astype(int)
    m = riskset_windows["match_id"].to_numpy()

    tasks = []
    for _, r in patterns_df.iterrows():
        pat = set(r[itemset_col])
        x = np.array([1 if pat.issubset(s) else 0 for s in all_items], dtype=int)
        tasks.append((pat, y, x, m))

    rows: list[dict] = []
    if n_workers and n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for out in ex.map(_fit_one_pattern, tasks):
                rows.append(out)
    else:
        for t in tasks:
            rows.append(_fit_one_pattern(t))
    return pd.DataFrame(rows)
