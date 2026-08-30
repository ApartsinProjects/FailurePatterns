"""W1 fix: cluster-preserving (within-entity) inference for the wind farms.

The window-level hypergeometric test treats every window as an
independent observation. On the wind farms only 6 (Kelmarsh) or 9
(Penmanshiel) turbines produce hundreds of windows, so the windows are
clustered and the hypergeometric q-values (down to 1e-98) are
anti-conservative as population-level evidence.

This script replaces them with two cluster-aware analyses for each
headline signature, on the same inference half used in the catalog:

  1. WITHIN-ENTITY LABEL PERMUTATION. Permute the failure/control label
     within each turbine (preserving each turbine's case and control
     counts and its clustering) B times; the empirical p-value is the
     fraction of permutations whose case-vs-control lift is at least the
     observed lift. This is a valid test of pattern-outcome association
     that respects the entity clustering.
  2. LEAVE-ONE-TURBINE-OUT CONSISTENCY. Hold out each turbine in turn and
     recompute the signature's lift on the remaining turbines; report in
     how many folds the signature stays enriched (lift > 1 with at least
     one case hit and zero-or-few control hits). Cross-turbine
     consistency, not a tiny p-value, is the load-bearing evidence when
     entities are few.

Outputs results/patterns/<trace>_cluster_inference.json.
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
    entity_disjoint_split, score_on_inference_half, by_qvalues,
)
from src.mine.itemsets import mine as mine_itemsets

SEED = 20260828
B = 2000  # permutations


def _items(row) -> set:
    return {f"{t}:{s}" for t, s in
            zip(row["event_type_seq"], row["event_subtype_seq"])}


def _lift(present: np.ndarray, is_fail: np.ndarray) -> float:
    n_f = is_fail.sum(); n_c = (~is_fail).sum()
    if n_f == 0 or n_c == 0:
        return float("nan")
    sf = (present & is_fail).sum() / n_f
    pooled = present.sum() / (n_f + n_c)
    return sf / pooled if pooled > 0 else float("nan")


def within_entity_perm_p(present: np.ndarray, is_fail: np.ndarray,
                         ent: np.ndarray, rng, B: int) -> float:
    obs = _lift(present, is_fail)
    if not np.isfinite(obs):
        return float("nan")
    ge = 1  # +1 smoothing
    labels = is_fail.copy()
    for _ in range(B):
        perm = labels.copy()
        for e in np.unique(ent):
            idx = np.where(ent == e)[0]
            perm[idx] = rng.permutation(labels[idx])
        if _lift(present, perm) >= obs - 1e-12:
            ge += 1
    return ge / (B + 1)


def analyse(trace: str, windows_path: Path, horizon: str, top_k: int = 12) -> dict:
    wins = pd.read_parquet(windows_path)
    sub = wins[wins["horizon"] == horizon]
    disc, inf = entity_disjoint_split(sub, discovery_frac=0.5)
    pats, _ = mine_itemsets(disc, [horizon], min_support=0.05)
    scored = score_on_inference_half(inf, pats[["itemset"]])
    # exclude terminal_failure-only markers so we test genuine precursors (W9)
    def has_term(items): return any("terminal_failure" in x for x in items)
    scored = scored[~scored["itemset"].apply(has_term)]
    scored = scored.sort_values("inf_lift", ascending=False).head(top_k)

    inf_items = [_items(r) for _, r in inf.iterrows()]
    is_fail = inf["is_failure"].to_numpy().astype(bool)
    ent = inf["entity_id"].to_numpy()
    turbines = np.unique(ent)
    rng = np.random.default_rng(SEED)

    rows = []
    for _, r in scored.iterrows():
        pat = set(r["itemset"])
        present = np.array([pat.issubset(s) for s in inf_items])
        p_perm = within_entity_perm_p(present, is_fail, ent, rng, B)
        # leave-one-turbine-out
        loto_ok = 0; loto_n = 0
        for t in turbines:
            keep = ent != t
            if is_fail[keep].sum() == 0 or (~is_fail[keep]).sum() == 0:
                continue
            loto_n += 1
            lift_t = _lift(present[keep], is_fail[keep])
            case_hits_t = int((present[keep] & is_fail[keep]).sum())
            if np.isfinite(lift_t) and lift_t > 1.0 and case_hits_t >= 1:
                loto_ok += 1
        rows.append({
            "itemset": list(r["itemset"]),
            "inf_lift": round(float(r["inf_lift"]), 3),
            "hypergeom_q_by": float(r.get("q_by", float("nan")))
                if "q_by" in scored.columns else None,
            "within_entity_perm_p": round(p_perm, 4),
            "loto_folds_enriched": f"{loto_ok}/{loto_n}",
        })
    # BY on the within-entity permutation p-values
    pv = np.array([x["within_entity_perm_p"] for x in rows], dtype=float)
    qv = by_qvalues(pv)
    for x, q in zip(rows, qv):
        x["within_entity_perm_q_by"] = round(float(q), 4)

    n_sig = int((qv < 0.05).sum())
    return {
        "trace": trace, "horizon": horizon,
        "n_turbines": int(len(turbines)),
        "n_inf_fail": int(is_fail.sum()), "n_inf_ctrl": int((~is_fail).sum()),
        "n_precursor_signatures_tested": len(rows),
        "n_within_entity_perm_by_sig_005": n_sig,
        "permutations": B,
        "signatures": rows,
    }


def main() -> int:
    out = {}
    for trace, path, h in [
        ("kelmarsh", ROOT / "data/processed/kelmarsh_windows.parquet", "last5"),
        ("penmanshiel", ROOT / "data/processed/penmanshiel_windows.parquet", "last5"),
    ]:
        res = analyse(trace, path, h)
        out[f"{trace}_{h}"] = res
        (ROOT / f"results/patterns/{trace}_cluster_inference.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        print(f"[{trace} {h}] turbines={res['n_turbines']} "
              f"precursor_sigs={res['n_precursor_signatures_tested']} "
              f"within-entity-perm BY-sig={res['n_within_entity_perm_by_sig_005']}")
        for s in res["signatures"][:4]:
            print(f"    {s['itemset']} lift={s['inf_lift']} "
                  f"perm_p={s['within_entity_perm_p']} perm_q={s['within_entity_perm_q_by']} "
                  f"LOTO={s['loto_folds_enriched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
