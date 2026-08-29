"""(1) Sequence discovery/inference split (extending W1 to PrefixSpan).

Mirrors scripts/post_selection_split.py but runs PrefixSpan (via
src.mine.sequences) on the discovery half and scores sequence
containment on the inference half.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.discovery_inference_split import (
    entity_disjoint_split, bh_qvalues, by_qvalues,
)
from src.mine.sequences import mine as mine_sequences

CONFIGS = [
    ("Azure",   ROOT / "data/processed/azure_windows.parquet",   ["24h", "last5", "last10"]),
    ("Alibaba", ROOT / "data/processed/alibaba_windows.parquet", ["last3", "last5", "last10"]),
    ("BGL",     ROOT / "data/processed/bgl_windows.parquet",     ["last5", "last10", "last20"]),
    ("SCANIA",  ROOT / "data/processed/scania_windows.parquet",  ["last5", "last10", "last20"]),
]


def _seq_in(seq: list[str], transaction: list[str]) -> bool:
    i = 0
    for it in transaction:
        if it == seq[i]:
            i += 1
            if i == len(seq):
                return True
    return False


def hypergeom_p_upper(hit_f, hit_c, n_f, n_c) -> float:
    K = hit_f + hit_c
    N = n_f + n_c
    if K == 0 or K == N or n_f == 0 or n_c == 0:
        return 1.0
    return float(hypergeom.sf(hit_f - 1, N, K, n_f))


def main() -> int:
    out = []
    for name, path, horizons in CONFIGS:
        wins = pd.read_parquet(path)
        for h in horizons:
            sub = wins[wins["horizon"] == h]
            if sub.empty or int(sub["is_failure"].sum()) < 5:
                continue
            disc, inf = entity_disjoint_split(sub, discovery_frac=0.5)
            patterns, _ = mine_sequences(disc, [h], min_support=0.05)
            if patterns.empty:
                out.append({"trace": name, "horizon": h,
                            "n_disc_windows": int(len(disc)),
                            "n_inf_windows": int(len(inf)),
                            "n_seq_mined_on_discovery": 0,
                            "n_significant_bh_005": 0,
                            "n_significant_by_005": 0})
                continue
            # score on inference half
            fail_inf = inf[inf["is_failure"]]
            ctrl_inf = inf[~inf["is_failure"]]
            n_f = len(fail_inf); n_c = len(ctrl_inf)

            def _items(row):
                return [f"{t}:{s}" for t, s in
                        zip(row["event_type_seq"], row["event_subtype_seq"])]
            fail_txs = [_items(r) for _, r in fail_inf.iterrows()]
            ctrl_txs = [_items(r) for _, r in ctrl_inf.iterrows()]

            pvals = []
            for _, r in patterns.iterrows():
                seq = list(r["sequence"])
                hit_f = sum(1 for t in fail_txs if _seq_in(seq, t))
                hit_c = sum(1 for t in ctrl_txs if _seq_in(seq, t))
                pvals.append(hypergeom_p_upper(hit_f, hit_c, n_f, n_c))
            pvals = np.array(pvals, dtype=float)
            q_bh = bh_qvalues(pvals)
            q_by = by_qvalues(pvals)
            n_bh = int((q_bh < 0.05).sum())
            n_by = int((q_by < 0.05).sum())
            out.append({
                "trace": name, "horizon": h,
                "n_disc_windows": int(len(disc)),
                "n_inf_windows": int(len(inf)),
                "n_seq_mined_on_discovery": int(len(patterns)),
                "n_significant_bh_005": n_bh,
                "n_significant_by_005": n_by,
                "fraction_bh": round(n_bh / max(1, len(patterns)), 3),
                "fraction_by": round(n_by / max(1, len(patterns)), 3),
            })
            print(f"[{name} {h}] disc={len(disc)}w inf={len(inf)}w seq_mined={len(patterns)} "
                  f"sig_BH={n_bh} sig_BY={n_by}", flush=True)

    (ROOT / "results/patterns/post_selection_sequences.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nWrote results/patterns/post_selection_sequences.json ({len(out)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
