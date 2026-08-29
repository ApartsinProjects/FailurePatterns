"""Subsequence-dominance analysis: for each top mined sequence,
does the ENTIRE sequence carry the predictive signal, or does a
proper subsequence already dominate?

For each sequence S with lift L(S), we check every proper subsequence
S' ⊂ S among the mined patterns and ask:
- "Subpart-dominant" if some S' has lift(S') >= lift(S) - EPS.
- "Full-sequence-dominant" if no proper subsequence matches its lift.

This directly tests the paper's question: is the predictor in the
full ordered sequence, or is it really in a shorter sub-sequence?
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"


def is_subsequence(sub: list, full: list) -> bool:
    """Check if `sub` occurs as an ordered (gapped) subsequence of `full`."""
    if len(sub) >= len(full):
        return False
    i = 0
    for x in full:
        if x == sub[i]:
            i += 1
            if i == len(sub):
                return True
    return False


def analyze(patterns_df: pd.DataFrame, trace_name: str, horizon: str,
            lift_col: str = "lift_failure", eps: float = 0.05) -> dict:
    df = patterns_df[patterns_df["horizon"] == horizon].copy()
    if df.empty:
        return {"trace": trace_name, "horizon": horizon, "n": 0}
    df["seq_list"] = df["sequence"].apply(list)
    # only consider sequences of length >= 2 (length-1 has no proper subseq to check)
    multi = df[df["seq_list"].apply(len) >= 2].sort_values(lift_col, ascending=False)
    all_by_key = {tuple(r["seq_list"]): float(r[lift_col]) for _, r in df.iterrows()}

    subpart_dominant = 0
    full_dominant = 0
    examples = []
    for _, r in multi.head(200).iterrows():
        S = r["seq_list"]
        L = float(r[lift_col])
        best_sub_lift = 0.0
        best_sub = None
        # scan all mined patterns of shorter length
        for key, lft in all_by_key.items():
            if len(key) >= len(S):
                continue
            if is_subsequence(list(key), S):
                if lft > best_sub_lift:
                    best_sub_lift = lft
                    best_sub = key
        if best_sub_lift >= L - eps:
            subpart_dominant += 1
            if len(examples) < 3:
                examples.append({
                    "sequence": S,
                    "lift": round(L, 3),
                    "dominant_subseq": list(best_sub) if best_sub else None,
                    "subseq_lift": round(best_sub_lift, 3),
                    "verdict": "subpart dominates",
                })
        else:
            full_dominant += 1
            if len(examples) < 6:
                examples.append({
                    "sequence": S,
                    "lift": round(L, 3),
                    "best_proper_subseq": list(best_sub) if best_sub else None,
                    "best_subseq_lift": round(best_sub_lift, 3),
                    "delta": round(L - best_sub_lift, 3),
                    "verdict": "full sequence dominates",
                })

    return {
        "trace": trace_name,
        "horizon": horizon,
        "n_multi_item_sequences_analyzed": int(len(multi.head(200))),
        "subpart_dominant": subpart_dominant,
        "full_sequence_dominant": full_dominant,
        "fraction_full_dominant": round(full_dominant / max(1, len(multi.head(200))), 3),
        "examples": examples,
    }


def main() -> int:
    results = []
    for name, path, horizons in [
        ("Azure",   ROOT / "results/patterns/azure_sequences.parquet",   ["last5", "last10"]),
        ("Alibaba", ROOT / "results/patterns/alibaba_sequences.parquet", ["last3", "last5", "last10"]),
    ]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for h in horizons:
            r = analyze(df, name, h)
            results.append(r)
            print(f"[{name} {h}] full-dominant={r['full_sequence_dominant']}, "
                  f"subpart-dominant={r['subpart_dominant']}, "
                  f"fraction_full={r['fraction_full_dominant']}")
    (OUT / "subsequence_dominance.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print()
    print("Concrete examples:")
    for r in results:
        for ex in r.get("examples", [])[:2]:
            print(f"  [{r['trace']} {r['horizon']}] {ex['verdict']}: {ex['sequence']} lift={ex['lift']}")
            if 'dominant_subseq' in ex:
                print(f"      dominated by {ex['dominant_subseq']} lift={ex['subseq_lift']}")
            elif 'best_proper_subseq' in ex:
                print(f"      best proper subseq {ex['best_proper_subseq']} lift={ex['best_subseq_lift']} (delta +{ex['delta']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
