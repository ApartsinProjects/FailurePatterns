"""Run the count-preserving order comparator (W4) on Azure and Alibaba."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.count_preserving_order import score_top_sequences


def main() -> int:
    out = {}
    for name, wpath, spath, horizons in [
        ("Azure",
         ROOT / "data/processed/azure_windows.parquet",
         ROOT / "results/patterns/azure_sequences.parquet",
         ["last5", "last10"]),
        ("Alibaba",
         ROOT / "data/processed/alibaba_windows.parquet",
         ROOT / "results/patterns/alibaba_sequences.parquet",
         ["last3", "last5", "last10"]),
    ]:
        wins = pd.read_parquet(wpath)
        seqs = pd.read_parquet(spath)
        for h in horizons:
            sub_w = wins[wins["horizon"] == h]
            sub_s = seqs[seqs["horizon"] == h]
            if sub_w.empty or sub_s.empty:
                continue
            scored = score_top_sequences(sub_w, sub_s, top_k=20, n_shuffles=20)
            if scored.empty:
                continue
            out[f"{name}_{h}"] = {
                "n_analyzed": int(len(scored)),
                "mean_real_lift": float(scored["real_lift"].mean()),
                "mean_shuffle_lift": float(scored["count_preserving_shuffle_lift"].mean()),
                "mean_order_effect": float(scored["order_effect"].mean()),
                "median_order_effect": float(scored["order_effect"].median()),
                "mean_naive_order_gain": float(scored["naive_order_gain"].mean()) if "naive_order_gain" in scored.columns else float("nan"),
                "top5": scored.head(5).to_dict(orient="records"),
            }
            print(f"[{name} {h}] n={len(scored)} real={out[f'{name}_{h}']['mean_real_lift']:.3f} "
                  f"shuf={out[f'{name}_{h}']['mean_shuffle_lift']:.3f} "
                  f"order_effect={out[f'{name}_{h}']['mean_order_effect']:+.3f}",
                  flush=True)

    (ROOT / "results/patterns/count_preserving_order.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
