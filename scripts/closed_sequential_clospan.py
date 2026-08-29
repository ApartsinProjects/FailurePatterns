"""(4) Closed sequential patterns via SPMF CloSpan.

A closed sequential pattern is one whose support is strictly larger than
any of its super-sequences. This is the sequence-mining analogue of the
LCM closed-itemset pass we already run for itemsets, and it reduces the
redundancy in the raw PrefixSpan output the paper reports.

For each trace we:
  - reuse src.mine.sequences helpers to build the SPMF encoding
  - run CloSpan (SPMF) at the same min_support as PrefixSpan (5%)
  - count closed patterns on the failure windows for each horizon
  - report the compression ratio closed / raw against the existing
    PrefixSpan output already saved to results/patterns
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mine.sequences import (
    SPMF_JAR, _build_vocab, _parse_spmf_output, _sequences, _to_spmf_format,
)

CONFIGS = [
    ("azure",   "azure_windows.parquet",   "azure_sequences.parquet",   ["24h", "last5", "last10"]),
    ("alibaba", "alibaba_windows.parquet", "alibaba_sequences.parquet", ["last3", "last5", "last10"]),
    ("bgl",     "bgl_windows.parquet",     "bgl_sequences.parquet",     ["last5", "last10", "last20"]),
    ("scania",  "scania_windows.parquet",  "scania_sequences.parquet",  ["last5", "last10", "last20"]),
]

MIN_SUPPORT = 0.05
TMP = ROOT / "_spmf_tmp_clospan"
TMP.mkdir(parents=True, exist_ok=True)


def run_clospan(in_path: Path, out_path: Path, min_support: float) -> None:
    pct = f"{min_support * 100:.4f}%"
    r = subprocess.run(
        ["java", "-Xmx6g", "-jar", str(SPMF_JAR), "run", "CloSpan",
         str(in_path), str(out_path), pct],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"CloSpan failed for {in_path.name}:\n{r.stderr}\n{r.stdout}")


def main() -> int:
    summary = []
    for name, wins_name, seq_name, horizons in CONFIGS:
        wins_path = ROOT / "data" / "processed" / wins_name
        wins = pd.read_parquet(wins_path)
        raw_seq_path = (ROOT / "results" / "patterns" / seq_name) if seq_name else None
        raw_seq = pd.read_parquet(raw_seq_path) if raw_seq_path and raw_seq_path.exists() else pd.DataFrame()

        for h in horizons:
            w = wins[wins["horizon"] == h]
            w_fail = w[w["is_failure"]]
            if len(w_fail) == 0:
                continue
            seq_fail = _sequences(w_fail)
            # Build vocab only over failure sequences (mining closed on fail only)
            vocab = _build_vocab(seq_fail)
            inv = {v: k for k, v in vocab.items()}
            in_path = TMP / f"{name}_{h}_fail.txt"
            out_path = TMP / f"{name}_{h}_fail_closed.out"
            in_path.write_text(_to_spmf_format(seq_fail, vocab), encoding="utf-8")
            try:
                run_clospan(in_path, out_path, MIN_SUPPORT)
                closed = _parse_spmf_output(out_path, inv)
            except Exception as e:
                closed = []
                print(f"[{name} {h}] CloSpan error: {e}")
                continue
            n_closed = len(closed)

            # Raw PrefixSpan count from existing parquet, restricted to this horizon.
            if not raw_seq.empty and "horizon" in raw_seq.columns:
                n_raw = int((raw_seq["horizon"] == h).sum())
            else:
                n_raw = None

            summary.append({
                "trace": name, "horizon": h,
                "n_fail_windows": int(len(w_fail)),
                "n_closed_sequences_clospan": n_closed,
                "n_prefixspan_patterns": n_raw,
                "compression_ratio": (
                    round(n_closed / n_raw, 3) if (n_raw and n_raw > 0) else None
                ),
            })
            print(f"[{name} {h}] closed={n_closed}  raw_prefixspan={n_raw}", flush=True)

    (ROOT / "results/patterns/closed_sequential_clospan.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote results/patterns/closed_sequential_clospan.json ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
