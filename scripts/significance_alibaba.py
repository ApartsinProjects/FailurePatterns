"""CLI: Phase 5 significance on Alibaba patterns."""

from __future__ import annotations

import json, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.significance import run


def main() -> int:
    summary = run(
        dataset="alibaba",
        windows_parquet=ROOT / "data" / "processed" / "alibaba_windows.parquet",
        itemsets_parquet=ROOT / "results" / "patterns" / "alibaba_itemsets.parquet",
        sequences_parquet=ROOT / "results" / "patterns" / "alibaba_sequences.parquet",
        out_dir=ROOT / "results" / "patterns",
    )
    print(json.dumps(asdict(summary), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
