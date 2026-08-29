"""CLI: risk-set matched sampling + hazard-ratio scoring on SCANIA."""

from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.windows_scania_riskset import run

DEFAULT_TTE = Path(r"E:/tmp/scania/train_tte.csv")


def main() -> int:
    stats, summary = run(
        ROOT / "data" / "processed" / "scania_events.parquet",
        DEFAULT_TTE,
        ROOT / "results" / "patterns",
    )
    print("=== RISK-SET WINDOWS ===")
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    print()
    print("=== HAZARD-RATIO PATTERNS ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
