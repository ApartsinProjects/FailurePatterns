"""CLI: normalize SCANIA Component X readouts to an event parquet."""

from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.scania import run

DEFAULT_READOUTS = Path(r"E:/tmp/scania/train_operational_readouts.csv")
DEFAULT_TTE = Path(r"E:/tmp/scania/train_tte.csv")


def main() -> int:
    if not DEFAULT_READOUTS.exists() or not DEFAULT_TTE.exists():
        print(f"Missing SCANIA files.", file=sys.stderr)
        return 2
    stats = run(DEFAULT_READOUTS, DEFAULT_TTE, ROOT / "data" / "processed")
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
