"""CLI: normalize BGL.log into an event parquet."""

from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.bgl import run

DEFAULT_SRC = Path(r"E:/tmp/bgl/BGL.log")


def main() -> int:
    if not DEFAULT_SRC.exists():
        print(f"Missing {DEFAULT_SRC}. Fetch BGL.log first.", file=sys.stderr)
        return 2
    stats = run(DEFAULT_SRC, ROOT / "data" / "processed")
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
