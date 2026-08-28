"""CLI: normalize Alibaba batch_task.csv into an event parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.alibaba import run

DEFAULT_SRC = Path(r"E:/tmp/alibaba/batch_task.csv")


def main() -> int:
    src = DEFAULT_SRC
    if not src.exists():
        print(f"Missing {src}. Fetch batch_task.csv first.", file=sys.stderr)
        return 2
    out_dir = ROOT / "data" / "processed"
    stats = run(src, out_dir)
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
