"""CLI: build per-job pre-failure + control windows over Alibaba events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.windows_alibaba import run


def main() -> int:
    events_parquet = ROOT / "data" / "processed" / "alibaba_events.parquet"
    out_dir = ROOT / "data" / "processed"
    stats = run(events_parquet, out_dir)
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
