"""CLI: build pre-failure + matched-control windows over the Azure event stream.

Usage:
    python scripts/build_windows_azure.py

Reads ``data/processed/azure_events.parquet``, writes
``data/processed/azure_windows.parquet`` and ``azure_windows_stats.json``,
prints stats, exits non-zero if any pre-declared invariant fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.windows import _AZURE_SEED_FAILURE_TS, run


def main() -> int:
    events_parquet = ROOT / "data" / "processed" / "azure_events.parquet"
    out_dir = ROOT / "data" / "processed"

    stats = run(
        events_parquet, out_dir,
        output_stem="azure_windows",
        failure_event_type="terminal_failure",
        seed_timestamps={_AZURE_SEED_FAILURE_TS},
        expected_seed_count=18,
    )
    print(json.dumps(stats.to_dict(), indent=2, default=str))

    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
