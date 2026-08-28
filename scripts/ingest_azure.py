"""CLI: normalize Azure PdM CSVs into a single event stream.

Usage:
    python scripts/ingest_azure.py

Reads ``data/raw/azure/*.csv``, writes ``data/processed/azure_events.parquet``
+ ``azure_telemetry.parquet`` + ``azure_load_stats.json``, and prints a
one-line status. Exits non-zero if any pre-declared invariant fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the project root importable regardless of where this is invoked from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.azure import run


def main() -> int:
    raw_dir = ROOT / "data" / "raw" / "azure"
    out_dir = ROOT / "data" / "processed"

    stats = run(raw_dir, out_dir)
    print(json.dumps(stats.to_dict(), indent=2, default=str))

    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
