"""CLI: build pre-onset + matched-control windows over the sepsis event stream.

Reuses the shared ``src.eval.windows.run`` machinery unchanged, so the sepsis
windows are constructed by exactly the same protocol as the six operational
datasets (failure window ``[onset - horizon, onset)`` with the onset excluded;
matched controls sampled from clean regions of the same patient, guaranteed to
have no onset within a horizon in either direction).

Usage:
    python scripts/build_windows_sepsis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.windows import run


def main() -> int:
    events_parquet = ROOT / "data" / "processed" / "sepsis_events.parquet"
    out_dir = ROOT / "data" / "processed"

    stats = run(
        events_parquet, out_dir,
        output_stem="sepsis_windows",
        failure_event_type="terminal_failure",
        seed_timestamps=None,
        expected_seed_count=None,
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
