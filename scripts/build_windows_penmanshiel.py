"""Build per-turbine pre-forced-outage windows over Penmanshiel events."""

from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.windows import run


def main() -> int:
    stats = run(
        ROOT / "data" / "processed" / "penmanshiel_events.parquet",
        ROOT / "data" / "processed",
        output_stem="penmanshiel_windows",
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
