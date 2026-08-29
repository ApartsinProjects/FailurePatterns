"""CLI: normalize Kelmarsh Status_*.csv files."""

from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.kelmarsh import run


def main() -> int:
    src = Path(r"E:/tmp/kelmarsh")
    if not any(src.glob("Status_Kelmarsh_*.csv")):
        print(f"No Status_*.csv files in {src}", file=sys.stderr)
        return 2
    stats = run(src, ROOT / "data" / "processed")
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
