"""CLI: PrefixSpan sequence mining on Azure windows (Phase 4).

Usage:
    python scripts/mine_azure_sequences.py [--min-support 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mine.sequences import DEFAULT_MIN_SUPPORT, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    args = ap.parse_args()

    windows_parquet = ROOT / "data" / "processed" / "azure_windows.parquet"
    out_dir = ROOT / "results" / "patterns"

    stats = run(windows_parquet, out_dir, min_support=args.min_support)
    print(json.dumps(stats.to_dict(), indent=2, default=str))

    failed = [k for k, v in stats.invariants.items() if not v]
    if failed:
        print(f"\nINVARIANT FAILURES: {failed}", file=sys.stderr)
        return 2
    print("\nAll pre-declared invariants passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
