"""CLI: FP-Growth itemset mining on Alibaba windows."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mine.itemsets import DEFAULT_MIN_SUPPORT, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    args = ap.parse_args()

    stats = run(
        ROOT / "data" / "processed" / "alibaba_windows.parquet",
        ROOT / "results" / "patterns",
        output_stem="alibaba_itemsets",
        min_support=args.min_support,
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
