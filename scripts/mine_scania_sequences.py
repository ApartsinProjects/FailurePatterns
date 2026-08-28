"""CLI: PrefixSpan sequence mining on SCANIA windows."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mine.sequences import DEFAULT_MIN_SUPPORT, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    args = ap.parse_args()

    stats = run(
        ROOT / "data" / "processed" / "scania_windows.parquet",
        ROOT / "results" / "patterns",
        output_stem="scania_sequences",
        min_support=args.min_support,
    )
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
