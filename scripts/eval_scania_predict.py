"""CLI: Phase 6 predictive evaluation on SCANIA per-vehicle windows.

SCANIA trace spans 2019-01 to 2020-05. Split at 2020-01-01 puts ~1 year
in train and ~5 months in held-out test.
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.predict import run


def main() -> int:
    stats = run(
        ROOT / "data" / "processed" / "scania_windows.parquet",
        ROOT / "results" / "tables",
        output_stem="scania_predictive",
        horizons=("last5", "last10", "last20"),
        cutoff=pd.Timestamp("2020-01-01"),
    )
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
