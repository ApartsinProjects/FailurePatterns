"""CLI: Phase 6 predictive evaluation on BGL per-rack windows.

BGL trace spans 2005-06-03 to 2006-01-04 (~215 days). Split at
2005-11-01 puts ~5 months in train and ~2 months in a held-out test.
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
        ROOT / "data" / "processed" / "bgl_windows.parquet",
        ROOT / "results" / "tables",
        output_stem="bgl_predictive",
        horizons=("last5", "last10", "last20"),
        cutoff=pd.Timestamp("2005-11-01"),
    )
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
