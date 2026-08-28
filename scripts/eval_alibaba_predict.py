"""CLI: Phase 6 predictive evaluation on Alibaba per-job windows.

Alibaba trace spans 2018-01-01 to 2018-01-09 (~9 days). Split at
2018-01-07 puts ~75% of jobs into train and 25% into a held-out test
set. Windows split by anchor timestamp, not by job identity.
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
        ROOT / "data" / "processed" / "alibaba_windows.parquet",
        ROOT / "results" / "tables",
        output_stem="alibaba_predictive",
        horizons=("last3", "last5", "last10"),
        cutoff=pd.Timestamp("2018-01-07"),
    )
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
