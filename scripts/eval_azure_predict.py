"""CLI: Phase 6 head-to-head predictive evaluation on Azure PdM.

Usage:
    python scripts/eval_azure_predict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.predict import run


def main() -> int:
    windows_parquet = ROOT / "data" / "processed" / "azure_windows.parquet"
    out_dir = ROOT / "results" / "tables"
    stats = run(windows_parquet, out_dir)
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
