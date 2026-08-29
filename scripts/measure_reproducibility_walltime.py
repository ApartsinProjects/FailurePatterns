"""(5) Measure wall-clock for the reproducible pipeline so the paper's
reproducibility claim is grounded in a directly measured artifact.

Times each downstream stage from the SAVED window parquets forward.
Ingest, download, and window construction are one-time steps documented
in the ingest_*.json artifacts and excluded here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


STAGES = [
    ("itemsets_azure",      [PY, "scripts/mine_azure_itemsets.py"]),
    ("itemsets_alibaba",    [PY, "scripts/mine_alibaba_itemsets.py"]),
    ("itemsets_bgl",        [PY, "scripts/mine_bgl_itemsets.py"]),
    ("itemsets_scania",     [PY, "scripts/mine_scania_itemsets.py"]),
    ("post_selection_split", [PY, "scripts/post_selection_split.py"]),
    ("post_selection_sequences", [PY, "scripts/post_selection_sequences.py"]),
    ("bh_by_matched",       [PY, "scripts/bh_by_matched_hazards.py"]),
    ("closed_sequential",   [PY, "scripts/closed_sequential_clospan.py"]),
]


def wall_time(cmd: list[str]) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    dt = time.perf_counter() - t0
    return dt, r.returncode, (r.stderr[-200:] if r.returncode else "")


def main() -> int:
    rows = []
    total = 0.0
    for label, cmd in STAGES:
        script = Path(cmd[1])
        if not (ROOT / script).exists():
            rows.append({"stage": label, "wall_seconds": None, "status": "skipped-missing"})
            continue
        print(f"[timing] {label} ...", flush=True)
        dt, rc, err = wall_time(cmd)
        rows.append({
            "stage": label,
            "wall_seconds": round(dt, 2),
            "status": "ok" if rc == 0 else f"fail rc={rc}",
            "stderr_tail": err if rc else "",
        })
        if rc == 0:
            total += dt
        print(f"  -> {dt:.1f}s  {'ok' if rc==0 else 'FAIL'}", flush=True)

    out = {
        "measured_stages": rows,
        "total_measured_seconds": round(total, 2),
        "total_measured_minutes": round(total / 60.0, 2),
        "python": sys.version.split()[0],
        "host": "single Windows CPU-only workstation",
        "note": "Wall-clock excludes raw-trace download and initial window "
                "construction; those are one-time steps documented in the "
                "ingest_*.json artifacts.",
    }
    (ROOT / "results/patterns/reproducibility_timing.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nTotal measured: {total:.1f}s ({total/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
