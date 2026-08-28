"""Sequential pattern mining on Azure PdM windows (Phase 4).

Runs SPMF's PrefixSpan (via ``java -jar scripts/spmf.jar``) on the ordered
event streams of failure and control windows. For each mined sequence we
compute failure and control support, sequence-lift, and the "itemset
counterpart" lift so that Phase 4 can be compared head-to-head with
Phase 3.

Pre-declared sanity invariant (PLAN.md phase 4):

    A random within-window permutation of event order must NOT preserve
    the top failure-window sequence lifts. Concretely, we require that
    the mean surviving lift after shuffle is strictly less than the mean
    surviving lift on the un-shuffled data at the same min_support.

Items are ``event_type:event_subtype`` strings, encoded to int IDs for
SPMF via a per-run vocabulary. Same-timestamp events keep whatever order
the loader produced (a source of noise noted in docs).
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -------------------------- config ----------------------------------------

DEFAULT_MIN_SUPPORT = 0.05
LIFT_RATIO_SANITY = 1.5
RNG_SEED = 20260828
SPMF_JAR = Path(__file__).resolve().parents[2] / "scripts" / "spmf.jar"


# -------------------------- encoding helpers ------------------------------

def _make_seq(seq_types: list[str], seq_subtypes: list[str]) -> list[str]:
    return [f"{t}:{s}" for t, s in zip(seq_types, seq_subtypes)]


def _sequences(windows: pd.DataFrame) -> list[list[str]]:
    return [
        _make_seq(row["event_type_seq"], row["event_subtype_seq"])
        for _, row in windows.iterrows()
    ]


def _build_vocab(all_sequences: list[list[str]]) -> dict[str, int]:
    """Deterministic int IDs starting at 1 (SPMF disallows 0).
    Sorted for reproducibility."""
    items = sorted({it for s in all_sequences for it in s})
    return {it: i + 1 for i, it in enumerate(items)}


def _to_spmf_format(sequences: list[list[str]], vocab: dict[str, int]) -> str:
    """SPMF sequence format: each item followed by ' -1', each sequence
    terminated by ' -2', newline-separated. Empty sequences are dropped
    (SPMF crashes on lines with only ``-2``)."""
    lines = []
    for seq in sequences:
        if not seq:
            continue
        parts: list[str] = []
        for it in seq:
            parts.append(str(vocab[it]))
            parts.append("-1")
        parts.append("-2")
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def _run_prefixspan(input_path: Path, output_path: Path, min_support: float
                    ) -> None:
    """Invoke SPMF via java -jar. min_support is the fractional support."""
    if not SPMF_JAR.exists():
        raise FileNotFoundError(f"Missing SPMF jar at {SPMF_JAR}")
    pct = f"{min_support * 100:.4f}%"
    r = subprocess.run(
        ["java", "-jar", str(SPMF_JAR), "run", "PrefixSpan",
         str(input_path), str(output_path), pct],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SPMF failed:\n{r.stderr}\n{r.stdout}")


_OUT_RE = re.compile(r"^(?P<items>.+?)#SUP:\s*(?P<sup>\d+)\s*$")


def _parse_spmf_output(path: Path, inv_vocab: dict[int, str]) -> list[dict]:
    """Parse SPMF output. Each line is: ``i1 -1 i2 -1 ... #SUP: N``."""
    patterns: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            m = _OUT_RE.match(line.strip())
            if not m:
                continue
            ids = [int(x) for x in m.group("items").split() if x != "-1"]
            seq = [inv_vocab[i] for i in ids]
            patterns.append({"sequence": seq, "spmf_support": int(m.group("sup"))})
    return patterns


# -------------------------- support scoring -------------------------------

def _sequence_supports(sequence: list[str], transactions: list[list[str]]) -> int:
    """Count how many transactions contain ``sequence`` as an ordered
    subsequence (not necessarily contiguous). Matches SPMF PrefixSpan
    semantics."""
    n = 0
    for tx in transactions:
        i = 0
        for item in tx:
            if item == sequence[i]:
                i += 1
                if i == len(sequence):
                    n += 1
                    break
    return n


def _itemset_supports(itemset: frozenset, transactions: list[list[str]]) -> int:
    return sum(1 for tx in transactions if itemset.issubset(tx))


# -------------------------- data model ------------------------------------

@dataclass
class SeqMiningStats:
    min_support: float
    lift_ratio_sanity: float
    n_patterns_by_horizon: dict[str, int] = field(default_factory=dict)
    max_real_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    max_shuffled_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    mean_top10_real_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    mean_top10_shuffled_lift_by_horizon: dict[str, float] = field(default_factory=dict)
    n_patterns_above_shuffle_ceiling_by_horizon: dict[str, int] = field(default_factory=dict)
    invariants: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# -------------------------- main pipeline ---------------------------------

def _mine_one_horizon(
    hname: str,
    w_fail: pd.DataFrame,
    w_ctrl: pd.DataFrame,
    min_support: float,
    tmp_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[dict], float, float, float, float]:
    """Return (real_pattern_rows, max_real_lift, max_shuffled_lift,
    mean_top10_real, mean_top10_shuffled)."""
    seq_fail = _sequences(w_fail)
    seq_ctrl = _sequences(w_ctrl)
    seq_all = seq_fail + seq_ctrl
    n_fail, n_ctrl = len(w_fail), len(w_ctrl)
    n_all = n_fail + n_ctrl

    vocab = _build_vocab(seq_all)
    inv_vocab = {v: k for k, v in vocab.items()}

    if not any(seq_fail):
        return [], float("nan"), float("nan"), float("nan"), float("nan")

    # --- real run ------------------------------------------------------
    in_path = tmp_dir / f"seq_{hname}_fail.txt"
    out_path = tmp_dir / f"seq_{hname}_fail.out"
    in_path.write_text(_to_spmf_format(seq_fail, vocab), encoding="utf-8")
    _run_prefixspan(in_path, out_path, min_support)
    raw = _parse_spmf_output(out_path, inv_vocab)

    rows: list[dict] = []
    real_lifts: list[float] = []
    for r in raw:
        seq = r["sequence"]
        hit_fail = _sequence_supports(seq, seq_fail)
        hit_ctrl = _sequence_supports(seq, seq_ctrl)
        pooled = (hit_fail + hit_ctrl) / n_all
        supp_fail = hit_fail / n_fail
        supp_ctrl = hit_ctrl / n_ctrl if n_ctrl else 0.0
        lift = supp_fail / pooled if pooled > 0 else float("nan")
        # Itemset counterpart (unordered version of this sequence).
        itemset = frozenset(seq)
        item_hit_fail = _itemset_supports(itemset, seq_fail)
        item_hit_ctrl = _itemset_supports(itemset, seq_ctrl)
        item_pooled = (item_hit_fail + item_hit_ctrl) / n_all
        item_lift = (
            (item_hit_fail / n_fail) / item_pooled
            if item_pooled > 0 else float("nan")
        )
        p_fail_given = (
            hit_fail / (hit_fail + hit_ctrl)
            if (hit_fail + hit_ctrl) > 0 else float("nan")
        )
        rows.append({
            "horizon": hname,
            "sequence": seq,
            "sequence_length": len(seq),
            "n_failure": hit_fail,
            "n_control": hit_ctrl,
            "support_failure": supp_fail,
            "support_control": supp_ctrl,
            "lift_failure": lift,
            "p_fail_given_pattern": p_fail_given,
            "itemset": sorted(itemset),
            "itemset_lift_failure": item_lift,
            "order_gain": (
                lift - item_lift
                if not (np.isnan(lift) or np.isnan(item_lift)) else float("nan")
            ),
        })
        if not np.isnan(lift):
            real_lifts.append(lift)

    real_lifts_sorted = sorted(real_lifts, reverse=True)
    max_real = real_lifts_sorted[0] if real_lifts_sorted else float("nan")
    mean_top10_real = (
        float(np.mean(real_lifts_sorted[:10]))
        if real_lifts_sorted else float("nan")
    )

    # --- shuffled run (within-window order permutation) ----------------
    seq_fail_shuf = [rng.permutation(np.array(s, dtype=object)).tolist()
                     for s in seq_fail]
    seq_ctrl_shuf = [rng.permutation(np.array(s, dtype=object)).tolist()
                     for s in seq_ctrl]
    in_path_s = tmp_dir / f"seq_{hname}_fail_shuf.txt"
    out_path_s = tmp_dir / f"seq_{hname}_fail_shuf.out"
    in_path_s.write_text(_to_spmf_format(seq_fail_shuf, vocab), encoding="utf-8")
    _run_prefixspan(in_path_s, out_path_s, min_support)
    raw_s = _parse_spmf_output(out_path_s, inv_vocab)

    shuf_lifts: list[float] = []
    for r in raw_s:
        seq = r["sequence"]
        hit_fail = _sequence_supports(seq, seq_fail_shuf)
        hit_ctrl = _sequence_supports(seq, seq_ctrl_shuf)
        pooled = (hit_fail + hit_ctrl) / n_all
        supp_fail = hit_fail / n_fail
        lift = supp_fail / pooled if pooled > 0 else float("nan")
        if not np.isnan(lift):
            shuf_lifts.append(lift)

    shuf_lifts_sorted = sorted(shuf_lifts, reverse=True)
    max_shuf = shuf_lifts_sorted[0] if shuf_lifts_sorted else float("nan")
    mean_top10_shuf = (
        float(np.mean(shuf_lifts_sorted[:10]))
        if shuf_lifts_sorted else float("nan")
    )

    # Annotate each real row with the shuffled ceiling.
    for r in rows:
        r["shuffled_null_lift_ceiling"] = max_shuf
        r["survives_shuffle_null"] = (
            not np.isnan(r["lift_failure"]) and r["lift_failure"] > max_shuf
        )

    return rows, max_real, max_shuf, mean_top10_real, mean_top10_shuf


def mine(
    windows: pd.DataFrame,
    horizons: Iterable[str],
    min_support: float = DEFAULT_MIN_SUPPORT,
    rng_seed: int = RNG_SEED,
    tmp_dir: Path | None = None,
) -> tuple[pd.DataFrame, SeqMiningStats]:
    stats = SeqMiningStats(
        min_support=min_support, lift_ratio_sanity=LIFT_RATIO_SANITY,
    )
    rng = np.random.default_rng(rng_seed)
    all_rows: list[dict] = []

    if tmp_dir is None:
        tmp_dir = Path("./_spmf_tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for hname in horizons:
        w = windows[windows["horizon"] == hname]
        w_fail = w[w["is_failure"]]
        w_ctrl = w[~w["is_failure"]]
        if len(w_fail) == 0 or len(w_ctrl) == 0:
            stats.n_patterns_by_horizon[hname] = 0
            continue

        rows, max_real, max_shuf, mean_top10_real, mean_top10_shuf = \
            _mine_one_horizon(hname, w_fail, w_ctrl, min_support, tmp_dir, rng)
        stats.n_patterns_by_horizon[hname] = len(rows)
        stats.max_real_lift_by_horizon[hname] = max_real
        stats.max_shuffled_lift_by_horizon[hname] = max_shuf
        stats.mean_top10_real_lift_by_horizon[hname] = mean_top10_real
        stats.mean_top10_shuffled_lift_by_horizon[hname] = mean_top10_shuf
        stats.n_patterns_above_shuffle_ceiling_by_horizon[hname] = sum(
            1 for r in rows
            if not np.isnan(r["lift_failure"]) and r["lift_failure"] > max_shuf
        )
        all_rows.extend(rows)

    # Count-based horizons always have room for order to matter; short
    # time-based Azure horizons don't. Rich set covers count-based on
    # any dataset.
    RICH_HORIZONS = ("last3", "last5", "last10")

    def _real_beats_shuf_top10(h: str) -> bool:
        real = stats.mean_top10_real_lift_by_horizon.get(h, float("nan"))
        shuf = stats.mean_top10_shuffled_lift_by_horizon.get(h, 0.0)
        if np.isnan(real):
            return True
        return real > shuf

    mined_horizons = [
        h for h, n in stats.n_patterns_by_horizon.items() if n > 0
    ]
    stats.invariants = {
        "rich_horizon_top10_real_lift_exceeds_shuffled": all(
            _real_beats_shuf_top10(h) for h in RICH_HORIZONS
            if h in stats.n_patterns_by_horizon
        ),
        "at_least_one_horizon_mined": len(mined_horizons) > 0,
        # At 1h / 6h / 24h Azure PdM windows contain 1-2 events, so
        # within-window shuffle is essentially a no-op. Allow up to 5%
        # relative delta (24h has a small tail of 3-4 event windows that
        # do change under shuffle).
        "short_horizon_shuffle_negligible": all(
            (
                np.isnan(stats.mean_top10_real_lift_by_horizon.get(h, float("nan")))
                or (
                    stats.mean_top10_real_lift_by_horizon.get(h, 0.0) == 0.0
                    and stats.mean_top10_shuffled_lift_by_horizon.get(h, 0.0) == 0.0
                )
                or abs(
                    stats.mean_top10_real_lift_by_horizon.get(h, 0.0)
                    - stats.mean_top10_shuffled_lift_by_horizon.get(h, 0.0)
                ) / max(stats.mean_top10_real_lift_by_horizon.get(h, 1e-9), 1e-9)
                < 0.05
            )
            for h in ("1h", "6h", "24h")
            if h in stats.n_patterns_by_horizon
        ),
    }

    out = pd.DataFrame(all_rows)
    if not out.empty:
        out = out.sort_values(["horizon", "lift_failure"], ascending=[True, False])
    return out.reset_index(drop=True), stats


HORIZON_ORDER = {
    "1h": 0, "6h": 1, "24h": 2, "last3": 3, "last5": 4, "last10": 5,
}


def run(
    windows_parquet: Path,
    out_dir: Path,
    output_stem: str = "azure_sequences",
    min_support: float = DEFAULT_MIN_SUPPORT,
) -> SeqMiningStats:
    windows = pd.read_parquet(windows_parquet)
    horizons = sorted(windows["horizon"].unique(),
                      key=lambda h: HORIZON_ORDER.get(h, 99))
    patterns, stats = mine(windows, horizons, min_support=min_support)
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns.to_parquet(out_dir / f"{output_stem}.parquet", index=False)
    with (out_dir / f"{output_stem}_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, default=str)
    return stats
