# Data inventory

## Azure PdM (baseline)

- **Kaggle slug:** `arnabbiswas1/microsoft-azure-predictive-maintenance`
- **License string reported by Kaggle CLI:** `unknown` (UNVERIFIED, flagged;
  scout notes downstream mirrors treat this as Microsoft sample data +
  Kaggle ToS).
- **Fetched:** 2026-08-28
- **Location:** `data/raw/azure/`

### Files

| File              | SHA256 (first 16)  | Rows (ex. header) | Notes                                                                 |
| ----------------- | ------------------ | ----------------- | --------------------------------------------------------------------- |
| PdM_errors.csv    | 9c2a2a010ad77227   | 3919              | `datetime,machineID,errorID`. Error1..error5 vocabulary.              |
| PdM_failures.csv  | 0c6c31a4fd52ef2d   | 761               | `datetime,machineID,failure`. comp1..comp4 vocabulary.                |
| PdM_machines.csv  | 5e8e1571c4999bf8   | 100               | `machineID,model,age`. Exactly 100 distinct machines as expected.     |
| PdM_maint.csv     | 481ed4e155f609e6   | 3286              | `datetime,machineID,comp`. Starts **2014-06-01**, precedes telemetry. |
| PdM_telemetry.csv | d957f3c45bb83416   | 876100            | `datetime,machineID,volt,rotate,pressure,vibration`. Hourly.          |

Full SHA256 in `data/raw/azure/CHECKSUMS.sha256`.

### Sanity vs scout report

- Distinct machines: 100 (matches scout expectation).
- Telemetry timestamps: 876,100 rows (matches scout's 876,101 stated
  timestamp count; off by 1 header-line accounting).
- Telemetry window: 2015-01-01 06:00 -> 2016-01-01 (spot-checked below).
- `PdM_maint` extending before telemetry (2014-06-01) is expected: the
  dataset preserves maintenance history for machines whose telemetry
  starts in 2015. Loader must handle this so early maintenance events
  don't become NaN or drop.

### Seed-failure batch (discovered 2026-08-28)

The `PdM_failures` table contains 18 rows at exactly
`2015-01-02 03:00:00` (one or more per machine, various comp codes) that
have NO matching entry in `PdM_maint`. Every failure AFTER that timestamp
does join to a maint row.

Read as a seed batch planted by the synthetic-data generator to
bootstrap failure history at the start of the observation window. Kept
as legitimate `terminal_failure` events. The Azure loader defines
`SEED_FAILURE_TS = 2015-01-02 03:00:00` for downstream code that needs
to filter or annotate them.

Impact: any pattern-mining or predictive-window sampling that requires
a preceding `PdM_maint` row for every failure MUST exclude or
special-case this timestamp. The seed failures have essentially no
pre-failure event history (telemetry starts 2015-01-01 06:00) and
should probably be excluded from Phase 2 window sampling.

## Alibaba cluster-trace-v2018 (primary)

Not yet fetched. See `BACKLOG.md` "Phase 1 — Alibaba primary".
