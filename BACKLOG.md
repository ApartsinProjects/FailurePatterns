# Backlog

Open work. When something is picked up, move it into `PROJECT_LOG.md` with a
date and the outcome.

## Now
- [x] Scout done, saved to `docs/scout-2026-08-28.md`, decisions in
      `PROJECT_LOG.md`.
- [ ] Pin Python env (`requirements.txt`): `mlxtend`, `pandas`, `pyarrow`,
      `scikit-learn`, `pytest`, and `spmf.py` (or subprocess-driven SPMF jar)
      for PrefixSpan. `prefixspan` PyPI package as fallback only.
- [ ] Confirm UNVERIFIED-license items from scout: Alibaba clusterdata
      primary-source license, Kaggle Azure PdM license string, direct
      Microsoft statement of Azure PdM synthetic origin.

## Phase 1 — Azure baseline first
- [ ] Fetch Azure PdM via Kaggle CLI (`arnabbiswas1/microsoft-azure-predictive-maintenance`)
      into `data/raw/azure/`. Record checksums.
- [ ] Loader `src/ingest/azure.py`: read the five CSVs
      (`PdM_telemetry`, `PdM_errors`, `PdM_failures`, `PdM_maint`,
      `PdM_machines`), emit `(entity_id, timestamp, event_type)` parquet
      into `data/processed/azure_events.parquet`.
      - Normalize `errorID` -> `software_error` (Azure errors are non-fatal).
      - Normalize `PdM_maint` non-failure events -> `maintenance`.
      - Normalize `PdM_failures` -> `component_replacement` +
        `terminal_failure` marker; note that failures are a strict subset of
        maint.
- [ ] Sanity numbers: row counts per source table, distinct machines (=100
      expected), time range 2015-01-01 to 2016-01-01.
- [ ] Timeline spot-check notebook: ten random machines, side-by-side raw
      CSV vs normalized event stream.

## Phase 1 — Alibaba primary (after Azure pipeline is green)
- [ ] Fetch cluster-trace-v2018 via `fetchData.sh`.
- [ ] Loader `src/ingest/alibaba.py`: extract discrete status transitions
      from `batch_task` / `batch_instance` / `machine_meta`.
      - Handle `end_time == 0` on non-terminated instances (do not treat as
        timestamp 1970-01-01).
      - Strip `task_name` DAG encoding to `task_type`.
      - Map `Failed / Interrupted / Cancelled` -> `task_failure`;
        machine-status changes -> `hardware_error` or `maintenance`.
- [ ] Vocab reconciliation table `docs/vocab-mapping.md`: shared vocab
      column mapped side-by-side to Azure PdM columns and Alibaba columns.
- [ ] Timeline spot-check notebook mirroring the Azure one.

## Phase 2
- [ ] Window builder with window-size sweep and matched controls.
- [ ] Per-window parquet artifact.

## Phase 3-4
- [ ] Itemset miner (mlxtend Apriori + FP-Growth) with the standing
      random-label invariant.
- [ ] Sequence miner (PrefixSpan) with the standing order-shuffle invariant.

## Phase 5
- [ ] Significance layer: RR, OR, permutation p-value, BH correction.

## Phase 6
- [ ] Baseline event-count classifier.
- [ ] Feature-set comparison harness (count / itemset / sequence / combined).

## Phase 7
- [ ] Cross-entity, cross-period, cross-failure-type stability sweep.

## Deferred / nice-to-have
- [ ] SPADE and GSP as sanity cross-checks against PrefixSpan.
- [ ] Interactive pattern browser (small Streamlit or plain HTML) for the
      final `results/patterns/` set.
