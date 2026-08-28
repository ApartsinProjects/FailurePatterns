# Research Plan

## Objective

Identify frequent and potentially predictive sequences of operational failures,
errors, retries, maintenance events, and recoveries from timestamped system
logs, and test whether preserving temporal order adds information beyond
unordered co-occurrence.

## Phases and gates

### Phase 1: Dataset construction

Produce a canonical `(entity_id, timestamp, event_type)` table for each source.

- Azure PdM: join `PdM_errors`, `PdM_failures`, `PdM_maint` on `machineID` +
  `datetime`. Vocabulary is small and named by the source.
- Alibaba: extract discrete status transitions (Failed, Terminated, Waiting,
  Running restarts, machine errors) from the task/instance/machine tables.
  Normalize to a shared vocabulary.

Categories to unify across datasets:
`software_error`, `hardware_error`, `task_failure`, `retry`, `eviction`,
`interruption`, `maintenance`, `component_replacement`, `recovery`,
`terminal_failure`.

Gate: sanity plots of event frequencies per entity per day, and a spot-check
of ten random entities' timelines against the raw source. Numbers must
reconcile with the source's own documentation (row counts, distinct entities).

### Phase 2: Analysis units

For every failure event, build pre-failure windows of several sizes (1 h, 6 h,
24 h; last 5 / 10 events). For each failure window, sample matched
control windows from the same entity in periods with no failure within the
horizon.

Gate: per-window artifact stored as parquet, one row per window, containing
`entity_id`, `window_start`, `window_end`, `is_failure`, `event_sequence`,
`event_set`, `target_failure_type` (nullable). Class balance and per-entity
counts logged.

### Phase 3: Frequent itemset mining

Apriori + FP-Growth on `event_set`. Report support, confidence, lift, failure
coverage. Sanity invariant: a random-permuted `is_failure` label must not
yield any high-lift patterns above the same threshold.

### Phase 4: Sequential pattern mining

PrefixSpan on `event_sequence`. Sweep min-support, max-length, max-gap, window
size. Sanity invariant: shuffling event order inside each window must destroy
the sequence-only patterns while leaving the itemset patterns unchanged.

### Phase 5: Pattern significance

For each mined pattern S compute relative risk
`P(Failure | S) / P(Failure | ¬S)` on the matched control windows, plus odds
ratio, lift, and a permutation-based p-value with BH correction. Report only
patterns that survive.

### Phase 6: Predictive evaluation

Turn surviving patterns into binary features. Compare four feature sets on a
temporally-held-out split: event-count baseline, itemsets only, sequences
only, combined. Metrics: precision, recall, F1, AUROC, AUPRC, and lead time
before the failure event.

Gate: the event-count baseline must be reported first, and any claim that
sequences help must beat both the count baseline and the itemset feature set
on the same split.

### Phase 7: Robustness

Stability across machines, time windows, failure types, support thresholds,
and window sizes. Which sequences generalize, which are entity-specific.

## Standing invariants (project-wide)

- Same panel / model / split / seed for any numeric comparison; results
  co-computed in one pass and saved as one artifact.
- Every experiment carries at least one pre-declared invariant (random-label,
  order-shuffle, degenerate baseline) whose expected outcome is stated up
  front.
- Negative and null results stay in `diagnostics/` and the experiment
  registry; the paper reports only what survived.
- Any suspiciously good, bad, or impossible number is a bug until proven
  otherwise.

## Experiment structure (mapping to plan)

- E1: Azure PdM end-to-end pipeline sanity + interpretable component-failure
  patterns.
- E2: Alibaba trace, real-world failure trajectories.
- E3: Itemset vs sequence, matched conditions.
- E4: Predictive utility versus event-count baseline.
