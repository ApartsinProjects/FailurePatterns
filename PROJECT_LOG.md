# Project Log

Chronological. Newest entry at the top. Each entry: what changed, why, what
comes next.

## 2026-08-28 — Paper-reviewer pass + first HTML/DOCX render

**Reviewer pass** on [paper/skeleton.md](paper/skeleton.md). Report at
[paper/reviewer_pass_1.md](paper/reviewer_pass_1.md). Found 4 blocking
issues and 3 substantive:
- W1 (SC-4 violation): §7 Limitations still said "No lead-time metric
  reported yet" even though §5.6 was added. Fixed.
- W2: §6 Discussion was a TODO placeholder. Filled with per-signature
  operational readings for both the Azure `error2 → error3` pattern
  and the Alibaba `M → M → M` pattern, plus a deployment note about
  the two lead-time regimes.
- W3: §2 Related Work was a stub. Rewrote as four cited paragraphs
  with pandoc `[@key]` cite calls into every entry in
  [paper/references.bib](paper/references.bib).
- W4 (SC-3 abstract-body parity): abstract claimed "same shape appears
  on Alibaba" without a concrete Alibaba pattern in the body. Added
  the two Alibaba patterns to §5.2 with sequence-vs-itemset lift
  numbers and rewrote the abstract to quote both concrete signatures.
- W5-W7 also addressed: abstract now names the specific per-dataset
  AUROC gains (+5.6 Azure, +6.2 Alibaba) and points at reproducible
  parquet artifacts; §5.3 gives precision AND recall ranges;
  §5.6 relabels Azure as "structural lead time set by the generator"
  vs Alibaba as "operational lead time".
- Wins-only scanner: clean, 0 hits across SC-15 patterns.
- Tone audit: 0 forbidden phrases, 0 em-dashes, 0 defensive framings.
- **Numbers audit re-run: 50/50 pass** (up from 44/44 after adding 6
  new claims). Every new abstract number and every new §5.2 sequence
  is programmatically verified against the parquet stats.

**First render** via pandoc:
- [paper/skeleton.html](paper/skeleton.html), 40 KB, embedded CSS
  (`paper/style.css`, "printed page" light identity with archival-blue
  accent, single-theme by design with explicit background so it holds
  on any host ground). Published as an Artifact:
  https://claude.ai/code/artifact/2b874f10-92e9-486b-8250-d7445dc88509.
- [paper/skeleton.docx](paper/skeleton.docx), 24 KB, native Word for
  co-author edits.
- Both include auto-generated TOC, numbered sections, and the full
  12-entry CSL reference list rendered from `references.bib`.

Paper is out of skeleton state; ready for internal review before
resubmission for a second reviewer pass and a final `bibtest` gate.

## 2026-08-29 — Four-trace expansion: BGL + SCANIA folded as boundary conditions

Added two new production traces to the study following the user's
conference-tier request.

**BGL (Loghub Blue Gene/L, per-rack).** 4,747,963 syslogs from LLNL
2005-06 to 2006-01; 913k events after dropping INFO noise; 64 racks;
348k alerts (7.34%). Loader [src/ingest/bgl.py](src/ingest/bgl.py)
parses `label`, `severity`, `component` fields; alerts map to
`terminal_alert`, severe non-alerts to `system_error`, warnings to
`system_warning`, INFO to `system_info`. Window sampler
[src/eval/windows_bgl.py](src/eval/windows_bgl.py) groups alerts into
episodes with a 1h inter-arrival threshold (7,224 episodes from 348k
alerts, 48× compression) and anchors on the first alert per episode.
Critical design fix: alerts are removed from the pre-alert event
stream so mining does not learn the trivial "alert follows alert"
signal.

**Result: BGL is a boundary case.** Best combined AUROC across all
horizons is 0.51 (chance). Non-alert log lines carry no discriminable
precursor signal on BGL, even with INFO included and component
granularity added. The mechanism: BGL alerts self-trigger in dense
cascades, so predicting "what non-alert events precede an alert
episode" reduces to background noise.

**SCANIA Component X (Nature Sci Data 2025, per-vehicle).** 23,550
trucks × 1.12M readouts × 105 numeric counter/histogram features;
2,272 vehicles (9.65%) undergo a component-X repair during the
study. Loader [src/ingest/scania.py](src/ingest/scania.py) converts
per-vehicle counter deltas to `counter_surprise` tokens (per-vehicle
90th-percentile threshold; changed from fleet-wide global threshold
after the initial version was too crude). Window sampler
[src/eval/windows_scania.py](src/eval/windows_scania.py) uses the
Alibaba-style cross-entity control design.

**Result: SCANIA is also a boundary case.** Best combined AUROC is
0.60 (last10) with per-vehicle binning; global binning gave 0.61.
Per-vehicle normalisation did not rescue the signal. Mechanism:
discrete tokens derived from continuous counters do not preserve
the ordered structure that pattern mining exploits.

**Cross-dataset final result (four traces):**

| trace   | best combined AUROC | verdict |
|---|---|---|
| Azure PdM (24h) | 0.996 | strong win |
| Alibaba v2018 (last3) | 0.813 | strong win |
| BGL (last20) | 0.512 | boundary — self-triggering alerts |
| SCANIA (last10) | 0.596 | boundary — derived tokens |

**Paper updates.**
- Restructured to canonical scientific layout: Abstract → Intro →
  Related Work → Data → Method → Experiments → Results → Discussion
  → Limitations → Conclusion.
- §2 Related Work expanded to 7 subsections with 20 new citations
  covering DL log anomaly (DeepLog, LogBERT, LogAnomaly, LogRobust,
  PLELog, LogFormer), sequence-mining variants (SPADE, GSP,
  CM-SPAM, VMSP), log-parsing infrastructure (Loghub, Drain), trace
  characterisation (Oliner BGL, Luo microservice), PdM broader
  (C-MAPSS, Serradilla, SCANIA IDA 2016).
- §7.2 "Regime of validity" formalises the two-wins-two-boundaries
  finding with the mechanistic explanation.
- Bibliography expanded from 12 to 29 validated entries;
  [bibtest](https://github.com/anthropics/claude-skills) caught two
  more hallucinated citations (buddhakulsomsiri wrong DOI, que2024
  wrong authors) which were removed.
- Numbers audit re-run: **58 / 58 pass** (up from 50 / 50 after adding
  BGL + SCANIA claims).
- Four-trace comparison figure at
  [results/figures/four_dataset_predictive_comparison.png](results/figures/four_dataset_predictive_comparison.png).

Failure Sequences Paper artifact republished at the same URL:
https://claude.ai/code/artifact/2b874f10-92e9-486b-8250-d7445dc88509

## 2026-08-28 — Bibliography, lead-time, numbers audit

**Lead-time metric** added to [src/eval/predict.py](src/eval/predict.py).
Windows now carry `last_event_ts`; predict.py computes lead time =
anchor - last_event_ts for TP failure windows on the held-out set. Both
[scripts/build_windows_azure.py](scripts/build_windows_azure.py) and
[scripts/build_windows_alibaba.py](scripts/build_windows_alibaba.py)
re-ran green. Predictive eval re-ran for both datasets; AUROC / AUPRC
unchanged (44/44 numbers still audit clean).

Findings:
- Azure lead times cluster at exactly 24h (the synthetic generator
  plants errors ~one day before failures).
- Alibaba lead times are ~seconds (task boundaries within a job).

**Bibliography** at [paper/references.bib](paper/references.bib), 12
entries. Validated via the `bibtest` skill against Crossref / OpenAlex
/ arXiv:
- 10 / 12 valid with intended-paper resolution confirmed.
- 2 not_found are legitimate grey-literature dataset URLs
  (Alibaba clusterdata repo, Kaggle Azure PdM mirror) with no DOI.
- One correction: `bibtest` caught that I had HALLUCINATED author names
  for the 2024 J. Supercomputing bus-fleet paper (`buscher` vs the
  correct `ifraz2024sequential` = Metin İfraz + Süleyman Ersöz). The
  DOI was correct; only the authors were wrong. Fixed.

**Numbers audit** at [paper/numbers_audit.md](paper/numbers_audit.md).
Programmatic check: [scripts/audit_paper_numbers.py](scripts/audit_paper_numbers.py)
resolves every quantitative claim in [paper/skeleton.md](paper/skeleton.md)
against the parquet / JSON stats that produced it. Result: **44 / 44
claims verified**. Any future skeleton edit that drops in a number
without a matching stats-file entry will fail this audit.

## 2026-08-28 — Paper skeleton, Phase 5, Phase 7 (scoped)

**Paper skeleton** at [paper/skeleton.md](paper/skeleton.md).
Abstract, intro, methods, results, discussion, limitations,
conclusion. Every headline number cross-referenced to the JSON stats
or parquet it came from. TODO markers name what is still missing
(bibliography, lead-time metric, HTML/DOCX build).

**Phase 5 (per-pattern significance)** at
[src/eval/significance.py](src/eval/significance.py). Insight: the
label-permutation p-value with fixed pattern hit-set IS exactly the
one-sided hypergeometric tail, so we compute it in closed form
instead of Monte Carlo. BH correction applied per (horizon x pattern
class).
- Azure: at BH q<0.05, 6/6 itemsets and 7/7 sequences significant at
  24h; 53/77 itemsets and 55/67 sequences at last5; 562/657
  sequences at last10. 1h and 6h flag zero as expected.
- Alibaba: 6/10 itemsets and 9/16 sequences at last3; 59/109
  sequences at last10.
- Outputs: `results/patterns/{azure,alibaba}_{itemsets,sequences}_significance.parquet`
  and `results/patterns/{azure,alibaba}_significance_summary.json`.

**Phase 7 (min_support sensitivity)** at
[scripts/sensitivity_azure.py](scripts/sensitivity_azure.py). Full
Phase 6 re-run at min_support in {0.02, 0.05, 0.10, 0.15} on Azure.
Result: the "combined dominates itemsets_only" ordering holds at
every operating point tested, with combined - itemsets_only >= +4
AUROC on last5 and >= +5 AUROC on last10 across the sweep.
- Output: [results/figures/azure_sensitivity_min_support.png](results/figures/azure_sensitivity_min_support.png),
  [results/tables/azure_sensitivity_min_support.parquet](results/tables/azure_sensitivity_min_support.parquet).

Both are folded into paper/skeleton.md sections 5.5 and 5.6. The old
"Phase 5/7 not yet computed" limitation is dropped from the paper.

## 2026-08-28 — Alibaba primary study end-to-end

Full pipeline replay on Alibaba cluster-trace-v2018.

**Fetch decision** (see `E:/tmp/alibaba/`): pulled only
`machine_meta.tar.gz` (92 KB) and `batch_task.tar.gz` (130 MB / 802 MB
uncompressed / 14.3M rows), skipping the 21 GB `batch_instance.tar.gz`
for this first pass. `batch_task` alone carries 83,276 Failed task rows
across 83,207 distinct jobs — plenty of failure signal for a first-pass
per-job study.

**Loader** [src/ingest/alibaba.py](src/ingest/alibaba.py) normalized
`batch_task.csv` into the shared event vocab. Entity = `job_name`. Event
type = normalized status (`task_failure` / `task_success` /
`task_waiting` / `task_running`). Event subtype = task_name letter
prefix (M / R / J / task / MergeTask / L). Handled the scout's
`end_time == 0` gotcha with a fallback to `start_time` (78,311 rows
affected). All load invariants pass.

**Window sampler** [src/eval/windows_alibaba.py](src/eval/windows_alibaba.py)
built per-job windows. Different design from Azure because Alibaba jobs
are short (minutes-hours, few events):
- Failure windows: last K events strictly before the first
  `task_failure` per job (16,237 eligible jobs; the other 66K failure
  jobs had the failure as their first event).
- Control windows: last K events of a non-failure job sampled from a
  pool of 1,826,069 candidates. 3 controls per failure window.
- Horizons: `last3`, `last5`, `last10` (time-based skipped: jobs are
  too short for hour-scale horizons).
- All five invariants pass.

**Phase 3 (itemsets):** [scripts/mine_alibaba_itemsets.py](scripts/mine_alibaba_itemsets.py).
Real top lifts 4.00 / 3.97 / 3.96 vs permuted top ~1.02 across the
three horizons: essentially perfect signal-vs-noise separation. 5-6
patterns survive permutation null per horizon.

**Phase 4 (sequences):** [scripts/mine_alibaba_sequences.py](scripts/mine_alibaba_sequences.py).
Mean top-10 real lift vs shuffled:
- last3: 2.57 vs 1.64 (+0.93)
- last5: 2.65 vs 1.52 (+1.13)
- last10: 2.71 vs 2.34 (+0.37)
Order carries measurable signal at all three horizons. Only 2 sequences
per horizon survive the shuffle null (much fewer than Azure's 6-16).

**Phase 6 (predictive eval):** [scripts/eval_alibaba_predict.py](scripts/eval_alibaba_predict.py).
Temporal split at 2018-01-07. AUROC / AUPRC:

| horizon | event_count | itemsets_only | sequences_only | combined |
|---|---|---|---|---|
| last3  | 0.69 / 0.50 | 0.75 / 0.44 | 0.50 / 0.20 | **0.81 / 0.63** |
| last5  | 0.60 / 0.50 | 0.67 / 0.34 | 0.51 / 0.21 | **0.74 / 0.57** |
| last10 | 0.59 / 0.50 | 0.68 / 0.36 | 0.52 / 0.23 | **0.74 / 0.59** |

**Cross-dataset comparison:** the same finding holds on BOTH datasets:
- Azure combined @ last5: AUROC 0.810, AUPRC 0.720.
- Alibaba combined @ last3: AUROC 0.813, AUPRC 0.631.
- On both, combined >> itemsets_only >> sequences_only alone.
- On both, sequences_only has few surviving features but very high
  precision (0.95+) at the 0.5 threshold — the ordered patterns fire
  rarely but reliably.

Figures:
- [results/figures/alibaba_predictive_comparison.png](results/figures/alibaba_predictive_comparison.png)
- [results/figures/cross_dataset_predictive_comparison.png](results/figures/cross_dataset_predictive_comparison.png)

Refactors made during the Alibaba pass:
- `src/eval/windows.py`: `build_windows` now accepts
  `failure_event_type`, `seed_timestamps`, `expected_seed_count`.
  Azure runner passes its Azure-specific values; Alibaba does not use
  this module (per-job windows are structurally different).
- `src/mine/itemsets.py` + `src/mine/sequences.py`: `run` takes an
  `output_stem`. HORIZON_ORDER extended to include `last3`.
- `src/mine/*` invariants generalized: instead of hard-coding
  ("24h", "last5", "last10"), require every MINED horizon (n_patterns > 0)
  to have surviving patterns.
- `src/mine/sequences.py`: RICH_HORIZONS covers `last3` too.
- `src/eval/predict.py`: `run` takes horizons + cutoff so the Alibaba
  runner can use its own (2018-01-07, last3/last5/last10).

**Every Azure pipeline stage re-run at the end and still green.**

## 2026-08-28 — Azure Phase 6 predictive evaluation done

Head-to-head on a temporal split (train: anchor < 2015-09-01, test:
anchor >= 2015-09-01), patterns re-mined on TRAIN only.
[src/eval/predict.py](src/eval/predict.py) via
[scripts/eval_azure_predict.py](scripts/eval_azure_predict.py).

Table (AUROC / AUPRC, higher is better):

| horizon | event_count | itemsets_only | sequences_only | combined |
|---|---|---|---|---|
| 24h    | 0.97 / 0.91 | 0.996 / 0.99 | -- (no features survived null) | 0.996 / 0.99 |
| last5  | 0.50 / 0.34 | 0.75 / 0.56  | 0.66 / 0.56 | **0.81 / 0.72** |
| last10 | 0.50 / 0.34 | 0.64 / 0.50  | 0.67 / 0.53 | **0.70 / 0.58** |

**Answer to Experiment 4:** on Azure PdM, temporal order in mined
sequences contributes real predictive information beyond the itemset
representation, but ONLY at rich (count-based) horizons where order is a
real degree of freedom. At 24h everything the mined patterns capture is
already carried by the count baseline and the itemsets; sequences add
nothing because no sequence beat its own shuffle-null. At last5 and
last10 the combined feature set beats itemsets-only by +5.3-5.6 AUROC
points and +7.6-15.7 AUPRC points.

Deliverables:
- [results/tables/azure_predictive.parquet](results/tables/azure_predictive.parquet),
  [results/tables/azure_predictive_stats.json](results/tables/azure_predictive_stats.json).
- [results/tables/azure_predictive.md](results/tables/azure_predictive.md).
- [results/figures/azure_predictive_comparison.png](results/figures/azure_predictive_comparison.png).

Feature-leakage guard: itemset and sequence mining ran on training
windows only. Survivors of the training-set permutation null were
carried forward as features and applied unchanged to the held-out test
set. No test information touches feature selection.

Substantive caveat: n_features imbalance is real — combined at last10
has 348 features on 2276 training windows (~6.5 obs/feature). Logistic
regression with default L2 is holding up fine (test AUROC 0.70), but a
proper Phase 7 stability sweep on regularization / feature-count is
warranted before writing the paper.

Next: Phase 5 (permutation p-values with BH correction), Phase 7
(robustness sweep across machines, time periods, thresholds), or start
the Alibaba primary study. Recommend Alibaba next: Azure Experiment 1
result is complete enough for a first draft, and the Alibaba trace is
where the paper's real production-data claim lives.

## 2026-08-28 — Azure Phase 3 + 4 mined; order gives real signal

Phase 3 (itemsets) and Phase 4 (sequences) both green on Azure PdM. All
pre-declared invariants pass. Substantive scientific findings:

**Phase 3 (FP-Growth, [src/mine/itemsets.py](src/mine/itemsets.py) via
[scripts/mine_azure_itemsets.py](scripts/mine_azure_itemsets.py)):**

- 1h + 6h: 0 patterns (windows too sparse). Expected.
- 24h: 6 patterns, top lift 3.99 for `{software_error:error2,
  software_error:error3}` (present in 38.2% of failure windows, in 0.04%
  of controls). All 6 dominate the random-label permutation null
  (permuted top lift 1.24). P(failure | pattern) = 99.6%.
- last5: 77 patterns, 45 above permutation ceiling.
- last10: 730 patterns, 214 above permutation ceiling.
- Sanity: random-label permutation catches multiple-testing tail
  correctly; the "real top > 1.5 x permuted top" invariant discriminates
  signal from noise cleanly at every horizon.

**Phase 4 (PrefixSpan via SPMF 2.64,
[src/mine/sequences.py](src/mine/sequences.py) via
[scripts/mine_azure_sequences.py](scripts/mine_azure_sequences.py)):**

- Order-shuffle invariants pass on both classes of horizon:
  - At 1h / 6h / 24h the mean top-10 shuffled lift equals the real
    lift within 5% (sequences 1-2 events, no order degree of freedom).
  - At last5 / last10, real top-10 EXCEEDS shuffled by +0.36 and +0.77.
- **Key result:** the sequence
  `maintenance:comp4 -> software_error:error2 -> software_error:error3`
  reaches lift 3.73 as an ordered pattern but only 2.22 as an itemset
  (same items, order-blind). Order-gain +1.51. Similar
  `error2 -> error3`-terminating signatures appear across the top-8 at
  both last5 and last10.
- Order-gain distribution:
  - 1h / 6h: mean 0 (no room to shuffle).
  - 24h: mean 0.002 (essentially no room).
  - last5: mean 0.31, max 1.51.
  - last10: mean 0.32, max 1.52.

**Deliverables:**

- Pattern parquets: [results/patterns/azure_itemsets.parquet](results/patterns/azure_itemsets.parquet),
  [results/patterns/azure_sequences.parquet](results/patterns/azure_sequences.parquet).
- Comparison figure: [results/figures/azure_itemset_vs_sequence_lift.png](results/figures/azure_itemset_vs_sequence_lift.png).
- Top-patterns table: [results/tables/azure_top_patterns.md](results/tables/azure_top_patterns.md).
- Stats JSONs live next to each parquet.

**Correction I made during this session:** first invariant version used
an ABSOLUTE lift threshold of 1.5 that failed at last10 (permuted-null
tail is inflated by 730 hypotheses). Rewrote as a RATIO invariant "real
top >= 1.5 x permuted top" so wide horizons with more hypotheses use a
correspondingly higher significance bar. Logged the finding rather than
tuning around it.

**Next steps** (waiting for user):

- Phase 5: proper permutation-based p-values with BH correction, on top
  of the current mined-vs-shuffled ratio, for pattern-level significance.
- Phase 6: predictive evaluation. Event-count baseline vs itemset
  features vs sequence features vs combined, on a temporally-held-out
  split. This is where the "does sequence mining actually help
  downstream" question gets answered head-to-head.
- Alibaba primary study (Phase 1 loader for cluster-trace-v2018).

Any of the three is a reasonable next step. Recommend Phase 6 next
because it directly closes the Experiment 4 loop and lets the paper
claim early-warning utility, not just associational lift.

## 2026-08-28 — Azure Phase 2 windows built; horizons characterised

- SPMF jar pinned at v2.64 (Java-21-compatible) at `scripts/spmf.jar`;
  v2.66 requires Java 25. PrefixSpan smoke-test passes.
  See [docs/tools.md](docs/tools.md).
- Window sampler [src/eval/windows.py](src/eval/windows.py) run via
  [scripts/build_windows_azure.py](scripts/build_windows_azure.py) produced
  `data/processed/azure_windows.parquet`:
  - 743 failure windows per horizon x 5 horizons (1h, 6h, 24h, last-5,
    last-10). 3 matched control windows per failure = 2229 controls per
    horizon.
  - 18 seed failures excluded from the event stream ENTIRELY (not just as
    anchors), so they cannot contaminate windows of subsequent real
    failures.
  - All five pre-declared invariants pass.
- **Second discovery, worth surfacing in the paper:** the 1h and 6h
  horizons are effectively useless on Azure PdM. 99.6% of 1h failure
  windows are empty; 98% of 6h. Immediate pre-failure minutes are quiet.
  The informative horizons are 24h and the count-based (last-5, last-10).
- **Third discovery, foreshadows the predictive baseline:** at 24h, event
  count ALONE separates failure vs control cleanly (failure mean 1.58,
  every window >= 1 event; control mean 0.077, 93.2% empty). This makes
  the event-count baseline strong. Any later claim that itemset or
  sequence mining adds information must beat that baseline on the same
  split.
- **Sanity spot-check (10 random machines)** in
  [diagnostics/azure_windows_summary.md](diagnostics/azure_windows_summary.md)
  already shows the target pattern: machine 23 has
  `error2 -> error3 -> comp2 failure`, machine 27 has
  `error3 -> error2 -> comp2 failure`. Same error set, different order,
  same failure — this is exactly the itemset-vs-sequence question.
- Figure: [diagnostics/azure_window_horizon_vs_events.png](diagnostics/azure_window_horizon_vs_events.png).
- Correction that got applied: my initial invariant "no terminal_failure
  appears inside any failure window" was too strict — legitimate failure
  cascades (failure A precedes failure B within horizon) count as signal,
  not contamination. Rewrote to check "the anchor failure itself is not
  in its own window" and confirmed via slice construction.
- Next: Phase 3 (itemset mining) + Phase 4 (sequence mining) on the
  informative horizons (24h, last-5, last-10). Recommend a user
  check-in first — the "1h/6h are useless" and "event-count is strong
  baseline" findings shape the rest of the plan.

## 2026-08-28 — Azure Phase 1 loader green; seed-failure discovery

- Env pinned in [requirements.txt](requirements.txt); `.venv` at
  `.venv/Scripts/python.exe` with `mlxtend 0.25.0`, `pandas 3.0.5`,
  `pyarrow 25.0.1`, `scikit-learn 1.9.0`, `prefixspan 0.5.2`,
  `kaggle 2.2.4`. FP-Growth and PrefixSpan smoke tests pass on toy data.
- Kaggle CLI authenticates from the existing `~/.kaggle/access_token`
  (no `kaggle.json` needed with kaggle 2.2.x).
- Azure PdM fetched to [data/raw/azure/](data/raw/azure/); five CSVs, SHA256
  recorded in [data/raw/azure/CHECKSUMS.sha256](data/raw/azure/CHECKSUMS.sha256),
  inventory in [docs/data-inventory.md](docs/data-inventory.md). Kaggle
  reports license as "unknown" (flagged UNVERIFIED).
- Loader [src/ingest/azure.py](src/ingest/azure.py) run via
  [scripts/ingest_azure.py](scripts/ingest_azure.py) produced
  `data/processed/azure_events.parquet` (7,966 events, 100 machines,
  2014-06-01 -> 2016-01-01) and `azure_telemetry.parquet` (876,100 rows).
- **Discovery, not a bug:** the initial invariant "every PdM_failures row
  joins to a PdM_maint row on (machineID, datetime, comp)" failed at
  97.6%. Inspection showed all 18 unmatched rows sit at exactly
  `2015-01-02 03:00:00` — a seed batch planted by the synthetic-data
  generator to bootstrap the simulation. Every failure AFTER that timestamp
  joins cleanly. Invariant rewritten as
  `non_seed_failures_all_matched_to_maint` (scoped to failures outside the
  seed timestamp) and now passes. The 18 seed events are kept as
  legitimate `terminal_failure` records. Constant
  `SEED_FAILURE_TS = 2015-01-02 03:00:00` is documented in the loader for
  downstream code.
- Event vocabulary observed: 3,919 `software_error`, 2,543 `maintenance`,
  743 `component_replacement`, 761 `terminal_failure`.
- All four pre-declared invariants now pass (100 machines, vocab subset,
  per-entity monotonic timestamps, non-seed failure/maint join).
- Next: SPMF jar for PrefixSpan (task #2), then Phase 2 window builder.
  Recommend a user check-in first — the seed-failure decision affects
  every downstream pattern significance test.

## 2026-08-28 — Scout returned; dataset choices locked

Scout report saved verbatim at [docs/scout-2026-08-28.md](docs/scout-2026-08-28.md).
Concrete decisions from it:

- **Primary dataset:** Alibaba **cluster-trace-v2018** (not v2017, not
  microservices, not gpu-v2020). ~4000 machines, 8 days, discrete status
  vocab `Ready|Waiting|Running|Terminated|Failed|Cancelled|Interrupted` on
  `batch_task` / `batch_instance`. Public fetch script; ~98 GB compressed.
- **Baseline dataset:** Azure PdM via Kaggle
  `arnabbiswas1/microsoft-azure-predictive-maintenance`. Synthetic per every
  downstream mirror; kept as a small clean corpus with an explicit
  `error -> component_failure` link, matching PLAN Experiment 1's role of
  validating the pipeline before real data.
- **Order of work:** Azure PdM first (100 machines, <1 GB, fastest to
  smoke-test the whole Phase 1-6 pipeline), then Alibaba v2018 primary
  study. This diverges from the scout's "Alibaba first" recommendation but
  is consistent with the plan's own Experiment 1 -> Experiment 2 ordering
  and lets us shake out the sanity invariants (random-label, order-shuffle,
  event-count baseline) on cheap data first.
- **Novelty confirmed:** no prior peer-reviewed paper applies Apriori /
  FP-Growth / PrefixSpan directly to either Alibaba v2018 batch_instance
  status transitions or Azure PdM `errorID -> failure`. Closest neighbours
  (Ren 2021 on BlueGene/LANL; bus-fleet 2024) leave both traces untouched.
- **Library stack locked:** `mlxtend` for FP-Growth/Apriori; SPMF (Java) via
  `spmf.py` or subprocess for PrefixSpan. The pure-Python `prefixspan`
  package looks semi-abandoned per Snyk and is a fallback only.
- **Gotchas queued** for Phase 1 loaders: Alibaba `task_name` DAG encoding
  needs stripping to `task_type`; Alibaba disk sentinel values `-1` and
  `101`; Alibaba `end_time == 0` on non-terminated instances; Azure
  `PdM_failures` is a strict subset of `PdM_maint`; gpu-v2020 polarity trap
  (out of scope but noted).
- **UNVERIFIED items** flagged in scout report and in `BACKLOG.md`.
- Next: pin `requirements.txt`, then write the Azure PdM loader and vocab
  normalizer as the first Phase 1 deliverable.

## 2026-08-28 — Project bootstrapped

- Created directory tree (`data/`, `src/`, `experiments/`, `results/`,
  `diagnostics/`, `scripts/`, `docs/`, `notebooks/`).
- Wrote `README.md`, `PLAN.md`, `BACKLOG.md`.
- Launched a `web-researcher` scout for (a) canonical Alibaba + Azure PdM
  dataset endpoints, schemas, licenses, and (b) prior sequence-mining work on
  either trace, with a gap analysis.
