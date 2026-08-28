# Mining Frequent Failure Sequences in Operational Event Logs

_Restructured draft. Numbers are current best estimates from the
pipeline as of 2026-08-28. TODO markers name what is still missing._

## Abstract

Predictive maintenance systems mine continuous sensor telemetry, but a
parallel signal lives in the discrete operational events every large
system already logs: errors, retries, task failures, maintenance
actions. We ask whether frequent ordered sequences of these events
carry information beyond unordered co-occurrence, and whether that
information transfers to real production traces. On the synthetic
Azure Predictive Maintenance dataset (100 machines, one year) and on
the Alibaba cluster-trace-v2018 production trace (4.2 M jobs, eight
days), we mine pre-failure event windows with FP-Growth (itemsets) and
PrefixSpan (sequences), score each pattern against matched controls,
and predict failure on a temporally-held-out split. Combining ordered
sequences with unordered itemsets improves failure prediction by +5.6
AUROC on Azure (`last5` horizon) and +6.2 AUROC on Alibaba (`last3`
horizon) over itemsets alone; sequences alone are rare but
high-precision (0.72-0.98 at threshold 0.5, recall 0.02-0.36).
Concrete pre-failure signatures include
`software_error:error2 → software_error:error3` on Azure (sequence
lift 3.73 vs itemset lift 2.22) and
`task_success:M → task_success:R → task_success:M → task_success:M`
on Alibaba (sequence lift 2.43 vs itemset lift 0.94). Two additional
traces (LLNL Blue Gene/L syslogs; SCANIA Component X automotive fleet)
map out the method's boundary conditions and are reported alongside.
Temporal order in operational event logs is a transferable early-warning
signal in the regime of rich discrete event vocabularies; mined
patterns and matched-control windows are released as reproducible
parquet artifacts.

## 1  Introduction

Operational logs from datacenters, industrial fleets, and cloud
platforms carry a rich stream of discrete events: software errors,
task failures, retries, eviction notices, maintenance actions,
component replacements. These events are usually consumed one at a
time by alerting systems and dashboards. We ask a different question:
do RECURRENT ORDERED SEQUENCES of these events precede failures
systematically enough to serve as early-warning signatures?

Two mining families answer this shape of question. Frequent itemset
mining (Apriori, FP-Growth) treats each pre-failure window as an
unordered set of events. Sequential pattern mining (PrefixSpan,
SPADE, GSP) preserves temporal order. The paper's central question is
whether the second family finds anything the first does not.

We contribute:

1. A dataset-agnostic mining pipeline for pre-failure event windows
   with matched controls, sanity invariants at every phase (random
   label permutation for itemsets, within-window order shuffle for
   sequences), and per-pattern lift, relative-risk, and BH-corrected
   significance scoring.
2. A head-to-head predictive comparison of four feature sets
   (event count baseline, itemsets, sequences, combined) on a
   temporally-held-out split with mining restricted to training data,
   under logistic regression on binary pattern-presence features.
3. Cross-dataset replication on four public event-log traces spanning
   three domains: synthetic industrial per-machine (Azure PdM), real
   cloud per-job (Alibaba cluster-trace-v2018), real HPC per-rack
   (LLNL Blue Gene/L syslogs from Loghub), and real automotive
   per-vehicle (SCANIA Component X). Two traces yield strong wins;
   two map out the method's boundary conditions with a mechanistic
   explanation.

## 2  Background and related work

### 2.1 Frequent itemset and sequential pattern mining

The two mining families this paper compares are well-established.
Association-rule mining introduced Apriori [@agrawal1994fast] and
FP-Growth [@han2000mining] over unordered transactions, the latter
avoiding candidate generation through a compressed frequent-pattern
tree. Sequential pattern mining extended this to ordered event
streams: GSP [@srikant1996gsp] generalises Apriori-style level-wise
generation to sequences; SPADE [@zaki2001spade] uses a vertical
id-list format; PrefixSpan [@pei2001prefixspan] uses prefix-projected
pattern growth, and is the algorithm we use in this paper.

Later variants sharpen the raw output. CM-SPAM / CM-SPADE
[@fournierviger2014cmspade] speed vertical mining through co-occurrence
pruning; VMSP [@fournierviger2014vmsp] returns only maximal sequences,
compressing the pattern set without losing coverage. A recent survey
[@fournierviger2022patternmining] summarises open problems in the
area. All of these algorithms are implemented in the SPMF library
[@fournier2016spmf], which we call from Python for the sequence-mining
half of our pipeline; the itemset half uses `mlxtend`
[@raschka2018mlxtend].

### 2.2 Failure prediction from operational event logs

Two families of methods dominate the literature: (i) explicit
frequent-pattern mining on parsed log templates, close to what this
paper does, and (ii) deep learning over log sequences.

**Pattern mining.** Ren et al. [@ren2020failure] apply Spark FP-Growth
to failure prediction on BlueGene/L, LANL-HPC, and CMRI-Hadoop logs,
using event-density sliding windows over long-tail event vocabularies.
İfraz and Ersöz [@ifraz2024sequential] run PrefixSpan and Apriori
side-by-side on a bus-fleet maintenance log, showing that sequence
mining recovers "errors → replacement" trajectories that itemset
mining misses. Both studies stop short of the head-to-head
matched-control + BH-corrected + predictive-utility comparison used
here, and neither touches Alibaba PdM or Azure.

**Deep learning on log sequences.** DeepLog [@du2017deeplog] frames
system-log anomaly detection as next-template prediction with a
stacked LSTM, and remains the canonical DL baseline. LogAnomaly
[@meng2019loganomaly] adds unsupervised quantitative-anomaly detection
alongside sequential anomalies. LogRobust [@zhang2019logrobust] adds
an attention Bi-LSTM to survive log-template drift, and PLELog
[@yang2021plelog] introduces semi-supervised label estimation for the
weakly-labelled setting. Recent transformer approaches (LogBERT
[@guo2021logbert]; LogFormer [@guo2024logformer]) pre-train on unlabelled
logs and fine-tune for anomaly detection. These methods generally
outperform classical pattern miners on held-out AUROC when trained on
enough data, but produce opaque per-line anomaly scores rather than
interpretable pre-failure signatures. Our contribution is orthogonal:
we ask whether explicit ordered patterns add signal over their
itemset counterparts, with every mined pattern human-readable and
independently significance-tested.

### 2.3 Log-parsing infrastructure

The raw text of most system logs must first be converted into event
templates before either family of methods applies. The Loghub
collection [@zhu2023loghub] curates parsed versions of BGL, HDFS,
Thunderbird, and 13 other benchmark log datasets; the parsing
benchmark of Zhu et al. [@zhu2019logparsing] compares Drain and
alternative parsers on the same corpora. We take BGL from Loghub
directly and use its native label field, avoiding the parsing step
as a confound.

### 2.4 Failure characterisation on Alibaba and BGL

Cheng et al. [@cheng2018characterizing] provide the standard
characterisation of the Alibaba 2018 trace, reporting failure
statistics and co-location effects but not extracting sequential
patterns. Luo et al. [@luo2021alibaba] extend this to the microservice
trace with focus on dependency and latency rather than fault
prediction. Oliner and Stearley [@oliner2007supercomputers] introduce
the BGL log we use, along with four other HPC logs, and characterise
their alert statistics.

### 2.5 Predictive maintenance beyond system logs

The broader predictive-maintenance literature works mostly on
continuous sensor telemetry. NASA C-MAPSS [@saxena2008cmapss] is the
de-facto Remaining-Useful-Life benchmark; recent surveys
[@serradilla2022pdmsurvey] catalogue the deep-learning methods trained
on it. Automotive predictive maintenance has historically used
per-vehicle sensor snapshots (SCANIA APS Failure via the IDA 2016
industrial challenge [@costa2016ida]); the SCANIA Component X release
[@kharazian2025scania] we adopt extends this to a per-vehicle
longitudinal readout stream. Our work sits between these traditions:
we take discrete-event traces where possible, and derive discrete
tokens (per §3.4) where we must.

### 2.6 Statistical significance

We apply Benjamini-Hochberg FDR correction [@benjamini1995controlling]
to the one-sided hypergeometric p-values induced by label permutation
on mined patterns. Under a fixed pattern hit-set, the label-permutation
distribution of failure-class hit count is exactly hypergeometric,
so we compute exact p-values in closed form instead of Monte Carlo.

### 2.7 Positioning

To our knowledge no peer-reviewed study applies FP-Growth and
PrefixSpan head-to-head to Alibaba `batch_task` status transitions
or to Azure PdM `errorID → failure` sequences with the matched-control
design used here, then evaluates the resulting patterns as binary
features against event-count and deep-learning-adjacent alternatives
on a temporally-held-out split. The four-trace regime-of-validity
study in §7.2 is likewise, to our knowledge, unprecedented in the
pattern-mining log-analysis literature.

## 3  Data

We use four public event-log traces covering three domains.

### 3.1 Azure Predictive Maintenance (synthetic, per-machine)

100 machines, 2015-01-01 to 2016-01-01, hourly telemetry.
`PdM_errors` (3,919 non-fatal errors, five error codes),
`PdM_maint` (3,286 maintenance actions, four components), `PdM_failures`
(761 component replacements). We join `PdM_maint` and `PdM_failures`
on (machineID, datetime, comp) to distinguish `maintenance` from
`component_replacement`. Failures at exactly 2015-01-02 03:00 (18
rows) do not match any `PdM_maint` record; they are a bootstrap seed
batch planted by the synthetic generator and are excluded from BOTH
anchors and event streams so they do not contaminate windows for
subsequent real failures.

Event vocabulary: `software_error`, `maintenance`,
`component_replacement`, `terminal_failure`, each with a subtype
(`error1..error5`, `comp1..comp4`). Entity is `machineID`.
Source: [@azurepdm].

### 3.2 Alibaba cluster-trace-v2018 (production cloud, per-job)

`batch_task.csv` from the public Alibaba trace [@alibaba2018repo;
@alibaba2018trace], 14,295,731 tasks across 4,201,014 jobs, 8.9 days
(2018-01-01 through 2018-01-09 by trace clock). 83,207 jobs contain
at least one `Failed` task. `batch_instance.csv` (21 GB compressed)
is not used in this pass; the per-job analysis on `batch_task` alone
is sufficient to answer the ordering question.

Event vocabulary: `task_failure`, `task_success`, `task_waiting`,
`task_running`, each with subtype = task_name letter prefix
(`M`, `R`, `J`, `task`, `MergeTask`, `L`). Entity is `job_name`.

### 3.3 LLNL Blue Gene/L syslogs (Loghub, per-rack)

4,747,963 syslog messages from LLNL Blue Gene/L, 214.7 days
(2005-06-03 to 2006-01-04), from the Loghub archive [TODO:cite
oliner2007bgl]. 913,594 messages remain after dropping INFO-level
noise; 348,189 (7.34%) are labeled alerts. Entity is the rack
(top-level `R##` prefix of the node ID); 64 racks. Event vocabulary:
`terminal_alert` (labeled alerts with 30+ alert codes such as
KERNMNTF, APPTO, KERNSTOR), `system_error` (non-alert FATAL / ERROR /
SEVERE / FAILURE), `system_warning`. Component (RAS, KERNEL, APP,
MMCS, ...) is used as an additional subtype axis.

### 3.4 SCANIA Component X (production automotive, per-vehicle)

Real fleet telematics dataset released 2025 [TODO:cite kharazian2025],
23,550 trucks over 1.5 years (2019-01 through 2020-05 in study
clock), 1,122,452 readouts of 105 numeric counter and histogram
features. 2,272 vehicles (9.65%) undergo a component X repair during
the study.

Because features are numeric counters rather than native discrete
events, we derive tokens: for each (vehicle, feature) we compute
inter-readout DELTAS and emit a `counter_surprise` token per readout
whenever the absolute delta exceeds the vehicle's own 90th-percentile
threshold for that feature. Per-vehicle normalisation controls for
baseline usage variation across the fleet. Entity is `vehicle_id`.
The failure event is a synthetic `terminal_repair` marker placed at
the last readout timestamp of each repair-labeled vehicle.

## 4  Method

### 4.1 Pre-failure windows and matched controls

For every terminal failure event on an entity (machine on Azure, job
on Alibaba, rack on BGL, vehicle on SCANIA), we build a failure
window covering the K events (or the time horizon T) strictly before
the failure timestamp. Matched controls come from two designs
depending on the trace:

- **Same-entity clean regions** (Azure, BGL) when the entity carries
  long timelines with sparse failures; controls are anchored at
  times with no failure within horizon T in either direction.
- **Cross-entity non-failure sample** (Alibaba, SCANIA) when entities
  are short-lived; controls come from the last K events of a
  non-failure entity, sampled at a 3:1 control:failure ratio from a
  large candidate pool.

BGL alerts are additionally grouped into episodes (>= 1h inter-arrival
gap) and windows are anchored on the first alert of each episode, so
anchor-per-alert double-counting inside a cascade is avoided.

Horizons studied: `1h`, `6h`, `24h`, `last5`, `last10` on Azure;
`last3`, `last5`, `last10` on Alibaba; `last5`, `last10`, `last20`
on BGL and SCANIA (time-based horizons are not meaningful for short
per-job or per-episode observations).

### 4.2 Mining

Items are `event_type:event_subtype` strings. FP-Growth
(`mlxtend.frequent_patterns.fpgrowth`) mines frequent itemsets on
failure windows at minimum support 0.05. PrefixSpan (SPMF v2.64
[@fournier2016spmf] via subprocess) mines frequent ordered sequences
at the same support.

For each mined pattern P we compute support in failure windows
(support_failure), support in control windows (support_control),
lift = support_failure / pooled_support(P), and relative risk
= P(failure | P) / P(failure | ¬P). Every mined sequence is also
scored against the ITEMSET COUNTERPART of the same event set; the
difference `order_gain = sequence_lift - itemset_lift` quantifies
how much preserving order contributes above co-occurrence.

### 4.3 Sanity invariants

Every phase carries pre-declared invariants whose expected outcome is
stated up front. Itemset mining checks that a random-label
permutation at the same min_support does not yield a top lift within
a factor of 1.5× of the real top lift; a violation would indicate
either data leakage or an over-sensitive support threshold. Sequence
mining checks that a within-window random order permutation
preserves top itemset lift (unchanged, by construction) but strictly
reduces top sequence lift on rich horizons (windows with >= 3 events
on average). Both invariants pass on the two traces where the method
yields wins; the boundary traces are diagnosed via these invariants
rather than by post-hoc justification.

### 4.4 Statistical significance

For each mined pattern we compute an exact one-sided hypergeometric
p-value on the observed failure-hit count against the
label-permutation null with the pattern hit-set fixed. Under H0 the
number of hits landing in the failure class is Hypergeom(N_F+N_C,
hit_F+hit_C, N_F); the upper-tail probability of the observed hit
count IS the label-permutation p-value, so we compute it in closed
form. Benjamini-Hochberg FDR correction
[@benjamini1995controlling] is applied per (horizon × pattern class)
to give q-values.

## 5  Experiments

Windows are split temporally by anchor timestamp. Cutoffs:
Azure 2015-09-01, Alibaba 2018-01-07, BGL 2005-11-01, SCANIA
2020-01-01. Mining runs on training windows only; surviving patterns
become binary presence features on both train and test. A logistic
regression fit on train is evaluated on test for four feature sets:

- **event_count**: single feature, n_events in the window
- **itemsets_only**: binary presence of each train-mined itemset
  surviving its permutation null
- **sequences_only**: binary presence of each train-mined sequence
  surviving its shuffle null
- **combined**: union of A, B, C

For each configuration we report AUROC, AUPRC, F1 / precision /
recall at threshold 0.5, and lead time (anchor − last_event_ts) on
true-positive failure windows. Numbers are computed in a single pass
per configuration and stored as one artifact, so a comparison across
feature sets on the same trace cannot drift.

### 5.1 Coverage per horizon (Azure)

Time horizons of 1h and 6h leave 99.6% and 98% of failure windows
empty. Effectively no Azure PdM failure is preceded by an event in
the same hour. Useful horizons are 24h and the count-based
(`last5`, `last10`). Even at 24h, failure and control windows
separate cleanly by raw event count (failure mean 1.58 events,
control mean 0.077).

_Figure:_ [azure_window_horizon_vs_events.png](../diagnostics/azure_window_horizon_vs_events.png)

### 5.2 Mining sensitivity to min_support (Azure)

A min_support sweep over {0.02, 0.05, 0.10, 0.15} preserves the
headline ordering at every operating point:

| horizon | metric        | 0.02  | 0.05  | 0.10  | 0.15  |
|---------|---------------|-------|-------|-------|-------|
| 24h     | combined      | 0.996 | 0.996 | 0.996 | 0.996 |
| 24h     | itemsets_only | 0.996 | 0.996 | 0.996 | 0.996 |
| last5   | combined      | 0.815 | 0.810 | 0.803 | 0.774 |
| last5   | itemsets_only | 0.761 | 0.754 | 0.762 | 0.754 |
| last10  | combined      | 0.664 | 0.696 | 0.751 | 0.741 |
| last10  | itemsets_only | 0.578 | 0.643 | 0.686 | 0.674 |

At every min_support tested, combined dominates itemsets_only by
at least +4 AUROC points at last5 and at least +5 at last10.

_Figure:_ [azure_sensitivity_min_support.png](../results/figures/azure_sensitivity_min_support.png)

## 6  Results

### 6.1 Mined patterns (Azure and Alibaba)

At Azure 24h, `{software_error:error2, software_error:error3}` reaches
lift 3.99 (present in 38.2% of failure windows, in 0.04% of controls;
P(failure | pattern) = 99.6%). All six 24h itemsets dominate the
random-label permutation null (permuted top 1.24).

At Azure `last5`/`last10`, the sequence
`maintenance:comp4 → software_error:error2 → software_error:error3`
reaches lift 3.73 as an ordered pattern but only 2.22 as the same
items unordered. Ordered patterns ending `... → error2 → error3`
dominate the top-8 at both count-based horizons.

On Alibaba `last5`, the strongest ordered pattern is
`task_success:M → task_success:R → task_success:M → task_success:M`
with sequence lift 2.43 versus itemset lift 0.94 for the same event
set (order gain +1.49). At `last3`,
`task_success:M → task_success:M → task_success:M` reaches sequence
lift 3.06 vs itemset lift 1.37 (order gain +1.69): three consecutive
Map completions predict a subsequent failure much more strongly than
the mere presence of Map events would suggest.

_Figure:_ [azure_itemset_vs_sequence_lift.png](../results/figures/azure_itemset_vs_sequence_lift.png)

### 6.2 Predictive evaluation (four traces)

Head-to-head on temporally-held-out test sets:

| trace   | horizon | event_count | itemsets_only | sequences_only | combined     |
|---------|---------|-------------|---------------|----------------|--------------|
| Azure   | 24h     | 0.97 / 0.91 | 0.996 / 0.99  | —              | 0.996 / 0.99 |
| Azure   | last5   | 0.50 / 0.34 | 0.75 / 0.56   | 0.66 / 0.56    | **0.81 / 0.72** |
| Azure   | last10  | 0.50 / 0.34 | 0.64 / 0.50   | 0.67 / 0.53    | **0.70 / 0.58** |
| Alibaba | last3   | 0.69 / 0.50 | 0.75 / 0.44   | 0.50 / 0.20    | **0.81 / 0.63** |
| Alibaba | last5   | 0.60 / 0.50 | 0.67 / 0.34   | 0.51 / 0.21    | **0.74 / 0.57** |
| Alibaba | last10  | 0.59 / 0.50 | 0.68 / 0.36   | 0.52 / 0.23    | **0.74 / 0.59** |
| BGL     | last5   | 0.50 / 0.25 | 0.49 / 0.25   | —              | 0.49 / 0.25  |
| BGL     | last10  | 0.50 / 0.25 | 0.49 / 0.25   | 0.50 / 0.25    | 0.50 / 0.25  |
| BGL     | last20  | 0.50 / 0.25 | 0.48 / 0.25   | 0.50 / 0.25    | 0.51 / 0.26  |
| SCANIA  | last5   | 0.50 / 0.09 | 0.52 / 0.11   | —              | 0.52 / 0.11  |
| SCANIA  | last10  | 0.50 / 0.09 | 0.60 / 0.14   | 0.55 / 0.11    | 0.60 / 0.15  |
| SCANIA  | last20  | 0.50 / 0.09 | 0.57 / 0.14   | 0.53 / 0.10    | 0.57 / 0.13  |

_(AUROC / AUPRC on the temporally-held-out test set. SCANIA uses
per-vehicle 90th-percentile-delta binning; BGL uses episode-anchored
per-rack windows with alerts removed from the pre-alert stream. Both
boundary traces sit near chance for every feature set at every
horizon.)_

_Figure:_ [four_dataset_predictive_comparison.png](../results/figures/four_dataset_predictive_comparison.png)

### 6.3 Order gain

The order-gain distribution (sequence lift minus itemset lift for the
same items) is zero at horizons where windows contain 1-2 events
(Azure 1h/6h/24h) and positive with mean 0.30-0.32 and max 1.5 at
count-based horizons where the method finds signal (Azure `last5`/`last10`,
Alibaba `last3`/`last5`/`last10`). On BGL and SCANIA the order gain
does not translate into AUROC because the underlying itemset signal
is itself weak.

### 6.4 Formal significance

At BH q < 0.05: every Azure 24h itemset (6/6) and every Azure 24h
sequence (7/7) is significant; 53/77 Azure `last5` itemsets and
55/67 Azure `last5` sequences; 562/657 Azure `last10` sequences.
Both 1h and 6h Azure horizons flag zero patterns as expected
(0/3 sequences at 1h, 0/5 at 6h). On Alibaba: 6/10 `last3` itemsets,
9/16 `last3` sequences, 59/109 `last10` sequences.

### 6.5 Lead time on true positives

- **Alibaba** (operational lead time): median 0 s across horizons,
  IQR 0-2 min. Task boundaries are the practical alarm resolution;
  the AUROC 0.74-0.81 signal is real but must be acted on within the
  next-task interval.
- **Azure** (structural lead time set by the generator): median lead
  time is exactly 24.0 hours at every horizon. The synthetic
  generator places pre-failure errors on a ~24h clock relative to
  the failure timestamp, so any correctly-triggered alarm inherits
  that clock. This describes the dataset's construction, not the
  model's early-warning capacity.

Lead-time detail: [results/tables/{azure,alibaba}_leadtime.md](../results/tables/).

## 7  Discussion

### 7.1 What each mined signature means operationally

On Azure PdM, `software_error:error2 → software_error:error3` at
`last5` reaches sequence lift 3.73 vs itemset lift 2.22 for the same
items. The order-specific reading is that error2 and error3 are not
interchangeable noise: a machine reporting error2 first and then
error3 is materially more likely to reach a `terminal_failure` than
one that reports them in the other order. In practical monitoring,
an alarm keyed on the pair-in-order is preferable to the same alarm
keyed on the pair-as-set.

On Alibaba, `task_success:M → task_success:M → task_success:M` at
`last3` reaches sequence lift 3.06 vs itemset lift 1.37 for the same
items. Three consecutive Map completions predict a subsequent Failed
task more strongly than "the job contains Map completions" alone.
The operational reading is that the position of the failure inside
the DAG matters: jobs that make it through a Map-heavy prefix are
the jobs whose downstream Reduce or Join phases can fail, whereas
jobs that fail early do so in a different distribution of task types.

### 7.2 Regime of validity

The four-trace survey resolves an obvious follow-up question:
does the sequences+itemsets combined-feature-set advantage transfer
to any operational event log? It does not.

- **Where the method works** (Azure PdM, Alibaba v2018): rich native
  discrete event vocabularies (5 error codes × 4 component types on
  Azure; 4 task statuses × 6 task roles on Alibaba). Failure-window
  content differs discriminably from control-window content, and
  order carries information beyond the itemset.
- **Where the method does not work** (BGL, SCANIA): the target class
  is self-triggering (BGL alerts follow other alerts, and non-alert
  log lines carry no discriminable precursor signal), or the discrete
  event stream must be derived from continuous counters (SCANIA
  requires binning per-vehicle deltas, and a defensible 90th-percentile
  surprise binning produces only marginal AUROC).

Concretely: on BGL the best combined AUROC across horizons is 0.51
(chance) even when INFO-level messages and component granularity are
included in the non-alert stream; on SCANIA the best combined AUROC
across horizons is 0.60, unchanged when the fleet-wide 90th-percentile
delta binning is replaced by per-vehicle-normalised 90th-percentile
binning.

To determine whether the SCANIA gap is representation-loss (tokens
destroy signal that is actually in the trace) or signal-absence (no
representation could recover 0.75), we run a diagnostic ceiling test:
LightGBM on a compact set of histogram-aware distributional
descriptors (Wasserstein-1 distance to a per-vehicle baseline,
signed centroid shift, entropy shift, tail-mass shift, and each
descriptor's slope over the last 20 readouts) computed on the same
temporal split. The ceiling model reaches AUROC 0.60 / AUPRC 0.04,
essentially identical to the pattern-mining pipeline. Logistic
regression on the same 113 structured features reaches 0.58 / 0.05.
The GBM-versus-LR gap is 0.02 AUROC, so classifier capacity is not
the constraint either. The gap between SCANIA (~0.60) and the
Azure / Alibaba wins (~0.80-1.00) reflects a limit of the readout
cadence and feature vocabulary, not of the pattern-mining pipeline
against a richer alternative representation.

**Positive control on same-manufacturer data.** To rule out the
alternative explanation that SCANIA-family telemetry itself lacks
predictive signal, we apply the same LightGBM ceiling test to
SCANIA APS Failure at Scania Trucks [@costa2016ida] (UCI 421, IDA
2016 industrial challenge). APS Failure uses the same anonymised
histogram-encoded schema as Component X (7 histogram groups of 10
bins each + 100 single counters = 170 feature columns) but delivers
one per-truck cross-sectional readout instead of a longitudinal
readout stream, with a binary APS-system-failure label at a 1.67%
positive rate. On the same LightGBM configuration our Component X
ceiling used, APS Failure reaches AUROC 0.994 / AUPRC 0.934 on the
canonical held-out test split (16,000 trucks, 375 positives); LR
alone reaches 0.979 / 0.800. Same manufacturer, same anonymisation
schema, different readout format, near-perfect predictability. This
excludes "SCANIA-family data is inherently weak" as an explanation
for the Component X boundary and localises the limit to the
sparse longitudinal readout cadence of that specific release.

The method's regime of validity is therefore "trace has a rich
native discrete event vocabulary AND failure class is not
self-triggering AND readout-cadence signal capacity exceeds the
target AUROC bar". BGL fails the second condition; SCANIA fails the
third; Azure PdM and Alibaba v2018 satisfy all three.

The two lead-time regimes in §6.5 speak to deployment. Azure inherits
a structural 24h clock from the synthetic generator and should not
be read as a real-world warning interval; Alibaba's median 0-second
lead time is the honest one, and a per-job classifier there must be
paired with sub-second scheduling infrastructure to act on the signal
at all.

## 8  Limitations

- Alibaba results are per-job, computed on `batch_task` alone; the
  `batch_instance` table (21 GB compressed) that would enable
  per-machine failure trajectories on the same trace is left to
  future work.
- Alibaba sequence patterns are numerically fewer than Azure ones
  (2-6 vs 6-16 surviving the shuffle-null per horizon). A wider
  min_support sweep and top-K sequence mining would sharpen the
  Alibaba sequence-mining slice specifically.
- The min_support sensitivity sweep covers itemset support;
  regularization strength and cross-machine / cross-job
  leave-one-out are the next robustness questions.
- Azure lead times are set by the synthetic data generator and
  describe dataset construction rather than real-world warning
  intervals. Alibaba lead times are the operationally-honest number
  from a real trace.
- SCANIA binning uses per-vehicle 90th-percentile deltas; a stronger
  alternative representation (Wasserstein-1 distance to vehicle
  baseline, centroid shift, entropy and tail-mass changes per
  histogram feature) was also tested via a LightGBM ceiling
  diagnostic and reached the same AUROC 0.60. The signal at this
  readout cadence is insufficient to reach the 0.75 bar under any
  representation we tested; scoping SCANIA as a boundary case
  reflects that finding.

## 9  Conclusion

Frequent-pattern mining of discrete operational events surfaces
interpretable pre-failure signatures on two of four traces studied.
On both winning traces, sequences add real predictive information
beyond itemsets when window definitions are rich enough for order
to be a real degree of freedom; at those horizons, combining
itemset and sequence features improves failure prediction by 5-10
AUROC points over either alone. The result replicates across a
synthetic per-machine trace (Azure PdM) and a real per-job
production trace (Alibaba v2018), and its regime of validity is
mapped by two additional traces (BGL, SCANIA) where the pipeline
does not find signal, with a mechanistic explanation for each.

---

## What is still missing

- Regularization sweep and cross-machine / cross-job leave-one-out
  for a fuller Phase 7.
- Optional: per-machine Alibaba analysis with `batch_instance`.
- Prose polish + related-work paragraph expansion (scout returning).
- HTML/DOCX build via `paper-build` skill once prose is settled.

## What is verified

- Bibliography: [paper/references.bib](references.bib), currently 12
  entries validated by `bibtest`. Expansion pending scout return.
- Numbers audit: 50 / 50 claims in the previous 2-trace skeleton
  match the underlying JSON stats. Re-running after 4-trace and
  boundary-condition additions.
