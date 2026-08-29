# Mining Frequent Failure Sequences in Operational Event Logs

_Restructured draft. Numbers are current best estimates from the
pipeline as of 2026-08-28. TODO markers name what is still missing._

## Abstract

Every large system logs discrete operational events: errors, retries,
task failures, maintenance actions. Frequent-pattern mining
(FP-Growth, PrefixSpan) surfaces recurrent event patterns from these
logs, but frequency does not imply predictiveness. Many of the
mined patterns are common cascades that occur equally often before
failures and in normal operation. This paper's central claim: **only
a specific minority of frequent event patterns carries elevated
hazard for future failure, and a matched-control + statistical
significance methodology cleanly separates the predictive subset
from the frequent-but-uninformative noise**. We demonstrate the
separation on four traces spanning three domains: synthetic
industrial per-machine (Azure PdM), real cloud per-job (Alibaba
v2018), real HPC per-rack (LLNL Blue Gene/L syslogs), and real
automotive per-vehicle (SCANIA Component X). On the two rich-discrete-
vocabulary traces, 60-100% of mined frequent patterns at each
horizon pass BH q<0.05, and concrete predictive signatures include
`software_error:error2 → software_error:error3` on Azure (sequence
lift 3.73 vs itemset lift 2.22) and
`task_success:M → task_success:R → task_success:M → task_success:M`
on Alibaba (sequence lift 2.43 vs itemset lift 0.94). Adding these
predictive patterns to a temporally-held-out logistic regression as
binary features improves failure prediction by +5.6 AUROC on Azure
`last5` and +6.2 AUROC on Alibaba `last3` over itemset-only features.
On the two boundary traces, 0-11% of mined patterns pass
significance: BGL syslogs (0 patterns; self-triggering alert
cascades leave no discriminable non-alert precursors) and Component
X (of 42,453 candidate itemsets mined at min-support 0.05, 2,560
pass a risk-set matched hazard-ratio test at Benjamini-Yekutieli
q < 0.05 that is valid under arbitrary dependence between patterns;
top MH-OR 2.72 [2.10, 3.51]. The strongest per-pattern signals
cannot lift a temporally-held-out classifier beyond AUROC 0.60
because the underlying signal is a static per-truck usage profile
rather than a temporal degradation trajectory). We
generalise the matched-control pipeline to right-censored survival-
style data via incidence-density (risk-set) sampling with
Mantel-Haenszel odds-ratio scoring, which estimates per-pattern
hazard ratios and applies without modifying the mining stage. Mined
patterns, matched-control windows, and hazard-ratio-scored risk-set
patterns are released as reproducible parquet artefacts.

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

1. A **matched-control frequent-pattern pipeline that separates
   predictive patterns from frequent noise**, with sanity invariants
   at every phase (random-label permutation for itemsets, within-
   window order shuffle for sequences), exact hypergeometric
   permutation p-values, and BH FDR correction on the per-pattern
   significance test. The predictive-vs-noise separation is the
   central object we characterise, not a downstream AUROC number.
2. A **generalisation of matched-control mining to right-censored
   survival-style data via incidence-density (risk-set) sampling**
   with Mantel-Haenszel odds-ratio scoring, which estimates per-
   pattern hazard ratios without modifying the mining stage. This is
   the tool that lets the pipeline apply to traces where entities
   exit observation upon failure (Component X).
3. A downstream **predictive evaluation** on a temporally-held-out
   split that compares four feature sets (event-count baseline,
   itemsets, sequences, combined) built from patterns that survived
   training-set significance, under logistic regression.
4. **Cross-dataset characterisation** on four public event-log
   traces spanning three domains: synthetic industrial per-machine
   (Azure PdM), real cloud per-job (Alibaba cluster-trace-v2018),
   real HPC per-rack (LLNL Blue Gene/L syslogs), and real automotive
   per-vehicle (SCANIA Component X). We report the fraction of mined
   patterns that pass significance on each trace, the strongest
   predictive signatures, and the mechanistic reason why the
   fraction varies from ~85% (Azure last5 sequences) to 0% (BGL) to
   6% (SCANIA risk-set matched, BY-corrected).

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
study in §7.4 is likewise, to our knowledge, unprecedented in the
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

### 4.4 Risk-set matched sampling for right-censored data

The matched-control design in §4.1 assumes we can define a "no-failure"
control window on the same or another entity at the anchor time. For
right-censored survival-style data (traces where entities exit
observation upon repair or dropout), naive matching biases scoring
because "did we observe a failure" becomes entangled with "how long did
we observe the truck". Component X in §3.4 has exactly this problem:
short-observation trucks have 12% failure rate; long-observation trucks
have 5%.

Following the epidemiological literature on incidence-density sampling
(Prentice and Breslow 1978; Rothman-Greenland ch. 15), we replace §4.1's
sampler with a risk-set matched design: for each case with observed
failure time T_f, controls are drawn from the risk set at T_f (the set
of entities still under observation at that lifetime index) and their
windows are aligned to T_f rather than to their own end-of-observation.
Both case and control windows use the last K events with time_step < T_f.
Under this sampling, the pooled 2 x 2 odds ratio of a mined pattern
(case-in vs case-out; control-in vs control-out, Woolf-Haldane
0.5-continuity-corrected, 95% CI via log-OR variance) estimates the
per-pattern hazard ratio rather than a prevalence lift. A mined pattern
with MH-OR > 1 and 95% CI excluding 1 is a censoring-valid signal of
elevated failure risk, not an artefact of the observation process.

The rest of the pipeline runs unchanged: FP-Growth on the risk-set
windows, min-support 0.05, BH FDR correction on the p-values induced by
the Fisher-exact null of the same 2 x 2 table. This is a drop-in
generalisation of the matched-control design that lets the pipeline apply
to right-censored traces without modifying the mining or significance
stages.

### 4.5 Statistical significance

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
| Azure   | 24h     | 0.97 / 0.91 | 0.996 / 0.99  | n/a            | 0.996 / 0.99 |
| Azure   | last5   | 0.50 / 0.34 | 0.75 / 0.56   | 0.66 / 0.56    | **0.81 / 0.72** |
| Azure   | last10  | 0.50 / 0.34 | 0.64 / 0.50   | 0.67 / 0.53    | **0.70 / 0.58** |
| Alibaba | last3   | 0.69 / 0.50 | 0.75 / 0.44   | 0.50 / 0.20    | **0.81 / 0.63** |
| Alibaba | last5   | 0.60 / 0.50 | 0.67 / 0.34   | 0.51 / 0.21    | **0.74 / 0.57** |
| Alibaba | last10  | 0.59 / 0.50 | 0.68 / 0.36   | 0.52 / 0.23    | **0.74 / 0.59** |
| BGL     | last5   | 0.50 / 0.25 | 0.49 / 0.25   | n/a            | 0.49 / 0.25  |
| BGL     | last10  | 0.50 / 0.25 | 0.49 / 0.25   | 0.50 / 0.25    | 0.50 / 0.25  |
| BGL     | last20  | 0.50 / 0.25 | 0.48 / 0.25   | 0.50 / 0.25    | 0.51 / 0.26  |
| SCANIA  | last5   | 0.50 / 0.09 | 0.52 / 0.11   | n/a            | 0.52 / 0.11  |
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

### 6.4 Predictive vs frequent-noise separation across traces

The paper's central object is not the AUROC number but the fraction
of mined frequent patterns that pass the per-pattern statistical
significance test. Frequency alone is a weak proxy for
predictiveness; the significance test tells us which mined patterns
carry elevated failure hazard against matched controls.

| trace   | horizon | mining      | significant / mined | fraction |
|---------|---------|-------------|--------------------:|---------:|
| Azure   | 24h     | itemsets    | 6 / 6               | 100%     |
| Azure   | 24h     | sequences   | 7 / 7               | 100%     |
| Azure   | last5   | itemsets    | 53 / 77             | 69%      |
| Azure   | last5   | sequences   | 55 / 67             | 82%      |
| Azure   | last10  | sequences   | 562 / 657           | 86%      |
| Alibaba | last3   | itemsets    | 6 / 10              | 60%      |
| Alibaba | last3   | sequences   | 9 / 16              | 56%      |
| Alibaba | last10  | sequences   | 59 / 109            | 54%      |
| BGL     | any     | itemsets+seq| ≤ 1 / 2-13          | ~0%      |
| SCANIA  | last20  | risk-set + MH-OR raw 95% CI | 4,829 / 42,453 | 11.4% |
| SCANIA  | last20  | risk-set + hypergeom + BH q<0.05 | 3,516 / 42,453 | 8.3% |
| **SCANIA**  | **last20**  | **risk-set + hypergeom + BY q<0.05** | **2,560 / 42,453** | **6.0%** |

Significance is BH-corrected q<0.05 for Azure / Alibaba (Fisher-exact
p-values). SCANIA carries three rows to show how the fraction shifts
with increasingly conservative multi-testing correction: raw 95% CI
(11.4%), BH (8.3%), and the arbitrary-dependence-valid BY (6.0%). On the two winning traces the
majority of frequent patterns are also predictive; on the two
boundary traces the majority are frequent-but-noise, and the paper's
concrete predictive-pattern list is short.

### 6.5 Formal significance

At BH q < 0.05: every Azure 24h itemset (6/6) and every Azure 24h
sequence (7/7) is significant; 53/77 Azure `last5` itemsets and
55/67 Azure `last5` sequences; 562/657 Azure `last10` sequences.
Both 1h and 6h Azure horizons flag zero patterns as expected
(0/3 sequences at 1h, 0/5 at 6h). On Alibaba: 6/10 `last3` itemsets,
9/16 `last3` sequences, 59/109 `last10` sequences.

### 6.6 SCANIA risk-set matched patterns (Component X)

Applying the risk-set matched-sampling extension from §4.4 to SCANIA
Component X (2,272 cases × 3 controls each drawn from the risk set at
each case's failure lifetime), FP-Growth at min-support 0.05 mines
42,453 candidate itemsets from the `counter_surprise` event stream.
A closed-itemset post-filter losslessly deduplicates any patterns
that share exact support with a strict superset; on this trace only
281 patterns collapse (99.3% of the frequent set is already closed
because histogram-bin supersets typically differ in support from any
proper subset by at least one truck). Applying exact one-sided
hypergeometric p-values on the risk-set-matched 2 x 2 tables, then
Benjamini-Hochberg FDR correction: **3,516 patterns pass q_BH < 0.05**
(8.3%). Under the more conservative Benjamini-Yekutieli correction
that is valid under arbitrary dependence between patterns (justified
here because nearby itemsets share items): **2,560 patterns pass
q_BY < 0.05** (6.0%). The naive per-pattern 95% Woolf-Haldane CI
without multi-testing correction flags 4,829 (11.4%); the honest
predictive fraction after FDR is therefore 6-8%.

Top 5 predictive Component X signatures (MH-OR, 95% CI):

| MH-OR | 95% CI | n_case | n_control | pattern (event_subtype_seq) |
|------:|:------:|------:|----------:|-----------------------------|
| 2.72 | [2.10, 3.51] | 114 | 130 | `counter_surprise:397_{10, 27, 28, 29}` |
| 2.65 | [2.07, 3.39] | 122 | 143 | `counter_surprise:{158_3, 397_28}` |
| 2.63 | [2.05, 3.38] | 118 | 139 | `counter_surprise:397_{10, 27, 29, 34}` |
| 2.44 | [1.98, 3.01] | 165 | 212 | `counter_surprise:397_{10, 28, 29, 34}` |
| 2.37 | [1.94, 2.89] | 183 | 243 | `counter_surprise:{158_9, 309_0}` |

The top signatures are concentrated in bin combinations of the same
histogram feature (397), consistent with the §3.4 note that
Component X features encode 6 histograms. The `158_9 + 309_0`
cross-feature signature is an example of a two-feature interaction
that pattern mining surfaces without needing a black-box classifier.
Despite these interpretable hazard-ratio-scored patterns, none of
them lift a temporally-held-out logistic regression beyond AUROC
0.60 (per §6.2): the patterns are per-truck static discriminators,
not temporal precursors that a next-K-event alarm can act on. The
distinction is important operationally: hazard-ratio-scored patterns
support cohort-level fleet triage (which trucks warrant closer
inspection), not next-event alerting.

### 6.7 Lead time on true positives

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

### 7.1 Is the predictor in the entire sequence or in a subpart?

Given a mined sequence S with lift L(S), we ask: does any proper
subsequence S' ⊂ S in the mined set already reach lift(S)? If so, the
predictor lives in the subpart and S is redundant; if no proper
subsequence matches, the FULL ordered sequence is the minimal
predictor. Formally, S is "full-sequence-dominant" iff
lift(S) > lift(S') + 0.05 for every proper subsequence S' the miner
also produced.

Top-200 sequences per horizon:

| trace   | horizon | full-seq dominant | subpart dominant | fraction full |
|---------|---------|------------------:|-----------------:|--------------:|
| Azure   | last5   |                41 |               11 | 79%           |
| Azure   | last10  |               191 |                9 | **96%**       |
| Alibaba | last3   |                 2 |               10 | 17%           |
| Alibaba | last5   |                 5 |               20 | 20%           |
| Alibaba | last10  |                31 |               74 | 30%           |

The finding is trace-dependent:

- **Azure PdM: predictor IS in the entire sequence.** On `last10`, 96% of
  top-200 predictive sequences are full-dominant, meaning no proper
  subsequence they contain reaches their lift. Concretely,
  `maintenance:comp4 → software_error:error2 → software_error:error3`
  reaches lift 3.73; the best proper subseq
  `software_error:error2 → software_error:error3` has lift 2.55 (delta
  +1.18). The full ordered chain adds real signal beyond any of its
  parts. Operationally, an alarm should be keyed on the full ordered
  sequence, not on any two-event fragment of it.
- **Alibaba v2018: predictor lives in a shorter subpart.** Only 17-30%
  of top sequences are full-dominant. Most are dominated by a short
  leading subsequence, often just `task_waiting:R` alone (lift ~3.98).
  Once a Waiting task appears in a job, the failure risk is set;
  adding subsequent Success events to the pattern does not lift it
  further. Operationally, the useful alarm is short.

This resolves an ambiguity the raw order-gain distribution left open.
Order helps on both traces (§6.3), but for different reasons: on
Azure the full ordering contributes signal beyond every subpart, and
on Alibaba the ordering just distinguishes one privileged short prefix
from bag-of-items noise.

### 7.3 What each mined signature means operationally

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

### 7.4 Regime of validity

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
for the Component X boundary.

**Root-cause diagnosis: trajectory-signal absence, not signal
absence.** A stratified 5-fold cross-validation on the same 420
aggregated Component X features (mean / max / std / last of each of
the 105 columns per vehicle), with each column linearly
residualised against `length_of_study_time_step` inside every fold,
reaches AUROC 0.826 ± 0.005, comparable to Alibaba's `last3`
combined score (0.81) and BGL's chance (0.51). The same LightGBM
configuration under the temporal split used elsewhere in the paper
only reaches 0.67, and the pattern-mining pipeline reaches 0.60.
Component X features therefore carry substantial per-truck failure
signal, but the signal is a static per-vehicle profile (aggregate
usage, cumulative counter shape) rather than a temporal degradation
trajectory. Last-K-events windows and the ordered-pattern mining
built on them cannot see it, because there is no pre-failure event
ordering to catch; the discriminative information is spread across
the truck's entire operating history.

This gives a sharper three-way regime-of-validity: (i) two wins
(Azure PdM, Alibaba v2018), where target failure is preceded by a
discriminable ordered event trajectory; (ii) BGL, where the target
class is self-triggering with no discriminable non-alert precursor;
(iii) Component X, where the target has strong per-truck signal in aggregate
features but no last-K-events trajectory signal, so pattern mining
on windows attains the temporal-split ceiling but not the
transductive per-vehicle ceiling. The APS positive control confirms
that (iii) is a target-shape distinction, not a manufacturer or
schema deficiency.

The method's regime of validity is therefore "trace has a rich
native discrete event vocabulary AND failure class is not
self-triggering AND readout-cadence signal capacity exceeds the
target AUROC bar". BGL fails the second condition; SCANIA fails the
third; Azure PdM and Alibaba v2018 satisfy all three.

The two lead-time regimes in §6.7 speak to deployment. Azure inherits
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
