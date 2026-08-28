# Reviewer pass 1: paper/skeleton.md

**Recommendation:** Major revision.

## Summary of contribution (as claimed)

The paper mines pre-failure event windows from discrete operational logs
with FP-Growth (itemsets) and PrefixSpan (sequences), scores every
pattern against matched controls, and tests four feature sets on a
temporally-held-out split. The central claim is that temporal order in
mined sequences contributes real predictive information beyond unordered
co-occurrence, and that the finding replicates across a small synthetic
per-machine trace (Azure PdM) and a real per-job production trace
(Alibaba v2018).

## Strengths

- **S1.** Every quantitative claim in the prose is programmatically
  verified against the parquet/JSON stats that produced it
  (paper/numbers_audit.md, 44/44 pass). No hand-typed numbers.
- **S2.** Every phase carries a pre-declared sanity invariant
  (random-label permutation for itemsets; within-window order shuffle
  for sequences). Both fire and both pass; §4.3 states them up front.
- **S3.** Cross-dataset replication is genuine, not a variant of the
  same benchmark: Azure PdM is per-machine, Alibaba is per-job; the
  finding replicates with different entities and different event
  vocabularies.
- **S4.** Bibliography validated by bibtest (10/12 resolved to the
  intended paper; 2 grey-literature dataset URLs).
- **S5.** Wins-only scanner and tone audit both clean.

## Blocking issues (must fix before this leaves skeleton state)

### W1. §7 Limitations still says "No lead-time metric reported yet"

SC-4 violation. §5.6 IS the lead-time section, added later. The
Limitations bullet is stale from the previous draft. Remove or replace
with a genuine boundary condition about lead-time (e.g. that Azure lead
times cluster at exactly 24h because the synthetic generator plants
errors ~1 day before failure, so Azure lead-time numbers describe the
data structure, not model performance).

### W2. §6 Discussion is a TODO placeholder

Empty section between Results and Limitations. Either fill it or drop
the heading. A discussion section that shows only a TODO leaves the
paper without an interpretation layer between the numbers and the
takeaway.

### W3. §2 Related Work is a TODO stub

The narrative paragraph and the "TODO: expand" line together read as a
partially-written section. Convert the four papers already cited in
prose into proper cite calls into `paper/references.bib`, and either
drop the TODO line or scope it as a specific missing citation.

### W4. Abstract claims "same shape appears on Alibaba batch_task
status sequences" but no concrete Alibaba sequence is shown in the body

§5.2 shows one concrete Azure pattern (`{error2, error3}` and
`maintenance:comp4 → error2 → error3`), but there is no matching
example for Alibaba. Either add one or two representative Alibaba
sequences to §5.2 (a "task_success:M → task_success:R → task_failure:R"
style), or soften the abstract claim to what the body actually shows
(e.g. "the same +5-10 AUROC combined-over-itemset gap holds on Alibaba,
Table N"). SC-3 abstract-body parity.

## Substantive issues

### W5. Abstract impact beat is generic

Sentence 6 ("Temporal order in operational logs is a real, transferable
signal for early failure warning") states the finding but does not
name a pointer, a recommended practice, or a released artifact. If
there is a code / patterns / dataset artifact to point at, the abstract
should name it; otherwise the impact beat is doing weak work at ~18
words that could carry a concrete pointer.

### W6. §5.3 sequences-only precision claim needs a footnote

"Sequences alone fire rarely but with >= 0.95 precision at threshold
0.5" is a strong claim; but at that precision the recall is very low
(0.02-0.36 per §5.3 table's recall column, which is not in the printed
table). Add the recall range next to the precision so the reader knows
the operating regime.

### W7. §5.6 Azure lead-time = 24.0 hours is a data-generator artifact

Currently phrased as a lead-time result; the second sentence
("the synthetic generator seeds pre-failure errors approximately one
day before the failure") already qualifies it. The Alibaba lead-time
result is a substantive property of production data (0-2 min); the
Azure one is not. Consider labelling them differently (e.g. "structural
lead time set by the generator" vs "operational lead time in
production") or moving the Azure lead-time to a footnote so the
paper's actual lead-time claim rests on Alibaba.

## Minor issues

| § | Issue | Severity |
|---|-------|----------|
| Abstract | "5-10 AUROC points" is a range across horizons; state the specific number for the headline horizon (e.g. "+5.6 on Azure last5, +6.5 on Alibaba last3") for the abstract's single-claim slot | minor |
| §4.3 | "1.5x" should render as "1.5×" in typeset form | minor |
| §5.1 | "1.58 events, control mean 0.077" — good, but a sentence-level effect-size claim (e.g. Cohen's d) would strengthen it | minor |
| §5.5 | The mixed formatting "6/6" vs "53/77" vs "562/657" is fine but a small table would let the reader compare significance counts across horizons | minor |

## Questions for authors

- **Q1.** The Alibaba `batch_instance` file is 21 GB compressed. What
  would you expect to see per-machine that per-job cannot show? Would
  the per-machine analysis change the headline claim, or only add a
  third replication?
- **Q2.** §5.4 says sequences add nothing at Azure 24h "because no
  sequence beat its own shuffle-null". Is that a horizon-length issue
  (windows are 1-2 events, no room for order) or a signal issue (the
  order information is genuinely absent)? A short paragraph in §5.4
  distinguishing the two would help.

## Recommendation

Address W1-W4 before the skeleton leaves this state; W1 is a two-line
fix, W2 is a paragraph, W3 is prose expansion into the validated bib,
W4 is either a body addition or an abstract softening. W5-W7 are
substantive and would strengthen the draft but do not need to block
this pass. After W1-W4 land, re-run this reviewer pass on the revised
draft; if clean, run bibtest one more time and hand to a human
co-reader.
