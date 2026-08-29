Markdown
# DAMI review

**Recommendation: Reject**

## Summary of contribution

The paper studies whether frequent itemsets and frequent sequential patterns mined from operational event logs are genuinely associated with subsequent failures, rather than merely frequent. It proposes a matched-control pipeline combining FP-Growth/PrefixSpan with per-pattern significance testing, extends the design to right-censored SCANIA data using risk-set sampling, and evaluates pattern-derived features on four operational traces. The empirical results are heterogeneous: Azure and Alibaba show useful pattern-based prediction and apparent sequence gains, while BGL and SCANIA are presented as boundary cases. The paper is squarely within the scope of *Data Mining and Knowledge Discovery*, but in its present form the central statistical inference, the censoring-aware estimator, the claimed novelty, and the “regime-of-validity” conclusion are not sufficiently sound for DAMI.

## Strengths

**S1. The paper asks a worthwhile knowledge-discovery question rather than merely reporting another classifier.** The distinction between “frequent” and “predictively discriminative” patterns is practically important, and the attempt to characterize which patterns survive comparison with controls is more interesting than simply running FP-Growth or PrefixSpan and listing high-support outputs. The paper explicitly makes this distinction its central object. 

**S2. The authors make a serious effort to include negative results.** BGL and SCANIA are not hidden when the proposed representation performs poorly. The SCANIA investigation in particular goes well beyond reporting AUROC 0.60: the draft tests histogram-aware representations, LightGBM, aggregate per-truck features, and an APS positive control. This is scientifically useful material. 

**S3. Temporally held-out downstream evaluation is substantially better than evaluating mined patterns only on the discovery sample.** Patterns are mined on training windows and then used as features on a later test period.  This means that some of the predictive findings, especially the held-out AUROC/AUPRC results, may survive even if the manuscript's per-pattern significance analysis must be replaced.

**S4. The comparison of itemsets and ordered sequences is potentially useful.** The Azure results suggest that a combination of ordered and unordered representations can improve held-out prediction, and the min-support sensitivity experiment is a useful robustness check. 

**S5. The authors have identified an important censoring issue in Component X.** Recognizing that controls must be sampled from the risk set rather than from arbitrary non-failure vehicles is correct in spirit. The underlying idea of aligning controls at each case failure time is appropriate for incidence-density/nested-case-control sampling. 

**S6. The manuscript is unusually transparent about its current limitations.** It explicitly acknowledges the synthetic nature of Azure lead time, the near-zero Alibaba lead time, incomplete robustness experiments, and ongoing bibliography/numerical auditing.  That transparency is appreciated, even though some of those items must be completed before journal submission.

---

## Blocking issues

### W1. The central “predictive fraction” significance test suffers from selection on the same labels being tested.

**WHY this blocks:** This is the most serious problem because the manuscript states that its central scientific object is the fraction of mined patterns surviving significance testing.  However, §4.2 mines patterns **from the failure windows** at `min_support = 0.05`, and §4.5 subsequently tests those selected patterns for enrichment in failure windows while treating each pattern's hit set as fixed.  

That conditioning is not innocuous. A pattern enters the candidate set precisely because it happened to have sufficient support among the cases. Under the null, the set of patterns that would have passed the mining threshold would change if the case/control labels changed. Therefore, an ordinary hypergeometric/Fisher p-value calculated after label-dependent candidate selection is not automatically a valid post-selection p-value. BH or BY correction over the selected family does not repair invalid marginal p-values.

This issue is particularly damaging because the paper then interprets values such as 69%, 82%, 86%, and 6% as empirical properties of the traces.  The 2019 DSPM-MTC literature is especially relevant here: it explicitly formulates discriminative sequential-pattern discovery as a multiple-hypothesis-testing problem and integrates Fisher testing and multiplicity correction into the mining process.  Earlier work also developed null-model/multiple-testing approaches specifically for statistically significant sequential patterns. 

There is a second exchangeability issue: Azure/BGL windows can share entities and potentially overlap, and SCANIA controls occur in matched risk sets. Global label permutation/hypergeometric arguments require substantially more care in those clustered structures.

**WHAT would fix it:** The authors need a statistically valid discovery/testing design. At least one of the following is necessary:

1. generate the candidate pattern universe without using the case/control labels, e.g. pooled mining, then test it;
2. use independent discovery and inference subsets;
3. rerun the **entire mining procedure inside each permutation**, so candidate selection is incorporated into the null;
4. adopt an established significant/discriminative-pattern-mining procedure with demonstrated FDR/FWER control.

For repeated entities/windows, null randomization must preserve the relevant entity/matched-set structure. The authors should verify type-I error and FDR empirically under synthetic null data before again making claims about the percentage of “predictive” patterns.

Until this is done, the paper's headline “some frequent patterns are predictive, most are noise” is not statistically established in the way claimed.

---

### W2. The SCANIA risk-set sampling idea is appropriate, but the estimator described in §4.4 is not the Prentice–Breslow matched risk-set analysis.

**WHY this blocks:** The sampling step is sensible: each Component X case is matched to controls still at risk at the case's failure lifetime. But the manuscript then says that it constructs a **pooled 2×2 table**, applies a Woolf-Haldane correction, and treats this pooled odds ratio as a hazard-ratio estimate; it subsequently calls the quantity “MH-OR.” 

This discards the very risk-set strata created by the design. In nested case-control/incidence-density sampling, the standard analysis retains each sampled risk set as a matched stratum. Conditional logistic regression, equivalently the sampled partial likelihood/stratified Cox formulation, is the standard estimator; a Mantel-Haenszel analysis would likewise have to stratify by matched risk set rather than collapse all observations into one crude 2×2 table. Modern methodological summaries explicitly state that NCC controls are sampled from each case's risk set and that estimation uses conditional logistic regression treating each case-control set as a stratum.  Matching reviews make the same point and note that time itself is the matching factor under risk-set sampling.  The cited Prentice–Breslow paper adapts proportional-hazards inference to retrospectively sampled cases and controls; it does not justify discarding the matched risk-set structure. 

The subsequent Fisher/hypergeometric test on the pooled table has the same problem. Moreover, controls may legitimately recur in multiple sampled risk sets, and a future case can serve as a control before failure; that is part of the standard NCC design rather than independent pooled observations. 

Consequently, the reported top “MH-OR 2.72 [2.10, 3.51]” and the 6%/8.3% discovery fractions cannot presently be regarded as validated censoring-adjusted hazard-ratio results.

**WHAT would fix it:** Retain a `risk_set_id` for every case and its three controls. For each candidate pattern, estimate its coefficient using conditional logistic regression or a mathematically equivalent matched-set Mantel-Haenszel/partial-likelihood estimator. Recompute confidence intervals and p-values from that matched analysis. Then validate the complete procedure in simulations generated with the actual Component-X censoring/follow-up structure: under HR=1 the type-I/FDR rates should be correct, and under known HR values the estimator should recover them without material bias.

This requires rerunning the entire SCANIA section; it is not a wording correction.

---

### W3. The claimed conceptual novelty is substantially overstated because the closest KDD literature is missing.

**WHY this blocks:** “Frequency does not imply predictiveness” is not a new data-mining observation. Emerging-pattern/contrast-pattern research has explicitly studied itemsets whose support differs between classes since at least 1999.  More directly, *Significance-based discriminative sequential pattern mining* defines discriminative sequences precisely as subsequences whose occurrence differs significantly between labelled sequence sets, uses Fisher testing and multiple-testing correction, and evaluates pattern-based classifiers.  Statistically significant sequential-pattern mining predates that work as well. 

Yet §2 frames the relevant predecessors primarily as FP-Growth, PrefixSpan, Ren et al., and deep log-anomaly systems.  The positioning claim then rests on the specific combination of particular datasets, miners, controls, and held-out logistic regression.  That is not a sufficiently strong novelty argument for DAMI.

Relative to the works named by the authors:

- **FP-Growth/PrefixSpan:** there is no new mining algorithm here.
- **Ren et al. 2021:** already studies failure prediction from large-scale cluster logs using frequent patterns, making the application area itself non-novel. 
- **DeepLog/LogBERT:** these establish strong sequence/log-analysis alternatives, but they do not themselves invalidate an interpretable-pattern contribution. However, if the manuscript makes predictive-performance claims, simply mentioning them as “orthogonal” is not sufficient.
- **DSPM-MTC / significant and contrast pattern mining:** these are the closest novelty threats and currently are not discussed at all.

The risk-set adaptation might provide a useful methodological contribution, but risk-set sampling itself is standard epidemiological methodology; the novel element would have to be a rigorously developed integration with significant pattern discovery, not the estimator as currently formulated.

**WHAT would fix it:** Reframe the novelty away from “we separate predictive frequent patterns from noise.” A defensible DAMI contribution might instead be: censoring-aware **statistically significant discriminative pattern mining for failure-event data**, with a correct matched-set test, scalable pruning/search, formal validity, and comprehensive empirical comparison against DSPM-MTC/contrast-pattern/statistically-significant-pattern methods. Alternatively, present this honestly as an extensive empirical application study rather than an algorithmic contribution, but then the empirical evidence must be considerably broader.

---

### W4. The headline “order gain” measure conflates temporal order with event multiplicity.

**WHY this blocks:** The manuscript compares each sequence with the “itemset counterpart of the same event set” and defines

`order_gain = sequence_lift - itemset_lift`. 

This is not a clean test of ordering. Consider the manuscript's Alibaba example:

`M → M → M`

versus the corresponding itemset `{M}`.

The sequence encodes **three occurrences** of M; the itemset encodes only presence/absence of M. There is no alternative temporal ordering of three identical symbols. Therefore the reported sequence-vs-itemset difference is primarily a multiplicity/count effect, not an ordering effect.

Likewise,

`M → R → M → M`

versus `{M,R}`

changes both ordering **and the requirement for three M occurrences**. The claimed Alibaba “order gain +1.49/+1.69” therefore does not isolate order. The manuscript's operational interpretation that “three consecutive Map completions” show the importance of ordering is especially problematic because repetition count alone could explain the difference. 

**WHAT would fix it:** Add a count-preserving unordered comparator. For every sequence, compare against a multiset/count predicate containing exactly the same event multiplicities. At model level, include event-type count vectors and unordered n-multiset features. A direct within-window order permutation that preserves every event multiplicity provides the appropriate order-specific null. Only the residual advantage over that comparator should be called “order gain.”

The Azure `error2 → error3` example is more naturally order-identifiable because two distinct events can be reversed, but even there the manuscript should directly compare the two orientations or a count-preserving shuffle distribution.

---

### W5. Four traces cannot establish the claimed general “regime of validity.”

**WHY this blocks:** Four heterogeneous case studies are useful empirical evidence. They are not sufficient to infer the conjunction

> rich native discrete vocabulary AND non-self-triggering failure AND sufficient readout-cadence capacity

as the general regime in which the method works. That rule is inferred post hoc from exactly two successes and two failures.  The conclusion goes further and says the method's regime of validity has been “mapped” by the additional traces. 

N=4 is not intrinsically too small for an empirical DAMI paper; it is too small for **this level of generalisation**. The four traces also differ along many axes simultaneously: domain, event vocabulary, synthetic versus real data, entity definition, target construction, observation cadence, preprocessing, class prevalence, and control sampling. Nothing identifies which proposed factor caused the change in performance.

The evidence is further weakened by the fact that one positive trace is synthetic and the other has median operational lead time of 0 seconds, while the two negative traces have substantially different representations. 

**WHAT would fix it:** Either (a) sharply reframe §7.4 as “observations from four contrasting case studies” and remove the general regime rule, or (b) substantiate the rule with a materially larger benchmark plus controlled synthetic experiments in which vocabulary richness, precursor strength, self-triggering behavior, cadence and event multiplicity are varied independently.

For DAMI, option (b) would make the paper substantially stronger.

---

### W6. The SCANIA “6% predictive fraction” is not currently interpretable as a meaningful fraction of distinct predictive knowledge.

**WHY this blocks:** The paper mines 42,453 SCANIA itemsets. Only 281 are removed by the closed-itemset filter, so 99.3% remain closed; 2,560 subsequently pass BY q<0.05. 

The 99.3% closed rate is not, by itself, evidence of a software error. “Closed” eliminates patterns with **exactly identical support** to a strict superset. With numerous histogram bins, it is entirely plausible that adding/removing one bin changes the hit set by one truck, leaving nearly every pattern technically closed. But this also means that closedness is doing almost nothing to solve the real redundancy problem.

Indeed, three of the five strongest patterns are slight variants of combinations from histogram feature 397.  Thus thousands of statistically separate itemsets may represent a much smaller number of highly overlapping support sets or underlying histogram phenomena.

BY is relevant but should not be overinterpreted. If the individual p-values were valid, BY would control the expected false-discovery proportion among reported hypotheses under arbitrary dependence; it would not imply that 2,560 discoveries correspond to 2,560 independent mechanisms. More importantly, W1 and W2 mean the p-values are not presently valid inputs to BY, so the nominal guarantee does not apply.

Finally, the quantity `significant patterns / patterns mined at support 0.05` is not an intrinsic property of SCANIA. Its denominator will change with support threshold, tokenization, maximum pattern length, histogram representation, and redundancy convention.

**WHAT would fix it:** After repairing W1/W2, show:
- null distributions for the **number and fraction** of discoveries generated by the full pipeline;
- min-support sensitivity for the SCANIA fraction;
- clustering/deduplication by support-set similarity, not merely exact closedness;
- results grouped by underlying histogram/counter source;
- preferably minimal generators or another nonredundant pattern representation.

Until then, “6% of SCANIA patterns are predictive” should not be a headline conclusion.

---

### W7. The downstream predictive experiment is too weak for the performance claims made.

**WHY this blocks:** The downstream table is useful, but the comparison is based on one temporal cutoff per dataset, one logistic-regression specification, and no confidence intervals or significance tests for AUROC/AUPRC differences.  The manuscript itself admits that regularization sweeps and cross-machine/cross-job leave-one-out experiments remain undone. 

The baseline called `event_count` is only one scalar: number of events. That is too weak to establish that frequent pattern mining is needed. A much more consequential comparison is against:
- per-event-type count vectors;
- count-preserving/multiset features;
- bigrams/n-grams;
- a regularized sequence/bag model;
- a tree/boosting baseline;
- and, if predictive performance remains part of the claim, at least one representative modern log sequence model.

DeepLog/LogBERT need not be the central competitors if the paper is reframed as interpretable knowledge discovery, but presently §2 claims these models generally outperform pattern mining while no direct comparison is performed. 

A DAMI paper also needs uncertainty on the reported +5.6/+6.2 AUROC improvements, preferably using entity/block bootstrap or multiple temporally ordered splits.

**WHAT would fix it:** Complete the robustness experiments already acknowledged by the authors, strengthen the unordered/count baselines, report confidence intervals and paired performance comparisons, and demonstrate entity-level as well as temporal generalisation where possible.

---

### W8. Case-control sampling makes several “probability,” lift, RR, precision and AUPRC interpretations questionable.

**WHY this blocks:** Controls are deliberately sampled, including a 3:1 control:case ratio for short-lived entities.  Consequently, quantities depending on class prevalence in the sampled dataset are not population probabilities.

For example, §6.1 reports `P(failure | pattern) = 99.6%` for an Azure pattern.  In a case-control style design, this should not be interpreted as a deployment failure probability unless the sampling fraction reproduces the population risk or an explicit correction is applied. Similarly, the stated “relative risk = P(failure|P)/P(failure|¬P)” is not generally estimable as a population risk ratio from a case-control sample. Odds ratios are the natural retrospective estimand.

AUPRC, precision and F1 are also prevalence-sensitive. If the test set uses artificially sampled controls, those values describe the sampled evaluation distribution, not operational prevalence.

**WHAT would fix it:** Precisely document control sampling in train and test; distinguish prevalence-invariant from prevalence-dependent statistics; use ORs for retrospective comparisons; and either evaluate AUPRC/precision on the natural test population or appropriately reweight them. Do not call sample-conditioned probabilities operational failure probabilities.

---

### W9. The abstract and conclusion overstate what the evidence shows.

**WHY this blocks:** The abstract begins with the claim that “only a specific minority” of frequent patterns are predictive, yet two datasets have 54–100% of patterns passing the reported test.  Thus the opening thesis does not actually summarize the results. The abstract is also overloaded: four datasets, multiple horizons, individual sequences, AUROC changes, BGL diagnosis, three SCANIA significance treatments, a hazard-ratio claim, a temporal-vs-static interpretation, censoring methodology, and artefact claims are packed into one paragraph.

“Cleanly separates,” “censoring-valid signal,” “root-cause diagnosis,” and “regime of validity” are stronger than warranted by the present analysis.

**WHAT would fix it:** The abstract should follow a simpler arc: problem → methodological design → principal findings → limitation. A more accurate thesis is that **the predictive content of frequent patterns varies sharply across operational traces, and frequency alone is insufficient to identify discriminative patterns**. Remove the “minority” assertion unless a statistically valid aggregate analysis actually supports it.

---

### W10. Related work and reproducibility are not yet at DAMI submission standard.

**WHY this blocks:** The manuscript's closest methodological literature is absent: emerging/contrast patterns, supervised discriminative pattern mining, statistical significant-pattern mining, statistically significant sequential patterns, and dependence-aware/permutation-based multiple-testing approaches. This is not a cosmetic bibliography gap; it changes the novelty assessment.

The artefact situation is also incomplete. The abstract says that mined patterns, matched-control windows and risk-set-scored patterns “are released as reproducible parquet artefacts,”  but the manuscript contains no corresponding reproducibility/data-and-code section, repository identifier, archive DOI, environment specification, commands, random seeds, split identifiers, or executable reproduction procedure. The final internal notes explicitly say bibliography expansion, robustness experiments and the four-trace numerical audit remain unfinished. 

For a computational DAMI paper whose claims depend on many preprocessing and sampling choices, tables alone are insufficient. Reproducibility is particularly important because the SCANIA result depends on risk-set construction, tokenization, thousands of candidate itemsets and multiple-testing correction.

**WHAT would fix it:** Supply an archived versioned code repository and complete end-to-end reproduction instructions, including dataset acquisition, preprocessing, event/token definitions, window and control construction, random seeds, exact temporal split IDs, SPMF version/JAR, Python environment, pattern files, inference outputs, and scripts that recreate every table/figure.

---

## Substantive issues

**A. The “sanity invariants” are not statistically calibrated.** A rule that a permuted top lift must not come within a factor of 1.5 of the real top lift is arbitrary, and apparently represents a diagnostic rather than a null distribution.  It cannot substitute for formal full-pipeline permutation testing.

**B. BH/BY treatment is inconsistent across traces.** SCANIA invokes BY because itemsets are dependent, but Azure and Alibaba patterns are also strongly dependent through shared subpatterns and support sets. Either justify the relevant positive-dependence assumptions for BH or use a common dependence-aware inference strategy.

**C. “Full-sequence dominant” is an ad hoc descriptive rule, not evidence that the full sequence is a minimal predictor.** The criterion `lift(S) > lift(S') + 0.05` is arbitrary and does not test whether the additional events provide statistically significant incremental information conditional on the shorter sequence.  Calling such a sequence “the minimal predictor” is too strong.

**D. Azure 24h is largely an event-density problem.** Failure windows average 1.58 events and controls 0.077, and event count alone reaches roughly AUROC 0.97.  The near-perfect pattern results therefore should not be presented as strong evidence for sophisticated pattern knowledge unless the authors demonstrate incremental value conditional on event density.

**E. Alibaba's median lead time of 0 seconds substantially weakens the early-warning framing.** The signal may still be useful for next-task scheduling/risk classification, but it is not a conventional predictive-maintenance warning horizon. 

**F. The BGL negative control is partly target-definition dependent.** Removing prior alerts while defining the target as the first alert in an episode is defensible, but naturally eliminates the strongest self-exciting signal. The paper should distinguish “no non-alert precursor under this target definition” from a stronger claim that BGL contains no useful sequential failure information.

**G. SCANIA tokenization is a major modeling decision.** A per-vehicle 90th-percentile absolute delta converts 105 numeric variables into many binary surprise tokens.  The ceiling experiments are useful, but they do not establish that all plausible trajectory representations have been exhausted.

**H. The “root-cause diagnosis” of SCANIA is too categorical.** Aggregate per-truck CV AUROC 0.826 versus temporal performance around 0.60–0.67 is evidence consistent with static-profile signal, but it can also reflect temporal distribution shift, differences in information available over the vehicle history, and validation-design effects.  Call this a diagnostic hypothesis/evidence, not a demonstrated root cause.

**I. Pattern-count fractions should not be compared as if their denominators were commensurate.** Six Azure patterns, hundreds of Azure sequences and 42,453 SCANIA itemsets are generated from very different feature spaces and support geometry. “86% vs 6%” has intuitive appeal but is not an apples-to-apples trace-level statistic.

**J. The paper needs computational complexity/runtime reporting.** A DAMI pattern-mining paper processing Alibaba-scale data and 42k+ SCANIA candidates should report mining time, scoring time, memory use, number of transactions/windows, number of candidates before/after pruning, and how these quantities scale with support.

**K. `closed-itemset post-filter losslessly deduplicates` should be stated more carefully.** Closed itemsets preserve support information for support-equivalent itemsets, but this does not make the resulting set nonredundant for interpretation or prediction.

**L. The paper currently oscillates between three contributions:** statistical pattern discovery, predictive modeling, and empirical characterization of operational logs. DAMI would benefit from choosing one as primary. At present the statistical discovery claim is central, but it is also the least technically secure component.

---

## Detailed comments by section

| § | line | issue | severity |
|---|---:|---|---|
| Abstract | 290–326 | “Only a specific minority” conflicts with the reported 54–100% significant fractions on Azure/Alibaba; abstract is too result-dense and overclaims “clean separation.” | Major |
| §1 | 333–339 | Introduction initially frames the central question as sequences vs itemsets, while later the paper says the central object is predictive-vs-frequent separation. The primary research question is unstable. | Major |
| §1 | 342–365 | Contribution 1 claims valid significance separation before establishing a post-selection-valid testing procedure. Contribution 2 describes a standard risk-set idea as a methodological generalisation but does not formulate the matched estimator correctly. | Critical |
| §2.1 | 373–393 | Classic pattern-mining background is adequate but disproportionately algorithm-centric given that the paper's claimed contribution is statistical discrimination. | Moderate |
| §2.2 | 399–423 | Missing the closest discriminative/significant sequential-pattern literature. DeepLog/LogBERT are less direct competitors than DSPM-MTC and contrast-pattern methods. | Critical |
| §2.6 | 460–465 | Exact hypergeometric p-value is exact for a fixed hypothesis/hit set under exchangeability; the manuscript does not address data-dependent pattern selection. | Critical |
| §2.7 | 467–475 | Novelty is argued mainly from a particular combination of datasets and methods. This is insufficient for DAMI. | Critical |
| §3.1 | 480–490 | Exclusion of synthetic generator seed failures is reasonable, but Azure's synthetic construction should be more prominent in interpreting generality. | Moderate |
| §3.2 | 498–510 | `batch_task` per-job formulation may be valid, but it is not a machine/component failure-prediction problem comparable to the other traces. Clarify the common target abstraction. | Moderate |
| §3.3 | 513–521 | TODO citation remains. The construction of rack entities and treatment of INFO/alerts need reproducible preprocessing details. | Major |
| §3.4 | 524–537 | TODO citation remains. The 90th-percentile tokenization is consequential and needs sensitivity analysis across thresholds/representations. | Major |
| §4.1 | 541–554 | “Matched controls” includes cross-entity random non-failure sampling without an explicit matching variable. This is case-control sampling, not necessarily matching. State exact ratio, replacement/reuse and exclusion rules. | Major |
| §4.1 | 547–559 | Same-entity controls can be dependent/overlapping with case windows; this matters for permutation inference and uncertainty. | Major |
| §4.2 | 567–571 | Candidates are mined only in failure windows. This creates the central post-selection inference problem. | Critical |
| §4.2 | 572–578 | `order_gain` does not preserve multiplicity and therefore does not isolate order. | Critical |
| §4.3 | 580–590 | Factor-of-1.5 permutation invariant is arbitrary and cannot establish statistical calibration. | Major |
| §4.4 | 601–614 | Risk-set sampling is appropriate in spirit, but pooling matched sets into one 2×2 table discards the matching structure. Calling the result MH-OR is inaccurate as written. | Critical |
| §4.4 | 615–620 | Fisher exact test on the pooled risk-set table is not the matched NCC inference implied by Prentice–Breslow. | Critical |
| §4.5 | 623–631 | Fixed-hit-set hypergeometric derivation does not incorporate selection by case-class support; BH therefore does not validate the pipeline. | Critical |
| §5 | 633–651 | Single temporal cutoff; no repeated temporal splits, entity-disjoint evaluation, model-selection description or uncertainty estimates. | Major |
| §5 | 647–651 | F1/precision/AUPRC are prevalence-sensitive; clarify whether the held-out controls retain natural prevalence or artificial sampling. | Major |
| §5.1 | 654–659 | Azure 24h has overwhelming raw event-count separation; interpretations of individual patterns need conditioning on event density. | Major |
| §5.2 | 663–728 | Useful min-support sweep, but only Azure receives this level of sensitivity analysis. SCANIA's 6% fraction especially needs the same treatment. | Major |
| §6.1 | 733–750 | `P(failure|pattern)=99.6%` is potentially misleading under case-control sampling; Alibaba repeated-M examples conflate count and order. | Critical |
| §6.2 | 754–877 | Good descriptive table, but no confidence intervals/significance tests for performance differences and baselines are weak. | Major |
| §6.3 | 880–887 | Mean `order_gain` inherits the multiplicity confound; it should not be interpreted as an order-specific effect. | Critical |
| §6.4 | 890–1006 | “Fraction significant” is treated as central trace property despite dependence on mining threshold, feature representation and invalid current inference. | Critical |
| §6.6 | 1017–1033 | BY does not repair post-selection-invalid or incorrectly matched p-values. “Honest predictive fraction” is therefore unjustified. | Critical |
| §6.6 | 1021–1024 | 99.3% closed is plausible but demonstrates that exact-support closedness does little to eliminate near-redundant histogram combinations. | Major |
| §6.6 | 1053–1119 | Top SCANIA signatures are dominated by variants of histogram 397, undercutting the implication of thousands of distinct discoveries. | Major |
| §6.7 | 1124–1133 | Alibaba 0-second median lead time and Azure generator-imposed 24h lead time substantially narrow the operational early-warning claim. | Major |
| §7.1 | 1120–1125 | `+0.05 lift` definition of “minimal predictor” has no statistical basis. | Major |
| §7.1 | 1184–1205 | “Predictor IS in the entire sequence” is too categorical without conditional/incremental testing. | Major |
| §7.3 | 1229–1245 | Operational interpretations of Azure/Alibaba sequences are plausible narratives, not demonstrated causal/mechanistic explanations. | Moderate |
| §7.4 | 1228–1249 | Four traces show heterogeneity but do not identify a transferable regime. | Critical |
| §7.4 | 1250–1263 | SCANIA ceiling diagnostic is useful, but “classifier capacity is not the constraint” and representation-limit claims exceed the tested model family. | Major |
| §7.4 | 1279–1307 | Static-profile explanation is plausible but not uniquely established; “root cause” and definitive regime statement should be softened. | Major |
| §8 | 1316–1337 | Several items listed as future work—especially robustness and sequence support sweeps—are required analyses rather than optional limitations for a DAMI submission. | Major |
| §9 | 1340–1350 | “Replicates” and “regime of validity is mapped” overstate evidence from one synthetic and one real positive trace plus two heterogeneous negatives. | Critical |
| Draft status | 1352–1370 | Manuscript explicitly documents unfinished experiments, literature expansion and numerical audit. It is not submission-ready. | Critical |
| Reproducibility | throughout | Abstract claims released parquet artefacts, but no repository/data/code availability section or executable reproduction path is provided. | Major |

---

## Minor / editorial

1. Remove the duplicated title/section-1 title structure at the beginning.
2. Remove all internal drafting text (“Restructured draft,” “TODO,” “scout returning,” “paper-build skill,” “What is still missing,” “What is verified”) before submission.
3. `survival- style`, `per- pattern`, `within- window`, and `rich-discrete- vocabulary` contain spacing/hyphenation artefacts.
4. Avoid all-caps prose such as “RECURRENT ORDERED SEQUENCES,” “ITEMSET COUNTERPART,” “BOTH,” and “FULL” in journal text.
5. §7 numbering jumps from 7.1 to 7.3.
6. Define exactly what the two numbers in the §6.2 cells represent in the table header itself rather than only in a post-table note.
7. Replace “wins,” “winning traces,” “honest predictive fraction,” and “ceiling” where possible with neutral statistical terminology.
8. `0 s` should be described carefully when timestamps are discretized/task-event based; distinguish zero measured interval from literally simultaneous events.
9. “self-triggering” needs a formal definition if retained as part of the claimed regime.
10. “readout-cadence signal capacity exceeds the target AUROC bar” is not a well-defined data property and should not appear as a formal regime condition without an operational definition.
11. Report all random seeds and whether the 3 controls per SCANIA case were sampled with replacement across risk sets.
12. The bibliography should cite Prentice–Breslow formally rather than only in prose.
13. Standardize “artefact/artifact” spelling.
14. Avoid calling an odds rat