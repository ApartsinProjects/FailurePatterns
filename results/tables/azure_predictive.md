# Azure PdM Phase 6: predictive evaluation

Temporal split at 2015-09-01. Patterns mined on train only.

Logistic regression on each feature set.


| horizon   | feature_set    |   n_features |   auroc |   auprc |   f1_at_0.5 |   precision_at_0.5 |   recall_at_0.5 |
|:----------|:---------------|-------------:|--------:|--------:|------------:|-------------------:|----------------:|
| 24h       | event_count    |            1 |   0.972 |   0.906 |       0.912 |              0.839 |           1     |
| 24h       | itemsets_only  |            6 |   0.996 |   0.988 |       0.98  |              0.96  |           1     |
| 24h       | sequences_only |            0 | nan     | nan     |     nan     |            nan     |         nan     |
| 24h       | combined       |            7 |   0.996 |   0.988 |       0.98  |              0.96  |           1     |
| last5     | event_count    |            1 |   0.5   |   0.339 |       0     |              0     |           0     |
| last5     | itemsets_only  |           41 |   0.754 |   0.563 |       0.574 |              0.618 |           0.536 |
| last5     | sequences_only |            6 |   0.665 |   0.561 |       0.501 |              0.827 |           0.36  |
| last5     | combined       |           48 |   0.81  |   0.72  |       0.613 |              0.709 |           0.54  |
| last10    | event_count    |            1 |   0.5   |   0.343 |       0     |              0     |           0     |
| last10    | itemsets_only  |          331 |   0.643 |   0.5   |       0.462 |              0.51  |           0.423 |
| last10    | sequences_only |           16 |   0.666 |   0.531 |       0.417 |              0.722 |           0.293 |
| last10    | combined       |          348 |   0.696 |   0.576 |       0.53  |              0.561 |           0.502 |

## Reading the numbers

- **24h horizon:** event-count alone already reaches AUROC 0.97 (mean 1.58 events in failure windows vs 0.077 in controls). Itemsets push to 0.996. **No sequence survived the shuffle-null at 24h**, so sequences_only has n_features = 0 and combined equals itemsets_only. At this horizon order does not help; the itemset already captures everything.
- **last5 horizon:** event-count is chance because n_events = 5 for both classes. Itemsets_only reaches AUROC 0.75. Sequences_only at 6 features is high-precision (0.83) but low-recall (0.36). Combined: **AUROC 0.81, AUPRC 0.72 - +5.6 AUROC and +15.7 AUPRC points above itemsets_only.**
- **last10 horizon:** same shape as last5 but with more features. Combined reaches AUROC 0.70, +5.3 AUROC over itemsets_only.

**Answer to the paper's Experiment 4 question:** temporal order in mined sequences contributes real predictive information beyond the itemset representation, but only when the window definition is rich enough for order to be a real degree of freedom. On Azure PdM that is the count-based (last-K events) horizons, not the short time horizons where 24h windows contain at most 1-2 events.
