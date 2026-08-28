# Azure lead time on true positives

temporal split at 2015-09-01, entity = machine. TPs are test-set failure windows the classifier labels positive at threshold 0.5. Lead time = anchor - last_event_ts.

| horizon   | feature_set    |   n_tp |   median_lead_min |   median_lead_h |   p25_lead_seconds |   p75_lead_seconds |
|:----------|:---------------|-------:|------------------:|----------------:|-------------------:|-------------------:|
| 24h       | event_count    |    239 |              1440 |              24 |              86400 |              86400 |
| 24h       | itemsets_only  |    239 |              1440 |              24 |              86400 |              86400 |
| 24h       | sequences_only |      0 |               nan |             nan |                nan |                nan |
| 24h       | combined       |    239 |              1440 |              24 |              86400 |              86400 |
| last5     | event_count    |      0 |               nan |             nan |                nan |                nan |
| last5     | itemsets_only  |    128 |              1440 |              24 |              86400 |              86400 |
| last5     | sequences_only |     86 |              1440 |              24 |              86400 |              86400 |
| last5     | combined       |    129 |              1440 |              24 |              86400 |              86400 |
| last10    | event_count    |      0 |               nan |             nan |                nan |                nan |
| last10    | itemsets_only  |    101 |              1440 |              24 |              86400 |              86400 |
| last10    | sequences_only |     70 |              1440 |              24 |              86400 |              86400 |
| last10    | combined       |    120 |              1440 |              24 |              86400 |              86400 |
