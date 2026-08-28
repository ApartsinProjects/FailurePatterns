# Alibaba lead time on true positives

temporal split at 2018-01-07, entity = job. TPs are test-set failure windows the classifier labels positive at threshold 0.5. Lead time = anchor - last_event_ts.

| horizon   | feature_set    |   n_tp |   median_lead_min |   median_lead_h |   p25_lead_seconds |   p75_lead_seconds |
|:----------|:---------------|-------:|------------------:|----------------:|-------------------:|-------------------:|
| last3     | event_count    |   1480 |              0.42 |            0.01 |                  4 |             107.25 |
| last3     | itemsets_only  |   1870 |              0.25 |            0    |                  4 |              85.75 |
| last3     | sequences_only |     24 |              0.1  |            0    |                  5 |               8    |
| last3     | combined       |   1701 |              0.22 |            0    |                  4 |              89    |
| last5     | event_count    |    806 |              0.37 |            0.01 |                  3 |              96    |
| last5     | itemsets_only  |    509 |              0.08 |            0    |                  4 |             962    |
| last5     | sequences_only |     86 |              0.08 |            0    |                  3 |               8    |
| last5     | combined       |   1630 |              0.17 |            0    |                  4 |              66.75 |
| last10    | event_count    |      0 |            nan    |          nan    |                nan |             nan    |
| last10    | itemsets_only  |    613 |              0.08 |            0    |                  4 |              23    |
| last10    | sequences_only |    173 |              0.08 |            0    |                  4 |               8    |
| last10    | combined       |    613 |              0.08 |            0    |                  4 |              23    |
