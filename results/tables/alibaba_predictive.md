# Alibaba cluster-trace-v2018 Phase 6: predictive evaluation

Per-job failure prediction. Entity = job_name, event vocabulary = (normalized-status, task_role letter prefix). Temporal split at 2018-01-07. Patterns mined on train only.

| horizon   | feature_set    |   n_features |   auroc |   auprc |   f1_at_0.5 |   precision_at_0.5 |   recall_at_0.5 |
|:----------|:---------------|-------------:|--------:|--------:|------------:|-------------------:|----------------:|
| last3     | event_count    |            1 |   0.688 |   0.499 |       0.547 |              1     |           0.376 |
| last3     | itemsets_only  |            6 |   0.751 |   0.437 |       0.502 |              0.531 |           0.475 |
| last3     | sequences_only |            2 |   0.503 |   0.201 |       0.012 |              1     |           0.006 |
| last3     | combined       |            9 |   0.813 |   0.631 |       0.603 |              0.996 |           0.432 |
| last5     | event_count    |            1 |   0.602 |   0.5   |       0.34  |              1     |           0.205 |
| last5     | itemsets_only  |            5 |   0.672 |   0.343 |       0.229 |              0.979 |           0.129 |
| last5     | sequences_only |            2 |   0.511 |   0.213 |       0.043 |              0.956 |           0.022 |
| last5     | combined       |            8 |   0.741 |   0.574 |       0.585 |              0.993 |           0.414 |
| last10    | event_count    |            1 |   0.588 |   0.498 |       0     |              0     |           0     |
| last10    | itemsets_only  |            5 |   0.676 |   0.358 |       0.269 |              0.979 |           0.156 |
| last10    | sequences_only |            2 |   0.522 |   0.23  |       0.084 |              0.977 |           0.044 |
| last10    | combined       |            8 |   0.741 |   0.593 |       0.269 |              0.979 |           0.156 |


## Comparison to Azure PdM

Azure PdM (per-machine): combined at last5 -> AUROC 0.810, AUPRC 0.720.
Alibaba (per-job): combined at last3 -> AUROC 0.813, AUPRC 0.631.

Same finding in both traces: combining sequences with itemsets adds 5-10 AUROC points over itemsets-only. Sequences-only has few surviving features (2 on Alibaba vs 6-16 on Azure) but is high-precision (~0.95+ at 0.5 threshold), suggesting the mined ordered patterns fire rarely but reliably.
