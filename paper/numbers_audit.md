# Numbers audit — paper/skeleton.md

Total claims audited: 61. Pass: 61. Mismatch: 0.

## Full audit trail

| section       | claim                                                      | cited      | computed   | match_str   |
|:--------------|:-----------------------------------------------------------|:-----------|:-----------|:------------|
| 3.1           | Azure: 100 machines                                        | 100        | 100        | PASS        |
| 3.1           | Azure: 3,919 non-fatal errors                              | 3,919      | 3919       | PASS        |
| 3.1           | Azure: 3,286 maintenance actions                           | 3,286      | 3286       | PASS        |
| 3.1           | Azure: 761 component replacements (=terminal failures)     | 761        | 761        | PASS        |
| 3.1           | Azure: 18 seed failures at 2015-01-02 03:00                | 18         | 18         | PASS        |
| 3.2           | Alibaba: 14,295,731 tasks                                  | 14,295,731 | 14295731   | PASS        |
| 3.2           | Alibaba: 4,201,014 jobs                                    | 4,201,014  | 4201014    | PASS        |
| 3.2           | Alibaba: 83,207 jobs with >= 1 Failed task                 | 83,207     | 83207      | PASS        |
| 5.1           | 99.6% of 1h failure windows empty                          | 99.6%      | 99.6       | PASS        |
| 5.1           | 98% of 6h failure windows empty                            | 98%        | 98.1       | PASS        |
| 5.1           | 24h failure mean events = 1.58                             | 1.58       | 1.58       | PASS        |
| 5.1           | 24h control mean events = 0.077                            | 0.077      | 0.077      | PASS        |
| 5.2           | Top 24h itemset lift 3.99 for {error2, error3}             | 3.99       | 3.99       | PASS        |
| 5.2           | Top 24h itemset support in failures = 38.2%                | 38.2%      | 38.2       | PASS        |
| 5.2           | Top 24h itemset support in controls = 0.04%                | 0.04%      | 0.04       | PASS        |
| 5.2           | Top 24h itemset P(failure|pattern) = 99.6%                 | 99.6%      | 99.6       | PASS        |
| 5.2           | Sequence maintenance:comp4 -> error2 -> error3 lift = 3.73 | 3.73       | 3.73       | PASS        |
| 5.2           | Same sequence as itemset lift = 2.22                       | 2.22       | 2.22       | PASS        |
| Abstract      | Combined − itemsets Azure last5 = +5.6 AUROC               | +5.6       | 5.6        | PASS        |
| Abstract      | Combined − itemsets Alibaba last3 = +6.2 AUROC             | +6.2       | 6.2        | PASS        |
| Abstract      | Alibaba last5 M→R→M→M sequence lift = 2.43                 | 2.43       | 2.43       | PASS        |
| Abstract      | Same Alibaba sequence itemset lift = 0.94                  | 0.94       | 0.94       | PASS        |
| 5.2           | Alibaba last3 M→M→M sequence lift = 3.06                   | 3.06       | 3.06       | PASS        |
| 5.2           | Same Alibaba sequence itemset lift = 1.37                  | 1.37       | 1.37       | PASS        |
| 7.2 (ceiling) | SCANIA LightGBM AUROC on histogram-aware features = 0.60   | 0.60       | 0.6        | PASS        |
| 7.2 (ceiling) | SCANIA LR AUROC on same features = 0.58                    | 0.58       | 0.58       | PASS        |
| 7.2 (ceiling) | SCANIA ceiling feature count = 113                         | 113        | 113        | PASS        |
| 5.3           | Azure 24h combined AUROC = 0.996                           | 0.996      | 0.996      | PASS        |
| 5.3           | Azure 24h combined AUPRC = 0.988                           | 0.988      | 0.988      | PASS        |
| 5.3           | Azure last5 combined AUROC = 0.81                          | 0.81       | 0.81       | PASS        |
| 5.3           | Azure last5 combined AUPRC = 0.72                          | 0.72       | 0.72       | PASS        |
| 5.3           | Azure last10 combined AUROC = 0.696                        | 0.696      | 0.696      | PASS        |
| 5.3           | Azure last10 combined AUPRC = 0.576                        | 0.576      | 0.576      | PASS        |
| 5.3           | Azure last5 itemsets_only AUROC = 0.754                    | 0.754      | 0.754      | PASS        |
| 5.3           | Azure last5 itemsets_only AUPRC = 0.563                    | 0.563      | 0.563      | PASS        |
| 5.3           | Alibaba last3 combined AUROC = 0.813                       | 0.813      | 0.813      | PASS        |
| 5.3           | Alibaba last3 combined AUPRC = 0.631                       | 0.631      | 0.631      | PASS        |
| 5.3           | Alibaba last5 combined AUROC = 0.741                       | 0.741      | 0.741      | PASS        |
| 5.3           | Alibaba last5 combined AUPRC = 0.574                       | 0.574      | 0.574      | PASS        |
| 5.3           | Alibaba last10 combined AUROC = 0.741                      | 0.741      | 0.741      | PASS        |
| 5.3           | Alibaba last10 combined AUPRC = 0.593                      | 0.593      | 0.593      | PASS        |
| 5.3           | BGL last20 combined AUROC = 0.512                          | 0.512      | 0.512      | PASS        |
| 5.3           | BGL last20 combined AUPRC = 0.256                          | 0.256      | 0.256      | PASS        |
| 5.3           | BGL last20 itemsets_only AUROC = 0.483                     | 0.483      | 0.483      | PASS        |
| 5.3           | BGL last20 itemsets_only AUPRC = 0.245                     | 0.245      | 0.245      | PASS        |
| 5.3           | SCANIA last10 combined AUROC = 0.596                       | 0.596      | 0.596      | PASS        |
| 5.3           | SCANIA last10 combined AUPRC = 0.154                       | 0.154      | 0.154      | PASS        |
| 5.3           | SCANIA last20 combined AUROC = 0.567                       | 0.567      | 0.567      | PASS        |
| 5.3           | SCANIA last20 combined AUPRC = 0.132                       | 0.132      | 0.132      | PASS        |
| 5.5           | Azure 24h itemsets significant: 6/6                        | 6/6        | 6/6        | PASS        |
| 5.5           | Azure 24h sequences significant: 7/7                       | 7/7        | 7/7        | PASS        |
| 5.5           | Azure last5 itemsets significant: 53/77                    | 53/77      | 53/77      | PASS        |
| 5.5           | Azure last10 sequences significant: 562/657                | 562/657    | 562/657    | PASS        |
| 5.5           | Alibaba last3 itemsets significant: 6/10                   | 6/10       | 6/10       | PASS        |
| 5.5           | Alibaba last10 sequences significant: 59/109               | 59/109     | 59/109     | PASS        |
| 5.6           | sensitivity last5 combined ms=0.02 AUROC = 0.815           | 0.815      | 0.815      | PASS        |
| 5.6           | sensitivity last5 combined ms=0.15 AUROC = 0.774           | 0.774      | 0.774      | PASS        |
| 5.6           | sensitivity last5 itemsets_only ms=0.02 AUROC = 0.761      | 0.761      | 0.761      | PASS        |
| 5.6           | sensitivity last10 combined ms=0.1 AUROC = 0.751           | 0.751      | 0.751      | PASS        |
| 5.6           | sensitivity last10 itemsets_only ms=0.02 AUROC = 0.578     | 0.578      | 0.578      | PASS        |
| 5.6           | sensitivity 24h combined ms=0.02 AUROC = 0.996             | 0.996      | 0.996      | PASS        |