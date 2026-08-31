# Numbers audit — paper/skeleton.md

Total claims audited: 139. Pass: 139. Mismatch: 0.

## Full audit trail

| section           | claim                                                              | cited        | computed        | match_str   |
|:------------------|:-------------------------------------------------------------------|:-------------|:----------------|:------------|
| 3.1               | Azure: 100 machines                                                | 100          | 100             | PASS        |
| 3.1               | Azure: 3,919 non-fatal errors                                      | 3,919        | 3919            | PASS        |
| 3.1               | Azure: 3,286 maintenance actions                                   | 3,286        | 3286            | PASS        |
| 3.1               | Azure: 761 component replacements (=terminal failures)             | 761          | 761             | PASS        |
| 3.1               | Azure: 18 seed failures at 2015-01-02 03:00                        | 18           | 18              | PASS        |
| 3.2               | Alibaba: 14,295,731 tasks                                          | 14,295,731   | 14295731        | PASS        |
| 3.2               | Alibaba: 4,201,014 jobs                                            | 4,201,014    | 4201014         | PASS        |
| 3.2               | Alibaba: 83,207 jobs with >= 1 Failed task                         | 83,207       | 83207           | PASS        |
| 5.1               | 99.6% of 1h failure windows empty                                  | 99.6%        | 99.6            | PASS        |
| 5.1               | 98% of 6h failure windows empty                                    | 98%          | 98.1            | PASS        |
| 5.1               | 24h failure mean events = 1.58                                     | 1.58         | 1.58            | PASS        |
| 5.1               | 24h control mean events = 0.077                                    | 0.077        | 0.077           | PASS        |
| 5.2               | Top 24h itemset lift 3.99 for {error2, error3}                     | 3.99         | 3.99            | PASS        |
| 5.2               | Top 24h itemset support in failures = 38.2%                        | 38.2%        | 38.2            | PASS        |
| 5.2               | Top 24h itemset support in controls = 0.04%                        | 0.04%        | 0.04            | PASS        |
| 5.2               | Top 24h itemset P(failure|pattern) = 99.6%                         | 99.6%        | 99.6            | PASS        |
| 5.2               | Sequence maintenance:comp4 -> error2 -> error3 lift = 3.73         | 3.73         | 3.73            | PASS        |
| 5.2               | Same sequence as itemset lift = 2.22                               | 2.22         | 2.22            | PASS        |
| Abstract          | Combined − itemsets Azure last5 = +5.6 AUROC                       | +5.6         | 5.6             | PASS        |
| Abstract          | Combined − itemsets Alibaba last3 = +6.2 AUROC                     | +6.2         | 6.2             | PASS        |
| Abstract          | Alibaba last5 M→R→M→M sequence lift = 2.43                         | 2.43         | 2.43            | PASS        |
| Abstract          | Same Alibaba sequence itemset lift = 0.94                          | 0.94         | 0.94            | PASS        |
| 5.2               | Alibaba last3 M→M→M sequence lift = 3.06                           | 3.06         | 3.06            | PASS        |
| 5.2               | Same Alibaba sequence itemset lift = 1.37                          | 1.37         | 1.37            | PASS        |
| 7.2 (ceiling)     | SCANIA LightGBM AUROC on histogram-aware features = 0.60           | 0.60         | 0.6             | PASS        |
| 7.2 (ceiling)     | SCANIA LR AUROC on same features = 0.58                            | 0.58         | 0.58            | PASS        |
| 7.2 (ceiling)     | SCANIA ceiling feature count = 113                                 | 113          | 113             | PASS        |
| 7.2 (APS)         | APS Failure LightGBM AUROC = 0.994                                 | 0.994        | 0.994           | PASS        |
| 7.2 (APS)         | APS Failure LightGBM AUPRC = 0.934                                 | 0.934        | 0.934           | PASS        |
| 7.2 (APS)         | APS Failure LR AUROC = 0.979                                       | 0.979        | 0.979           | PASS        |
| 7.2 (APS)         | APS Failure LR AUPRC = 0.800                                       | 0.800        | 0.8             | PASS        |
| 7.2 (APS)         | APS Failure feature count = 170                                    | 170          | 170             | PASS        |
| 7.2 (APS)         | APS Failure test set size = 16000                                  | 16000        | 16000           | PASS        |
| 7.2 (APS)         | APS Failure train positive rate = 0.0167                           | 0.0167       | 0.0167          | PASS        |
| 6.5               | SCANIA risk-set mined 42,453 candidate itemsets                    | 42453        | 42453           | PASS        |
| 6.5               | SCANIA risk-set 4,829 significant patterns (CI excludes 1)         | 4829         | 4829            | PASS        |
| 6.5               | SCANIA risk-set top pattern MH-OR = 2.72                           | 2.72         | 2.72            | PASS        |
| 6.5               | SCANIA risk-set top pattern CI low = 2.10                          | 2.10         | 2.1             | PASS        |
| 6.6               | SCANIA risk-set top pattern CI high = 3.51                         | 3.51         | 3.51            | PASS        |
| 7.1               | Azure last10 full-sequence-dominant fraction = 0.955               | 0.955        | 0.955           | PASS        |
| 7.1               | Azure last10 full-dominant count = 191 / 200                       | 191          | 191             | PASS        |
| 7.1               | Azure last5 full-sequence-dominant fraction = 0.788                | 0.788        | 0.788           | PASS        |
| 7.1               | Alibaba last3 full-dominant fraction = 0.167                       | 0.167        | 0.167           | PASS        |
| 7.1               | Alibaba last10 full-dominant fraction = 0.295                      | 0.295        | 0.295           | PASS        |
| 6.4 (Kelmarsh)    | Kelmarsh last5: 30 BH-sig / 35 mined                               | 30/35        | 30/35           | PASS        |
| 6.4 (Kelmarsh)    | Kelmarsh last5: 30 BY-sig / 35 mined                               | 30/35        | 30/35           | PASS        |
| 6.4 (Kelmarsh)    | Kelmarsh last10: 115 BH-sig / 136 mined                            | 115/136      | 115/136         | PASS        |
| 3.4 (Kelmarsh)    | Kelmarsh 2016-2017: 482 forced outages                             | 482          | 482             | PASS        |
| 3.4 (Kelmarsh)    | Kelmarsh: 6 turbines                                               | 6            | 6               | PASS        |
| 6.4b              | Kelmarsh last5 mined-itemset F1 = 0.677                            | 0.677        | 0.677           | PASS        |
| 6.4b              | Kelmarsh last10 mined-itemset F1 = 0.729                           | 0.729        | 0.729           | PASS        |
| 6.4b              | Penmanshiel last5 mined-itemset F1 = 0.484                         | 0.484        | 0.484           | PASS        |
| 6.4c              | Kelmarsh last5 mined-itemset PPV@0.01 = 0.20                       | 0.20         | 0.2             | PASS        |
| 6.4c              | Kelmarsh last10 mined-itemset PPV@0.01 = 0.11                      | 0.11         | 0.11            | PASS        |
| 6.4 (Penmanshiel) | Penmanshiel last5: 27 BY-sig / 30 mined                            | 27/30        | 27/30           | PASS        |
| 6.4 (Penmanshiel) | Penmanshiel last10: 83 BY-sig / 98 mined                           | 83/98        | 83/98           | PASS        |
| 3.5 (Penmanshiel) | Penmanshiel 2016: 790 forced outages                               | 790          | 790             | PASS        |
| 3.5 (Penmanshiel) | Penmanshiel: 9 turbines                                            | 9            | 9               | PASS        |
| 6.4               | Azure 24h {error2, error3}: 135 cases, 0 controls, lift 4.0        | 135/0        | 135/0           | PASS        |
| 6.4               | Alibaba last5 {task_waiting:R} lift = 4.01                         | 4.01         | 4.01            | PASS        |
| 6.4               | Alibaba last5 {task_waiting:R}: 829 cases, 9 controls              | 829/9        | 829/9           | PASS        |
| 6.4               | BGL: 0 signatures pass post-selection-valid BY q<0.05              | 0            | 0               | PASS        |
| 6.10              | SCANIA matched (valid) inference BH q<0.05 and CI>1: 93/200        | 93           | 93              | PASS        |
| 6.10              | SCANIA matched (valid) inference BY q<0.05 and CI>1: 74/200        | 74           | 74              | PASS        |
| 6.4               | Prospective Kelmarsh recall = 0.95                                 | 0.95         | 0.948           | PASS        |
| 6.4               | Prospective Kelmarsh median lead = 198 min                         | 198          | 198             | PASS        |
| 6.4               | Prospective Penmanshiel precision = 0.78                           | 0.78         | 0.78            | PASS        |
| 6.4               | Kelmarsh clean degradation chains = 3                              | 3            | 3               | PASS        |
| 6.4               | Penmanshiel clean degradation chains = 2                           | 2            | 2               | PASS        |
| 6.4               | Penmanshiel longest clean chain lead ~11.4h                        | ~11.4        | 11.4            | PASS        |
| 6.4               | 72h guarded Penmanshiel chains = 0                                 | 0            | 0               | PASS        |
| 6.4               | Kelmarsh->Penmanshiel transfer recall = 0.95                       | 0.95         | 0.952           | PASS        |
| 6.4               | Penmanshiel->Kelmarsh transfer recall = 0.50                       | 0.50         | 0.498           | PASS        |
| 6.4               | Kelmarsh fan-family co-location span < 1 min                       | 0.16         | 0.16            | PASS        |
| 6.3               | Kelmarsh presence effect ~0.244 CI excludes 0                      | ~0.244       | 0.244           | PASS        |
| 6.3               | Kelmarsh order + multiplicity both CI-include-0 (presence only)    | both incl 0  | both incl 0     | PASS        |
| 6.3               | Penmanshiel presence effect ~0.326 CI excludes 0                   | ~0.326       | 0.326           | PASS        |
| 6.3               | Penmanshiel order + multiplicity both CI-include-0 (presence only) | both incl 0  | both incl 0     | PASS        |
| 6.3               | Azure order increment +0.032 CI excludes 0                         | +0.032       | 0.032           | PASS        |
| 6.3               | Azure multiplicity increment CI includes 0                         | incl 0       | [-0.015, 0.034] | PASS        |
| 6.3               | Alibaba multiplicity increment +0.029 CI excludes 0                | +0.029       | 0.029           | PASS        |
| 6.3               | Alibaba order increment CI includes 0                              | incl 0       | [-0.022, 0.013] | PASS        |
| 6.7               | Kelmarsh headline signature selected in 20/20 splits               | 20/20        | 20/20           | PASS        |
| 6.7               | Penmanshiel Jaccard stability >= 0.75                              | >=0.75       | 0.79            | PASS        |
| 8                 | Landmarked SCANIA token_counts AUROC = 0.62                        | 0.62         | 0.615           | PASS        |
| 8                 | Landmarked SCANIA: no representation clears 0.75                   | <0.75        | 0.615           | PASS        |
| 6.7               | Validation: split false discoveries on permuted labels = 0         | 0            | 0               | PASS        |
| 6.7               | Validation: naive BH false discoveries on permuted labels > 0      | >0           | 6               | PASS        |
| 6.5b              | Post-selection Azure last10 seqs: 537 BH-sig / 694 mined           | 537/694      | 537/694         | PASS        |
| 6.5b              | Post-selection Azure last10 seqs BY: 461/694                       | 461          | 461             | PASS        |
| 6.5b              | Post-selection SCANIA last20 seqs: 6262 mined, 3 BY-sig            | 6262/3       | 6262/3          | PASS        |
| 6.5b              | BGL last5 CloSpan closed 10 / raw 20                               | 10/20        | 10/20           | PASS        |
| 6.5b              | BGL last10 CloSpan closed 26 / raw 39                              | 26/39        | 26/39           | PASS        |
| 6.5b              | BGL last20 CloSpan closed 138 / raw 150                            | 138/150      | 138/150         | PASS        |
| 6.5b              | Alibaba last3 CloSpan closed 15 / raw 16                           | 15/16        | 15/16           | PASS        |
| 6.10              | SCANIA matched (valid) top inference HR = 1.60                     | 1.60         | 1.6             | PASS        |
| 6.10              | SCANIA matched (valid) top HR CI = [1.35, 1.91]                    | [1.35, 1.91] | [1.348, 1.911]  | PASS        |
| 6.4               | kelmarsh within-entity-perm BY-significant = tested count          | 12/12        | 12/12           | PASS        |
| 6.4               | penmanshiel within-entity-perm BY-significant = tested count       | 12/12        | 12/12           | PASS        |
| 6.4               | Post-selection Azure last10: 379 BH-sig / 815 mined (46%)          | 379/815      | 379/815         | PASS        |
| 6.4               | Post-selection Azure last10 BY: 241/815 (30%)                      | 241          | 241             | PASS        |
| 6.4               | Post-selection SCANIA last20: 0 BH-sig / 37,797 mined (0%)         | 0/37797      | 0/37797         | PASS        |
| 6.3               | Azure last10 count-preserving order effect ≈ +1.09                 | +1.09        | 1.09            | PASS        |
| 6.3               | Azure last5 count-preserving order effect ≈ +0.52                  | +0.52        | 0.52            | PASS        |
| 6.3               | Alibaba last3 count-preserving order effect ≈ 0 (null)             | ≈0           | -0.02           | PASS        |
| 5.3               | Azure 24h combined AUROC = 0.996                                   | 0.996        | 0.996           | PASS        |
| 5.3               | Azure 24h combined AUPRC = 0.988                                   | 0.988        | 0.988           | PASS        |
| 5.3               | Azure last5 combined AUROC = 0.81                                  | 0.81         | 0.81            | PASS        |
| 5.3               | Azure last5 combined AUPRC = 0.72                                  | 0.72         | 0.72            | PASS        |
| 5.3               | Azure last10 combined AUROC = 0.696                                | 0.696        | 0.696           | PASS        |
| 5.3               | Azure last10 combined AUPRC = 0.576                                | 0.576        | 0.576           | PASS        |
| 5.3               | Azure last5 itemsets_only AUROC = 0.754                            | 0.754        | 0.754           | PASS        |
| 5.3               | Azure last5 itemsets_only AUPRC = 0.563                            | 0.563        | 0.563           | PASS        |
| 5.3               | Alibaba last3 combined AUROC = 0.813                               | 0.813        | 0.813           | PASS        |
| 5.3               | Alibaba last3 combined AUPRC = 0.631                               | 0.631        | 0.631           | PASS        |
| 5.3               | Alibaba last5 combined AUROC = 0.741                               | 0.741        | 0.741           | PASS        |
| 5.3               | Alibaba last5 combined AUPRC = 0.574                               | 0.574        | 0.574           | PASS        |
| 5.3               | Alibaba last10 combined AUROC = 0.741                              | 0.741        | 0.741           | PASS        |
| 5.3               | Alibaba last10 combined AUPRC = 0.593                              | 0.593        | 0.593           | PASS        |
| 5.3               | BGL last20 combined AUROC = 0.512                                  | 0.512        | 0.512           | PASS        |
| 5.3               | BGL last20 combined AUPRC = 0.256                                  | 0.256        | 0.256           | PASS        |
| 5.3               | BGL last20 itemsets_only AUROC = 0.483                             | 0.483        | 0.483           | PASS        |
| 5.3               | BGL last20 itemsets_only AUPRC = 0.245                             | 0.245        | 0.245           | PASS        |
| 5.3               | SCANIA last10 combined AUROC = 0.596                               | 0.596        | 0.596           | PASS        |
| 5.3               | SCANIA last10 combined AUPRC = 0.154                               | 0.154        | 0.154           | PASS        |
| 5.3               | SCANIA last20 combined AUROC = 0.567                               | 0.567        | 0.567           | PASS        |
| 5.3               | SCANIA last20 combined AUPRC = 0.132                               | 0.132        | 0.132           | PASS        |
| 5.5               | Azure 24h itemsets significant: 6/6                                | 6/6          | 6/6             | PASS        |
| 5.5               | Azure 24h sequences significant: 7/7                               | 7/7          | 7/7             | PASS        |
| 5.5               | Azure last5 itemsets significant: 53/77                            | 53/77        | 53/77           | PASS        |
| 5.5               | Azure last10 sequences significant: 562/657                        | 562/657      | 562/657         | PASS        |
| 5.5               | Alibaba last3 itemsets significant: 6/10                           | 6/10         | 6/10            | PASS        |
| 5.5               | Alibaba last10 sequences significant: 59/109                       | 59/109       | 59/109          | PASS        |
| 5.6               | sensitivity last5 combined ms=0.02 AUROC = 0.815                   | 0.815        | 0.815           | PASS        |
| 5.6               | sensitivity last5 combined ms=0.15 AUROC = 0.774                   | 0.774        | 0.774           | PASS        |
| 5.6               | sensitivity last5 itemsets_only ms=0.02 AUROC = 0.761              | 0.761        | 0.761           | PASS        |
| 5.6               | sensitivity last10 combined ms=0.1 AUROC = 0.751                   | 0.751        | 0.751           | PASS        |
| 5.6               | sensitivity last10 itemsets_only ms=0.02 AUROC = 0.578             | 0.578        | 0.578           | PASS        |
| 5.6               | sensitivity 24h combined ms=0.02 AUROC = 0.996                     | 0.996        | 0.996           | PASS        |