# Azure PdM: top mined patterns

Both mining passes at ``min_support = 0.05``. Sequences with ``survives_shuffle_null = True`` beat the within-window order-permutation null at that horizon.

## Top itemsets (Phase 3, FP-Growth)

### 1h

_no patterns above min_support_

### 6h

_no patterns above min_support_

### 24h (6 patterns)

| itemset                                       |   supp_fail |   supp_ctrl |   lift |   RR |   P(fail|patt) | survives   |
|:----------------------------------------------|------------:|------------:|-------:|-----:|---------------:|:-----------|
| software_error:error2 + software_error:error3 |       0.382 |       0     |   3.99 | 5.83 |          0.996 | True       |
| software_error:error5                         |       0.272 |       0.002 |   3.9  | 4.99 |          0.976 | True       |
| software_error:error2                         |       0.397 |       0.011 |   3.7  | 5.48 |          0.925 | True       |
| software_error:error3                         |       0.398 |       0.011 |   3.69 | 5.47 |          0.922 | True       |
| software_error:error4                         |       0.213 |       0.007 |   3.63 | 4.34 |          0.908 | True       |
| software_error:error1                         |       0.281 |       0.012 |   3.56 | 4.56 |          0.889 | True       |

### last5 (77 patterns)

| itemset                                                               |   supp_fail |   supp_ctrl |   lift |   RR |   P(fail|patt) | survives   |
|:----------------------------------------------------------------------|------------:|------------:|-------:|-----:|---------------:|:-----------|
| software_error:error3 + software_error:error4 + software_error:error5 |       0.052 |       0.005 |   3.12 | 3.24 |          0.78  | True       |
| software_error:error2 + software_error:error3 + software_error:error5 |       0.097 |       0.009 |   3.1  | 3.32 |          0.774 | True       |
| software_error:error2 + software_error:error4 + software_error:error5 |       0.067 |       0.008 |   2.94 | 3.08 |          0.735 | True       |
| software_error:error2 + software_error:error3 + software_error:error4 |       0.166 |       0.025 |   2.76 | 3.11 |          0.691 | True       |
| software_error:error1 + software_error:error2 + software_error:error3 |       0.191 |       0.035 |   2.59 | 2.97 |          0.648 | True       |
| software_error:error4 + software_error:error5                         |       0.152 |       0.028 |   2.58 | 2.87 |          0.646 | True       |

### last10 (730 patterns)

| itemset                                                                                                                                   |   supp_fail |   supp_ctrl |   lift |   RR |   P(fail|patt) | survives   |
|:------------------------------------------------------------------------------------------------------------------------------------------|------------:|------------:|-------:|-----:|---------------:|:-----------|
| maintenance:comp1 + software_error:error2 + software_error:error3 + software_error:error4 + software_error:error5                         |       0.098 |       0.021 |   2.43 | 2.59 |          0.608 | True       |
| maintenance:comp1 + software_error:error1 + software_error:error2 + software_error:error3 + software_error:error4 + software_error:error5 |       0.051 |       0.012 |   2.34 | 2.41 |          0.585 | True       |
| maintenance:comp1 + maintenance:comp2 + software_error:error2 + software_error:error4 + software_error:error5                             |       0.055 |       0.014 |   2.25 | 2.32 |          0.562 | True       |
| maintenance:comp1 + maintenance:comp2 + maintenance:comp3 + maintenance:comp4 + software_error:error2 + software_error:error3             |       0.058 |       0.015 |   2.23 | 2.31 |          0.558 | True       |
| maintenance:comp1 + maintenance:comp2 + software_error:error3 + software_error:error4 + software_error:error5                             |       0.055 |       0.015 |   2.22 | 2.29 |          0.554 | True       |
| software_error:error1 + software_error:error2 + software_error:error3 + software_error:error4 + software_error:error5                     |       0.114 |       0.031 |   2.21 | 2.36 |          0.552 | True       |

## Top sequences (Phase 4, PrefixSpan)

### 1h (3 patterns)

| sequence              |   supp_fail |   seq_lift |   iset_lift |   order_gain |   P(fail|patt) | survives_shuf   |
|:----------------------|------------:|-----------:|------------:|-------------:|---------------:|:----------------|
| software_error:error5 |       0.001 |       4    |        4    |            0 |          1     | False           |
| software_error:error4 |       0.001 |       2    |        2    |            0 |          0.5   | False           |
| software_error:error1 |       0.001 |       1.33 |        1.33 |            0 |          0.333 | False           |

### 6h (5 patterns)

| sequence              |   supp_fail |   seq_lift |   iset_lift |   order_gain |   P(fail|patt) | survives_shuf   |
|:----------------------|------------:|-----------:|------------:|-------------:|---------------:|:----------------|
| software_error:error4 |       0.005 |       1.45 |        1.45 |            0 |          0.364 | False           |
| software_error:error3 |       0.007 |       1.18 |        1.18 |            0 |          0.294 | False           |
| software_error:error5 |       0.001 |       1    |        1    |            0 |          0.25  | False           |
| software_error:error2 |       0.004 |       0.75 |        0.75 |            0 |          0.188 | False           |
| software_error:error1 |       0.001 |       0.27 |        0.27 |            0 |          0.067 | False           |

### 24h (7 patterns)

| sequence                                       |   supp_fail |   seq_lift |   iset_lift |   order_gain |   P(fail|patt) | survives_shuf   |
|:-----------------------------------------------|------------:|-----------:|------------:|-------------:|---------------:|:----------------|
| software_error:error3 -> software_error:error2 |       0.071 |       4    |        3.99 |         0.01 |          1     | False           |
| software_error:error2 -> software_error:error3 |       0.315 |       3.98 |        3.99 |        -0    |          0.996 | False           |
| software_error:error5                          |       0.272 |       3.9  |        3.9  |         0    |          0.976 | False           |
| software_error:error2                          |       0.397 |       3.7  |        3.7  |         0    |          0.925 | False           |
| software_error:error3                          |       0.398 |       3.69 |        3.69 |         0    |          0.922 | False           |
| software_error:error4                          |       0.213 |       3.63 |        3.63 |         0    |          0.908 | False           |

### last5 (67 patterns)

| sequence                                                                |   supp_fail |   seq_lift |   iset_lift |   order_gain |   P(fail|patt) | survives_shuf   |
|:------------------------------------------------------------------------|------------:|-----------:|------------:|-------------:|---------------:|:----------------|
| maintenance:comp4 -> software_error:error2 -> software_error:error3     |       0.092 |       3.73 |        2.22 |         1.51 |          0.932 | True            |
| software_error:error4 -> software_error:error2 -> software_error:error3 |       0.105 |       3.71 |        2.76 |         0.95 |          0.929 | True            |
| software_error:error2 -> software_error:error2 -> software_error:error3 |       0.11  |       3.64 |        2.2  |         1.45 |          0.911 | True            |
| maintenance:comp1 -> software_error:error2 -> software_error:error3     |       0.085 |       3.55 |        2.21 |         1.34 |          0.887 | True            |
| software_error:error3 -> software_error:error2 -> software_error:error3 |       0.109 |       3.52 |        2.2  |         1.33 |          0.88  | True            |
| software_error:error1 -> software_error:error2 -> software_error:error3 |       0.127 |       3.51 |        2.59 |         0.92 |          0.879 | True            |

### last10 (657 patterns)

| sequence                                                                                         |   supp_fail |   seq_lift |   iset_lift |   order_gain |   P(fail|patt) | survives_shuf   |
|:-------------------------------------------------------------------------------------------------|------------:|-----------:|------------:|-------------:|---------------:|:----------------|
| maintenance:comp1 -> software_error:error4 -> software_error:error2 -> software_error:error3     |       0.067 |       3.23 |        1.71 |         1.52 |          0.806 | True            |
| software_error:error1 -> maintenance:comp4 -> software_error:error2 -> software_error:error3     |       0.07  |       3.2  |        1.75 |         1.45 |          0.8   | True            |
| software_error:error4 -> software_error:error3 -> software_error:error2 -> software_error:error3 |       0.061 |       3.05 |        1.73 |         1.32 |          0.763 | True            |
| software_error:error3 -> software_error:error4 -> software_error:error2 -> software_error:error3 |       0.069 |       3.04 |        1.73 |         1.31 |          0.761 | True            |
| maintenance:comp3 -> software_error:error1 -> software_error:error2 -> software_error:error3     |       0.106 |       3.04 |        1.69 |         1.35 |          0.76  | True            |
| software_error:error4 -> software_error:error2 -> software_error:error2 -> software_error:error3 |       0.075 |       3.03 |        1.73 |         1.29 |          0.757 | True            |

## Where order matters most (top 10 by order_gain)

| horizon   | sequence                                                                                               |   seq_lift |   iset_lift |   order_gain |   supp_fail |
|:----------|:-------------------------------------------------------------------------------------------------------|-----------:|------------:|-------------:|------------:|
| last10    | maintenance:comp1 -> software_error:error4 -> software_error:error2 -> software_error:error3           |       3.23 |        1.71 |         1.52 |       0.067 |
| last5     | maintenance:comp4 -> software_error:error2 -> software_error:error3                                    |       3.73 |        2.22 |         1.51 |       0.092 |
| last10    | maintenance:comp3 -> software_error:error4 -> software_error:error2 -> software_error:error3           |       3.02 |        1.54 |         1.47 |       0.062 |
| last10    | software_error:error1 -> component_replacement:comp1 -> software_error:error2 -> software_error:error3 |       2.88 |        1.41 |         1.47 |       0.055 |
| last10    | software_error:error1 -> maintenance:comp4 -> software_error:error2 -> software_error:error3           |       3.2  |        1.75 |         1.45 |       0.07  |
| last5     | software_error:error2 -> software_error:error2 -> software_error:error3                                |       3.64 |        2.2  |         1.45 |       0.11  |
| last10    | maintenance:comp2 -> software_error:error1 -> software_error:error2 -> software_error:error3           |       2.98 |        1.54 |         1.44 |       0.059 |
| last10    | software_error:error2 -> maintenance:comp3 -> software_error:error2 -> software_error:error3           |       3.02 |        1.6  |         1.42 |       0.066 |
| last5     | maintenance:comp3 -> software_error:error2 -> software_error:error3                                    |       3.23 |        1.84 |         1.39 |       0.085 |
| last10    | maintenance:comp4 -> maintenance:comp4 -> software_error:error2 -> software_error:error3               |       3.02 |        1.64 |         1.38 |       0.058 |
