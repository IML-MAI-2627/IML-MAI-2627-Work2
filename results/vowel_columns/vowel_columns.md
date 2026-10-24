# vowel: which columns to keep

Best vowel kNN of the grid search: `{'distance': 'euclidean', 'k': '3', 'reduction': 'none', 'retention': 'always', 'voting': 'inverse_distance'}`. Only the dropped columns change.

| Variant | Mean accuracy | Std | Mean rank | Storage (%) |
|---|---|---|---|---|
| A: 10 numeric | 0.9788 | 0.0181 | 2.00 | 111.1 |
| B: + Sex (current) | 0.9788 | 0.0154 | 2.00 | 111.1 |
| C: + Sex + Speaker_Number | 0.9788 | 0.0175 | 2.00 | 111.1 |

Friedman + Iman-Davenport over the 10 folds: F_F = 0.000, p = 1 (not significant at alpha = 0.05).

| Pair | Mean difference (second - first) | Wilcoxon p | Holm-adjusted p | Significant |
|---|---|---|---|---|
| A: 10 numeric vs B: + Sex (current) | +0.0000 | 1.0000 | 1.0000 | no |
| B: + Sex (current) vs C: + Sex + Speaker_Number | +0.0000 | 1.0000 | 1.0000 | no |
| A: 10 numeric vs C: + Sex + Speaker_Number | +0.0000 | 1.0000 | 1.0000 | no |

Per-fold accuracy:

| fold | A: 10 numeric | B: + Sex (current) | C: + Sex + Speaker_Number |
|---|---|---|---|
| 0 | 0.9697 | 0.9697 | 0.9697 |
| 1 | 0.9394 | 0.9495 | 0.9394 |
| 2 | 0.9798 | 0.9798 | 0.9899 |
| 3 | 1.0 | 1.0 | 0.9798 |
| 4 | 0.9697 | 0.9697 | 0.9798 |
| 5 | 0.9697 | 0.9697 | 0.9697 |
| 6 | 1.0 | 1.0 | 1.0 |
| 7 | 0.9899 | 0.9798 | 0.9798 |
| 8 | 0.9798 | 0.9798 | 1.0 |
| 9 | 0.9899 | 0.9899 | 0.9798 |

With the 10 numeric features only (variant A, HEOM), the nearest training case of a test case is the same speaker in 972 of 990 test cases (98.2%), and the same speaker and vowel in 965 (97.5%).
