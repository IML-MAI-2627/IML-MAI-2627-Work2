# Selection bias (leave-one-fold-out)

| Dataset | Model | Configurations | Best mean over the 10 folds (reported) | Leave-one-fold-out estimate | Estimated bias | Distinct winners in the 10 rounds | Rounds that chose the reported winner |
|---|---|---|---|---|---|---|---|
| splice | kNN | 120 | 0.8439 | 0.8439 | +0.0000 | 2 | 9 |
| splice | SVM | 168 | 0.9690 | 0.9680 | +0.0009 | 3 | 8 |
| splice | RF | 96 | 0.9690 | 0.9690 | +0.0000 | 1 | 10 |
| vowel | kNN | 120 | 0.9788 | 0.9747 | +0.0040 | 2 | 7 |
| vowel | SVM | 168 | 0.9909 | 0.9879 | +0.0030 | 4 | 3 |
| vowel | RF | 96 | 0.9798 | 0.9758 | +0.0040 | 3 | 8 |
