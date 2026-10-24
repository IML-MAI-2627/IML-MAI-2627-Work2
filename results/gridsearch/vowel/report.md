## vowel

### kNN without reduction

#### kNN: k

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| k=3 | 40 | 0.9713 | 0.9788 | 0.0076 | 105.1462 |
| k=5 | 40 | 0.9538 | 0.9747 | 0.0090 | 103.6695 |
| k=7 | 40 | 0.9267 | 0.9727 | 0.0073 | 100.6392 |

#### kNN: distance

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| distance=canberra | 24 | 0.9475 | 0.9737 | 0.0097 | 103.0831 |
| distance=clark | 24 | 0.9462 | 0.9747 | 0.0102 | 102.8704 |
| distance=euclidean | 24 | 0.9529 | 0.9788 | 0.0064 | 103.2669 |
| distance=heom | 24 | 0.9529 | 0.9788 | 0.0067 | 103.2669 |
| distance=manhattan | 24 | 0.9535 | 0.9788 | 0.0068 | 103.2707 |

#### kNN: voting

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| voting=inverse_distance | 60 | 0.9635 | 0.9788 | 0.0080 | 103.1186 |
| voting=majority | 60 | 0.9377 | 0.9747 | 0.0080 | 103.1846 |

#### kNN: retention

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| retention=always | 30 | 0.9574 | 0.9788 | 0.0080 | 111.1111 |
| retention=different_class | 30 | 0.9559 | 0.9758 | 0.0084 | 100.4897 |
| retention=never | 30 | 0.9533 | 0.9758 | 0.0070 | 100.0000 |
| retention=oblivion_by_goodness | 30 | 0.9357 | 0.9778 | 0.0085 | 101.0056 |

#### kNN: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| kNN[distance=euclidean;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0154 | 0.9787 | 0.0059 | 111.1111 |
| kNN[distance=manhattan;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0121 | 0.9787 | 0.0063 | 111.1111 |
| kNN[distance=heom;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0154 | 0.9787 | 0.0074 | 111.1111 |
| kNN[distance=manhattan;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9778 | 0.0115 | 0.9778 | 0.0064 | 109.4388 |
| kNN[distance=euclidean;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9768 | 0.0143 | 0.9767 | 0.0063 | 109.3378 |
| kNN[distance=heom;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9768 | 0.0143 | 0.9767 | 0.0073 | 109.3378 |
| kNN[distance=euclidean;k=3;reduction=none;retention=never;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0052 | 100.0000 |
| kNN[distance=euclidean;k=3;reduction=none;retention=different_class;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0060 | 100.2694 |
| kNN[distance=manhattan;k=3;reduction=none;retention=different_class;voting=inverse_distance] | 0.9758 | 0.0098 | 0.9757 | 0.0060 | 100.2694 |
| kNN[distance=heom;k=3;reduction=none;retention=never;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0066 | 100.0000 |

### SVM without reduction

#### SVM: kernel

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| kernel=linear | 8 | 0.6614 | 0.7919 | 0.0114 | 100.0000 |
| kernel=poly | 144 | 0.6881 | 0.9909 | 0.0127 | 100.0000 |
| kernel=rbf | 16 | 0.7290 | 0.9859 | 0.0162 | 100.0000 |

#### SVM: class_weight

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| class_weight=None | 84 | 0.6907 | 0.9909 | 0.0129 | 100.0000 |
| class_weight=balanced | 84 | 0.6907 | 0.9909 | 0.0130 | 100.0000 |

#### SVM: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Support Vector Machine[C=1.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0086 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0086 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0088 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0088 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0095 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0103 | 100.0000 |
| Support Vector Machine[C=0.1;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0085 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=0.0;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0086 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=None;coef0=0.0;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0086 | 100.0000 |
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0087 | 100.0000 |

### RF without reduction

#### RF: criterion

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| criterion=entropy | 48 | 0.9523 | 0.9798 | 0.4960 | 100.0000 |
| criterion=gini | 48 | 0.9421 | 0.9758 | 0.3235 | 100.0000 |

#### RF: n_estimators

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| n_estimators=50 | 24 | 0.9440 | 0.9677 | 0.1286 | 100.0000 |
| n_estimators=100 | 24 | 0.9456 | 0.9778 | 0.2521 | 100.0000 |
| n_estimators=200 | 24 | 0.9491 | 0.9798 | 0.5029 | 100.0000 |
| n_estimators=300 | 24 | 0.9500 | 0.9788 | 0.7555 | 100.0000 |

#### RF: max_features

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| max_features=None | 48 | 0.9271 | 0.9475 | 0.5823 | 100.0000 |
| max_features=sqrt | 48 | 0.9673 | 0.9798 | 0.2373 | 100.0000 |

#### RF: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9798 | 0.0126 | 0.9797 | 0.3450 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9798 | 0.0126 | 0.9797 | 0.3582 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9788 | 0.0138 | 0.9787 | 0.5217 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9788 | 0.0138 | 0.9787 | 0.5355 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=10;max_features=sqrt;min_samples_leaf=1;n_estimators=100;n_jobs=1;random_state=42] | 0.9778 | 0.0093 | 0.9777 | 0.1722 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=100;n_jobs=1;random_state=42] | 0.9758 | 0.0128 | 0.9755 | 0.1728 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=100;n_jobs=1;random_state=42] | 0.9758 | 0.0128 | 0.9755 | 0.1758 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9758 | 0.0136 | 0.9758 | 0.3516 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9758 | 0.0136 | 0.9758 | 0.3633 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9747 | 0.0167 | 0.9747 | 0.2444 | 100.0000 |

### Best kNN vs best SVM vs best RF (no reduction)

| config | accuracy | macro_f1 | time | storage |
|---|---|---|---|---|
| Support Vector Machine[C=1.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.9909 | 0.0086 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9798 | 0.9797 | 0.3450 | 100.0000 |
| kNN[distance=euclidean;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.9787 | 0.0059 | 111.1111 |

### Instance reduction on the best kNN

All variants on the best kNN:

| config | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| kNN[distance=euclidean;k=3;reduction=ennth;retention=always;threshold=0.0;voting=inverse_distance] | 0.9515 | 0.9512 | 0.0558 | 0.0066 | 108.1481 |
| kNN[distance=euclidean;k=3;reduction=ennth;retention=always;threshold=0.5;voting=inverse_distance] | 0.9475 | 0.9467 | 0.0557 | 0.0066 | 107.9910 |
| kNN[distance=euclidean;k=3;reduction=del;retention=always;voting=inverse_distance] | 0.9414 | 0.9411 | 0.7455 | 0.0037 | 45.3086 |
| kNN[distance=euclidean;k=3;reduction=fcnn;retention=always;version=2;voting=inverse_distance] | 0.9414 | 0.9413 | 0.0625 | 0.0033 | 40.7183 |
| kNN[distance=euclidean;k=3;reduction=fcnn;retention=always;version=1;voting=inverse_distance] | 0.9384 | 0.9381 | 0.0672 | 0.0034 | 41.0325 |
| kNN[distance=euclidean;k=3;reduction=drop;retention=always;version=3;voting=inverse_distance] | 0.9101 | 0.9098 | 0.5889 | 0.0040 | 42.7722 |
| kNN[distance=euclidean;k=3;reduction=ennth;retention=always;threshold=0.7;voting=inverse_distance] | 0.8980 | 0.8959 | 0.0552 | 0.0062 | 98.1594 |

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9788 | 0.9787 | 0.0059 | 0.0059 | 111.1111 |
| ennth | 0.9515 | 0.9512 | 0.0558 | 0.0066 | 108.1481 |
| del | 0.9414 | 0.9411 | 0.7455 | 0.0037 | 45.3086 |
| fcnn | 0.9414 | 0.9413 | 0.0625 | 0.0033 | 40.7183 |
| drop | 0.9101 | 0.9098 | 0.5889 | 0.0040 | 42.7722 |

Top kNN configurations with each technique, accuracy (storage):

| rank | kNN | none | fcnn | ennth | drop | del |
|---|---|---|---|---|---|---|
| 1 | kNN(3,euclidean,inverse_distance,always) | 0.9788 (111%) | 0.9414 (41%) | 0.9515 (108%) | 0.9101 (43%) | 0.9414 (45%) |
| 2 | kNN(3,manhattan,inverse_distance,always) | 0.9788 (111%) | 0.9384 (40%) | 0.9475 (108%) | 0.8970 (43%) | 0.9293 (45%) |
| 3 | kNN(3,heom,inverse_distance,always) | 0.9788 (111%) | 0.9323 (41%) | 0.9515 (108%) | 0.9101 (43%) | 0.9414 (45%) |

Retention vs reduction:

| variant | accuracy | time | solve_time | storage |
|---|---|---|---|---|
| ret=always | 0.9788 | 0.0059 | 0.0059 | 111.1111 |
| ret=oblivion_by_goodness | 0.9778 | 0.0064 | 0.0064 | 109.4388 |
| ret=never | 0.9758 | 0.0052 | 0.0052 | 100.0000 |
| ret=different_class | 0.9758 | 0.0060 | 0.0060 | 100.2694 |
| red=ennth | 0.9515 | 0.0558 | 0.0066 | 108.1481 |
| red=del | 0.9414 | 0.7455 | 0.0037 | 45.3086 |
| red=fcnn | 0.9414 | 0.0625 | 0.0033 | 40.7183 |
| red=drop | 0.9101 | 0.5889 | 0.0040 | 42.7722 |

### Instance reduction on the best SVM

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9909 | 0.9909 | 0.0086 | 0.0086 | 100.0000 |
| ennth | 0.9697 | 0.9690 | 0.0576 | 0.0085 | 97.0370 |
| fcnn | 0.9525 | 0.9519 | 0.0629 | 0.0036 | 29.6072 |
| del | 0.9313 | 0.9309 | 0.7454 | 0.0036 | 34.1975 |
| drop | 0.8889 | 0.8880 | 0.5882 | 0.0033 | 31.6611 |

### Instance reduction on the best RF

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9798 | 0.9797 | 0.3450 | 0.3450 | 100.0000 |
| ennth | 0.9535 | 0.9532 | 0.3935 | 0.3444 | 97.0370 |
| fcnn | 0.9061 | 0.9060 | 0.2252 | 0.1659 | 29.6072 |
| del | 0.9000 | 0.8996 | 0.9132 | 0.1714 | 34.1975 |
| drop | 0.8778 | 0.8772 | 0.7544 | 0.1695 | 31.6611 |

### kNN vs SVM vs RF with each reduction technique

### All tests

| section | question | metric | k | test | p_value | significant | winner / post-hoc |
|---|---|---|---|---|---|---|---|
| kNN | kNN k main effect | accuracy | 3 | Friedman + Iman-Davenport | 0 | True | k=3. Nemenyi (CD=1.05): not different from best: k=3, k=5 |
| kNN | kNN k best | accuracy | 3 | Friedman + Iman-Davenport | 0.192 | False | k=3 |
| kNN | kNN distance main effect | accuracy | 5 | Friedman + Iman-Davenport | 0.266 | False | distance=manhattan |
| kNN | kNN distance best | accuracy | 5 | Friedman + Iman-Davenport | 0.598 | False | distance=euclidean |
| kNN | kNN voting main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | voting=inverse_distance. R+=0, R-=55; paired t-test p=2.45e-06 (reference) |
| kNN | kNN voting best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.25 | False | voting=inverse_distance. R+=10.5, R-=34.5; paired t-test p=0.104 (reference) |
| kNN | kNN retention main effect | accuracy | 4 | Friedman + Iman-Davenport | 1.83e-07 | True | retention=always. Nemenyi (CD=1.48): not different from best: retention=always, retention=different_class, retention=never |
| kNN | kNN retention best | accuracy | 4 | Friedman + Iman-Davenport | 0.954 | False | retention=always |
| kNN | kNN top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.986 | False | #1 kNN(3,euclidean,inverse_distance,always) |
| SVM | SVM kernel main effect | accuracy | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kernel=rbf. Nemenyi (CD=1.05): not different from best: kernel=rbf |
| SVM | SVM kernel best | accuracy | 3 | Friedman + Iman-Davenport | 2.64e-06 | True | kernel=poly. Nemenyi (CD=1.05): not different from best: kernel=poly, kernel=rbf |
| SVM | SVM class_weight main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=None. R+=27.5, R-=27.5; paired t-test p=0.343 (reference) |
| SVM | SVM class_weight best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=None. R+=27.5, R-=27.5; paired t-test p=nan (reference) |
| SVM | SVM top-5 | accuracy | 5 | Friedman + Iman-Davenport | 1 | False | #1 SVM(poly,1.0,scale,5,-0.5,None) |
| RF | RF criterion main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0645 | False | criterion=entropy. R+=9, R-=46; paired t-test p=0.0516 (reference) |
| RF | RF criterion best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.219 | False | criterion=entropy. R+=14.5, R-=40.5; paired t-test p=0.269 (reference) |
| RF | RF n_estimators main effect | accuracy | 4 | Friedman + Iman-Davenport | 0.0464 | True | n_estimators=300. Nemenyi (CD=1.48): not different from best: n_estimators=300, n_estimators=200, n_estimators=100, n_estimators=50 |
| RF | RF n_estimators best | accuracy | 4 | Friedman + Iman-Davenport | 0.0813 | False | n_estimators=300 |
| RF | RF max_features main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | max_features=sqrt. R+=55, R-=0; paired t-test p=6.51e-06 (reference) |
| RF | RF max_features best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | max_features=sqrt. R+=55, R-=0; paired t-test p=3.04e-05 (reference) |
| RF | RF top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.945 | False | #1 RF(entropy,200,20,sqrt,1) |
| Models | kNN vs SVM vs RF | accuracy | 3 | Friedman + Iman-Davenport | 0.0321 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF, kNN |
| Models | kNN vs SVM vs RF: kNN vs SVM | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.16 | False | SVM. raw p=0.08008 |
| Models | kNN vs SVM vs RF: kNN vs RF | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.922 | False | RF. raw p=0.9219 |
| Models | kNN vs SVM vs RF | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Reduction | kNN reduction | accuracy | 5 | Friedman + Iman-Davenport | 1.28e-10 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, fcnn worse, drop worse |
| Reduction | kNN reduction | solve_time | 5 | Friedman + Iman-Davenport | 1.07e-15 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): fcnn better |
| Reduction | retention vs reduction | accuracy | 8 | Friedman + Iman-Davenport | 6.11e-17 | True | ret=always. Nemenyi (CD=3.32): not different from best: ret=always, ret=oblivion_by_goodness, ret=never, ret=different_class, red=ennth |
| Reduction | SVM reduction | accuracy | 5 | Friedman + Iman-Davenport | 2.13e-13 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, drop worse |
| Reduction | SVM reduction | solve_time | 5 | Friedman + Iman-Davenport | 2.45e-16 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): fcnn better, drop better, del better |
| Reduction | RF reduction | accuracy | 5 | Friedman + Iman-Davenport | 6.82e-12 | True | none. Bonferroni-Dunn vs none (CD=1.77): fcnn worse, del worse, drop worse |
| Reduction | RF reduction | solve_time | 5 | Friedman + Iman-Davenport | 9.5e-14 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): fcnn better, drop better, del better |
| Models | models with fcnn | accuracy | 3 | Friedman + Iman-Davenport | 0.00316 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, kNN |
| Models | models with fcnn | time | 3 | Friedman + Iman-Davenport | 6.87e-08 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, kNN |
| Models | models with ennth | accuracy | 3 | Friedman + Iman-Davenport | 0.098 | False | SVM |
| Models | models with ennth | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with drop | accuracy | 3 | Friedman + Iman-Davenport | 0.0391 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with drop | time | 3 | Friedman + Iman-Davenport | 6.87e-08 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, kNN |
| Models | models with del | accuracy | 3 | Friedman + Iman-Davenport | 0.192 | False | kNN |
| Models | models with del | time | 3 | Friedman + Iman-Davenport | 2.64e-06 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
