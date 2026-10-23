## vowel

### kNN without reduction

#### kNN: k

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| k=3 | 40 | 0.9713 | 0.9788 | 0.0161 | 105.1462 |
| k=5 | 40 | 0.9538 | 0.9747 | 0.0183 | 103.6695 |
| k=7 | 40 | 0.9267 | 0.9727 | 0.0172 | 100.6392 |

#### kNN: distance

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| distance=canberra | 24 | 0.9475 | 0.9737 | 0.0192 | 103.0831 |
| distance=clark | 24 | 0.9462 | 0.9747 | 0.0220 | 102.8704 |
| distance=euclidean | 24 | 0.9529 | 0.9788 | 0.0152 | 103.2669 |
| distance=heom | 24 | 0.9529 | 0.9788 | 0.0150 | 103.2669 |
| distance=manhattan | 24 | 0.9535 | 0.9788 | 0.0148 | 103.2707 |

#### kNN: voting

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| voting=inverse_distance | 60 | 0.9635 | 0.9788 | 0.0172 | 103.1186 |
| voting=majority | 60 | 0.9377 | 0.9747 | 0.0173 | 103.1846 |

#### kNN: retention

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| retention=always | 30 | 0.9574 | 0.9788 | 0.0182 | 111.1111 |
| retention=different_class | 30 | 0.9559 | 0.9758 | 0.0176 | 100.4897 |
| retention=never | 30 | 0.9533 | 0.9758 | 0.0151 | 100.0000 |
| retention=oblivion_by_goodness | 30 | 0.9357 | 0.9778 | 0.0181 | 101.0056 |

#### kNN: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| kNN[distance=heom;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0154 | 0.9787 | 0.0144 | 111.1111 |
| kNN[distance=manhattan;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0121 | 0.9787 | 0.0146 | 111.1111 |
| kNN[distance=euclidean;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.0154 | 0.9787 | 0.0149 | 111.1111 |
| kNN[distance=manhattan;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9778 | 0.0115 | 0.9778 | 0.0151 | 109.4388 |
| kNN[distance=heom;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9768 | 0.0143 | 0.9767 | 0.0148 | 109.3378 |
| kNN[distance=euclidean;k=3;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.9768 | 0.0143 | 0.9767 | 0.0149 | 109.3378 |
| kNN[distance=heom;k=3;reduction=none;retention=never;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0124 | 100.0000 |
| kNN[distance=euclidean;k=3;reduction=none;retention=never;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0128 | 100.0000 |
| kNN[distance=heom;k=3;reduction=none;retention=different_class;voting=inverse_distance] | 0.9758 | 0.0159 | 0.9757 | 0.0140 | 100.2694 |
| kNN[distance=manhattan;k=3;reduction=none;retention=different_class;voting=inverse_distance] | 0.9758 | 0.0098 | 0.9757 | 0.0143 | 100.2694 |

### SVM without reduction

#### SVM: kernel

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| kernel=linear | 8 | 0.6614 | 0.7919 | 0.0298 | 100.0000 |
| kernel=poly | 144 | 0.6881 | 0.9909 | 0.0336 | 100.0000 |
| kernel=rbf | 16 | 0.7290 | 0.9859 | 0.0372 | 100.0000 |

#### SVM: class_weight

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| class_weight=None | 84 | 0.6907 | 0.9909 | 0.0341 | 100.0000 |
| class_weight=balanced | 84 | 0.6907 | 0.9909 | 0.0333 | 100.0000 |

#### SVM: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Support Vector Machine[C=3.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0218 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0218 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0224 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0224 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0225 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.0088 | 0.9909 | 0.0227 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=None;coef0=0.0;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0216 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=None;coef0=0.0;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0217 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=0.0;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0218 | 100.0000 |
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9899 | 0.0082 | 0.9898 | 0.0221 | 100.0000 |

### RF without reduction

#### RF: criterion

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| criterion=entropy | 48 | 0.9519 | 0.9798 | 1.1335 | 100.0000 |
| criterion=gini | 48 | 0.9423 | 0.9747 | 0.7389 | 100.0000 |

#### RF: n_estimators

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| n_estimators=50 | 24 | 0.9442 | 0.9667 | 0.3015 | 100.0000 |
| n_estimators=100 | 24 | 0.9447 | 0.9747 | 0.5932 | 100.0000 |
| n_estimators=200 | 24 | 0.9489 | 0.9788 | 1.1595 | 100.0000 |
| n_estimators=300 | 24 | 0.9505 | 0.9798 | 1.6906 | 100.0000 |

#### RF: max_features

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| max_features=None | 48 | 0.9273 | 0.9495 | 1.2826 | 100.0000 |
| max_features=sqrt | 48 | 0.9669 | 0.9798 | 0.5898 | 100.0000 |

#### RF: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9798 | 0.0126 | 0.9798 | 1.2231 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9798 | 0.0126 | 0.9798 | 1.2622 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9788 | 0.0130 | 0.9786 | 0.8213 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9788 | 0.0130 | 0.9786 | 0.8214 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=100;n_jobs=1;random_state=42] | 0.9747 | 0.0137 | 0.9748 | 0.3082 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=100;n_jobs=1;random_state=42] | 0.9747 | 0.0137 | 0.9748 | 0.3661 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=200;n_jobs=1;random_state=42] | 0.9747 | 0.0167 | 0.9747 | 0.5853 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9747 | 0.0145 | 0.9748 | 0.8769 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9747 | 0.0145 | 0.9748 | 0.8793 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=10;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9747 | 0.0071 | 0.9747 | 1.2890 | 100.0000 |

### Best kNN vs best SVM vs best RF (no reduction)

| config | accuracy | macro_f1 | time | storage |
|---|---|---|---|---|
| Support Vector Machine[C=3.0;class_weight=None;coef0=-0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9909 | 0.9909 | 0.0218 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9798 | 0.9798 | 1.2231 | 100.0000 |
| kNN[distance=heom;k=3;reduction=none;retention=always;voting=inverse_distance] | 0.9788 | 0.9787 | 0.0144 | 111.1111 |

### Instance reduction on the best kNN

All variants on the best kNN:

| config | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| kNN[distance=heom;k=3;reduction=eenth;retention=always;threshold=0.0;voting=inverse_distance] | 0.9515 | 0.9512 | 0.1232 | 0.0149 | 108.1481 |
| kNN[distance=heom;k=3;reduction=eenth;retention=always;threshold=0.5;voting=inverse_distance] | 0.9475 | 0.9467 | 0.1248 | 0.0171 | 107.9910 |
| kNN[distance=heom;k=3;reduction=del;retention=always;voting=inverse_distance] | 0.9414 | 0.9411 | 2.0326 | 0.0081 | 45.3086 |
| kNN[distance=heom;k=3;reduction=fcnn;retention=always;version=1;voting=inverse_distance] | 0.9404 | 0.9400 | 0.1884 | 0.0094 | 41.1448 |
| kNN[distance=heom;k=3;reduction=fcnn;retention=always;version=2;voting=inverse_distance] | 0.9323 | 0.9322 | 0.1873 | 0.0068 | 40.6397 |
| kNN[distance=heom;k=3;reduction=drop;retention=always;version=3;voting=inverse_distance] | 0.9101 | 0.9098 | 1.5086 | 0.0081 | 42.7722 |
| kNN[distance=heom;k=3;reduction=eenth;retention=always;threshold=0.7;voting=inverse_distance] | 0.8980 | 0.8959 | 0.1230 | 0.0131 | 98.1594 |

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9788 | 0.9787 | 0.0144 | 0.0144 | 111.1111 |
| eenth | 0.9515 | 0.9512 | 0.1232 | 0.0149 | 108.1481 |
| del | 0.9414 | 0.9411 | 2.0326 | 0.0081 | 45.3086 |
| fcnn | 0.9404 | 0.9400 | 0.1884 | 0.0094 | 41.1448 |
| drop | 0.9101 | 0.9098 | 1.5086 | 0.0081 | 42.7722 |

Top kNN configurations with each technique, accuracy (storage):

| rank | kNN | none | fcnn | eenth | drop | del |
|---|---|---|---|---|---|---|
| 1 | kNN(3,heom,inverse_distance,always) | 0.9788 (111%) | 0.9404 (41%) | 0.9515 (108%) | 0.9101 (43%) | 0.9414 (45%) |
| 2 | kNN(3,manhattan,inverse_distance,always) | 0.9788 (111%) | 0.9394 (42%) | 0.9475 (108%) | 0.8970 (43%) | 0.9293 (45%) |
| 3 | kNN(3,euclidean,inverse_distance,always) | 0.9788 (111%) | 0.9384 (41%) | 0.9515 (108%) | 0.9101 (43%) | 0.9414 (45%) |

Retention vs reduction:

| variant | accuracy | time | solve_time | storage |
|---|---|---|---|---|
| ret=always | 0.9788 | 0.0144 | 0.0144 | 111.1111 |
| ret=oblivion_by_goodness | 0.9778 | 0.0151 | 0.0151 | 109.4388 |
| ret=never | 0.9758 | 0.0124 | 0.0124 | 100.0000 |
| ret=different_class | 0.9758 | 0.0140 | 0.0140 | 100.2694 |
| red=eenth | 0.9515 | 0.1232 | 0.0149 | 108.1481 |
| red=del | 0.9414 | 2.0326 | 0.0081 | 45.3086 |
| red=fcnn | 0.9404 | 0.1884 | 0.0094 | 41.1448 |
| red=drop | 0.9101 | 1.5086 | 0.0081 | 42.7722 |

### Instance reduction on the best SVM

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9909 | 0.9909 | 0.0218 | 0.0218 | 100.0000 |
| eenth | 0.9697 | 0.9690 | 0.1290 | 0.0207 | 97.0370 |
| fcnn | 0.9455 | 0.9448 | 0.1865 | 0.0075 | 30.0337 |
| del | 0.9313 | 0.9309 | 2.0325 | 0.0080 | 34.1975 |
| drop | 0.8889 | 0.8880 | 1.5079 | 0.0073 | 31.6611 |

### Instance reduction on the best RF

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9798 | 0.9798 | 1.2231 | 1.2231 | 100.0000 |
| eenth | 0.9505 | 0.9500 | 1.4812 | 1.3729 | 97.0370 |
| del | 0.8980 | 0.8975 | 2.7514 | 0.7268 | 34.1975 |
| fcnn | 0.8960 | 0.8948 | 0.8511 | 0.6721 | 30.0337 |
| drop | 0.8697 | 0.8689 | 2.3337 | 0.8331 | 31.6611 |

### kNN vs SVM vs RF with each reduction technique

### All tests

| section | question | metric | k | test | p_value | significant | winner / post-hoc |
|---|---|---|---|---|---|---|---|
| kNN | kNN k main effect | accuracy | 3 | Friedman + Iman-Davenport | 0 | True | k=3. Nemenyi (CD=1.05): not different from best: k=3, k=5 |
| kNN | kNN k best | accuracy | 3 | Friedman + Iman-Davenport | 0.192 | False | k=3 |
| kNN | kNN distance main effect | accuracy | 5 | Friedman + Iman-Davenport | 0.266 | False | distance=manhattan |
| kNN | kNN distance best | accuracy | 5 | Friedman + Iman-Davenport | 0.598 | False | distance=heom |
| kNN | kNN voting main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | voting=inverse_distance. R+=0, R-=55; paired t-test p=2.45e-06 (reference) |
| kNN | kNN voting best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.25 | False | voting=inverse_distance. R+=10.5, R-=34.5; paired t-test p=0.104 (reference) |
| kNN | kNN retention main effect | accuracy | 4 | Friedman + Iman-Davenport | 1.83e-07 | True | retention=always. Nemenyi (CD=1.48): not different from best: retention=always, retention=different_class, retention=never |
| kNN | kNN retention best | accuracy | 4 | Friedman + Iman-Davenport | 0.954 | False | retention=always |
| kNN | kNN top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.986 | False | #1 kNN(3,heom,inverse_distance,always) |
| SVM | SVM kernel main effect | accuracy | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kernel=rbf. Nemenyi (CD=1.05): not different from best: kernel=rbf |
| SVM | SVM kernel best | accuracy | 3 | Friedman + Iman-Davenport | 2.64e-06 | True | kernel=poly. Nemenyi (CD=1.05): not different from best: kernel=poly, kernel=rbf |
| SVM | SVM class_weight main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=None. R+=27.5, R-=27.5; paired t-test p=0.343 (reference) |
| SVM | SVM class_weight best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=None. R+=27.5, R-=27.5; paired t-test p=nan (reference) |
| SVM | SVM top-5 | accuracy | 5 | Friedman + Iman-Davenport | 1 | False | #1 SVM(poly,3.0,scale,5,-0.5,None) |
| RF | RF criterion main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0488 | True | criterion=entropy. R+=8, R-=47; paired t-test p=0.049 (reference) |
| RF | RF criterion best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.312 | False | criterion=entropy. R+=17, R-=38; paired t-test p=0.213 (reference) |
| RF | RF n_estimators main effect | accuracy | 4 | Friedman + Iman-Davenport | 0.00708 | True | n_estimators=300. Nemenyi (CD=1.48): not different from best: n_estimators=300, n_estimators=200, n_estimators=100 |
| RF | RF n_estimators best | accuracy | 4 | Friedman + Iman-Davenport | 0.102 | False | n_estimators=300 |
| RF | RF max_features main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | max_features=sqrt. R+=55, R-=0; paired t-test p=1.35e-05 (reference) |
| RF | RF max_features best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00195 | True | max_features=sqrt. R+=55, R-=0; paired t-test p=0.000131 (reference) |
| RF | RF top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.957 | False | #1 RF(entropy,300,None,sqrt,1) |
| Models | kNN vs SVM vs RF | accuracy | 3 | Friedman + Iman-Davenport | 0.0321 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, kNN, RF |
| Models | kNN vs SVM vs RF: kNN vs SVM | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.16 | False | SVM. raw p=0.08008 |
| Models | kNN vs SVM vs RF: kNN vs RF | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.977 | False | RF. raw p=0.9766 |
| Models | kNN vs SVM vs RF | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Reduction | kNN reduction | accuracy | 5 | Friedman + Iman-Davenport | 7.44e-12 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, fcnn worse, drop worse |
| Reduction | kNN reduction | solve_time | 5 | Friedman + Iman-Davenport | 3.48e-08 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): fcnn better, drop better |
| Reduction | retention vs reduction | accuracy | 8 | Friedman + Iman-Davenport | 2.57e-19 | True | ret=always. Nemenyi (CD=3.32): not different from best: ret=always, ret=oblivion_by_goodness, ret=never, ret=different_class, red=eenth |
| Reduction | SVM reduction | accuracy | 5 | Friedman + Iman-Davenport | 8.51e-15 | True | none. Bonferroni-Dunn vs none (CD=1.77): fcnn worse, del worse, drop worse |
| Reduction | SVM reduction | solve_time | 5 | Friedman + Iman-Davenport | 4.79e-20 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better, fcnn better, del better |
| Reduction | RF reduction | accuracy | 5 | Friedman + Iman-Davenport | 7.19e-13 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, fcnn worse, drop worse |
| Reduction | RF reduction | solve_time | 5 | Friedman + Iman-Davenport | 5.92e-09 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): fcnn better |
| Models | models with fcnn | accuracy | 3 | Friedman + Iman-Davenport | 7.94e-07 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, kNN |
| Models | models with fcnn | time | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with eenth | accuracy | 3 | Friedman + Iman-Davenport | 0.0589 | False | SVM |
| Models | models with eenth | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with drop | accuracy | 3 | Friedman + Iman-Davenport | 0.0751 | False | kNN |
| Models | models with drop | time | 3 | Friedman + Iman-Davenport | 3.81e-06 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with del | accuracy | 3 | Friedman + Iman-Davenport | 0.192 | False | kNN |
| Models | models with del | time | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
