## splice

### kNN without reduction

#### kNN: k

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| k=3 | 40 | 0.7978 | 0.8025 | 0.0786 | 103.9407 |
| k=5 | 40 | 0.8172 | 0.8248 | 0.0803 | 102.2820 |
| k=7 | 40 | 0.8292 | 0.8439 | 0.0874 | 100.6901 |

#### kNN: distance

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| distance=canberra | 24 | 0.8147 | 0.8439 | 0.0861 | 102.3042 |
| distance=clark | 24 | 0.8147 | 0.8439 | 0.0784 | 102.3042 |
| distance=euclidean | 24 | 0.8147 | 0.8439 | 0.0829 | 102.3044 |
| distance=heom | 24 | 0.8147 | 0.8439 | 0.0834 | 102.3044 |
| distance=manhattan | 24 | 0.8147 | 0.8439 | 0.0796 | 102.3042 |

#### kNN: voting

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| voting=inverse_distance | 60 | 0.8156 | 0.8439 | 0.0854 | 102.3022 |
| voting=majority | 60 | 0.8138 | 0.8411 | 0.0787 | 102.3064 |

#### kNN: retention

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| retention=always | 30 | 0.8123 | 0.8254 | 0.0831 | 111.1111 |
| retention=different_class | 30 | 0.8121 | 0.8269 | 0.0860 | 102.0872 |
| retention=never | 30 | 0.8117 | 0.8266 | 0.0708 | 100.0000 |
| retention=oblivion_by_goodness | 30 | 0.8228 | 0.8439 | 0.0883 | 96.0188 |

#### kNN: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| kNN[distance=euclidean;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0775 | 89.7108 |
| kNN[distance=heom;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0894 | 89.7108 |
| kNN[distance=clark;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0898 | 89.7108 |
| kNN[distance=canberra;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0914 | 89.7108 |
| kNN[distance=manhattan;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.1118 | 89.7108 |
| kNN[distance=manhattan;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0752 | 89.7108 |
| kNN[distance=euclidean;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0790 | 89.7108 |
| kNN[distance=clark;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0901 | 89.7108 |
| kNN[distance=canberra;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0910 | 89.7108 |
| kNN[distance=heom;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.1016 | 89.7108 |

### SVM without reduction

#### SVM: kernel

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| kernel=linear | 8 | 0.9304 | 0.9524 | 0.1880 | 100.0000 |
| kernel=poly | 144 | 0.7533 | 0.9690 | 0.5653 | 100.0000 |
| kernel=rbf | 16 | 0.9404 | 0.9659 | 0.4330 | 100.0000 |

#### SVM: class_weight

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| class_weight=None | 84 | 0.7855 | 0.9690 | 0.4908 | 100.0000 |
| class_weight=balanced | 84 | 0.7736 | 0.9690 | 0.5788 | 100.0000 |

#### SVM: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0079 | 0.9650 | 0.6382 | 100.0000 |
| Support Vector Machine[C=0.1;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0086 | 0.9654 | 0.6651 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=0.0;degree=3;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0087 | 0.9653 | 0.6353 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=0.0;degree=3;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0087 | 0.9653 | 0.6883 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6507 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6589 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6678 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.7743 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.8064 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.8901 | 100.0000 |

### RF without reduction

#### RF: criterion

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| criterion=entropy | 48 | 0.9606 | 0.9684 | 2.3770 | 100.0000 |
| criterion=gini | 48 | 0.9608 | 0.9680 | 2.1117 | 100.0000 |

#### RF: n_estimators

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| n_estimators=50 | 24 | 0.9590 | 0.9649 | 0.4835 | 100.0000 |
| n_estimators=100 | 24 | 0.9605 | 0.9665 | 0.9721 | 100.0000 |
| n_estimators=200 | 24 | 0.9616 | 0.9684 | 2.6690 | 100.0000 |
| n_estimators=300 | 24 | 0.9617 | 0.9680 | 4.8528 | 100.0000 |

#### RF: max_features

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| max_features=None | 48 | 0.9573 | 0.9580 | 3.7951 | 100.0000 |
| max_features=sqrt | 48 | 0.9641 | 0.9684 | 0.6936 | 100.0000 |

#### RF: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9684 | 0.0092 | 0.9643 | 1.0487 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9680 | 0.0086 | 0.9640 | 1.5820 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9680 | 0.0091 | 0.9639 | 1.1022 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0089 | 0.9630 | 1.7423 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0076 | 0.9628 | 1.5002 | 100.0000 |
| Random Forest[criterion=gini;max_depth=10;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0096 | 0.9626 | 1.3467 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9671 | 0.0071 | 0.9625 | 0.5444 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9668 | 0.0077 | 0.9624 | 0.6363 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9665 | 0.0088 | 0.9624 | 1.5708 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9665 | 0.0084 | 0.9621 | 1.5497 | 100.0000 |

### Best kNN vs best SVM vs best RF (no reduction)

| config | accuracy | macro_f1 | time | storage |
|---|---|---|---|---|
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.9650 | 0.6382 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9684 | 0.9643 | 1.0487 | 100.0000 |
| kNN[distance=euclidean;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.8407 | 0.0775 | 89.7108 |

### Instance reduction on the best kNN

All variants on the best kNN:

| config | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| kNN[distance=euclidean;k=7;reduction=fcnn;retention=oblivion_by_goodness;version=1;voting=inverse_distance] | 0.8530 | 0.8414 | 2.1467 | 0.0800 | 30.7174 |
| kNN[distance=euclidean;k=7;reduction=del;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8492 | 0.8347 | 24.3002 | 0.0536 | 33.6016 |
| kNN[distance=euclidean;k=7;reduction=drop;retention=oblivion_by_goodness;version=3;voting=inverse_distance] | 0.8482 | 0.8297 | 17.5334 | 0.0445 | 23.2950 |
| kNN[distance=euclidean;k=7;reduction=fcnn;retention=oblivion_by_goodness;version=2;voting=inverse_distance] | 0.8302 | 0.8142 | 2.1508 | 0.0588 | 29.6029 |
| kNN[distance=euclidean;k=7;reduction=eenth;retention=oblivion_by_goodness;threshold=0.0;voting=inverse_distance] | 0.8166 | 0.8160 | 0.6924 | 0.0835 | 71.7971 |
| kNN[distance=euclidean;k=7;reduction=eenth;retention=oblivion_by_goodness;threshold=0.5;voting=inverse_distance] | 0.8013 | 0.8017 | 0.7450 | 0.0982 | 66.6665 |
| kNN[distance=euclidean;k=7;reduction=eenth;retention=oblivion_by_goodness;threshold=0.7;voting=inverse_distance] | 0.7602 | 0.7627 | 0.6800 | 0.0688 | 51.2015 |

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| fcnn | 0.8530 | 0.8414 | 2.1467 | 0.0800 | 30.7174 |
| del | 0.8492 | 0.8347 | 24.3002 | 0.0536 | 33.6016 |
| drop | 0.8482 | 0.8297 | 17.5334 | 0.0445 | 23.2950 |
| none | 0.8439 | 0.8407 | 0.0775 | 0.0775 | 89.7108 |
| eenth | 0.8166 | 0.8160 | 0.6924 | 0.0835 | 71.7971 |

Top kNN configurations with each technique, accuracy (storage):

| rank | kNN | none | fcnn | eenth | drop | del |
|---|---|---|---|---|---|---|
| 1 | kNN(7,euclidean,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8530 (31%) | 0.8166 (72%) | 0.8482 (23%) | 0.8492 (34%) |
| 2 | kNN(7,heom,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8502 (31%) | 0.8166 (72%) | 0.8482 (23%) | 0.8492 (34%) |
| 3 | kNN(7,clark,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8410 (30%) | 0.8179 (72%) | 0.8467 (23%) | 0.8464 (33%) |

Retention vs reduction:

| variant | accuracy | time | solve_time | storage |
|---|---|---|---|---|
| red=fcnn | 0.8530 | 2.1467 | 0.0800 | 30.7174 |
| red=del | 0.8492 | 24.3002 | 0.0536 | 33.6016 |
| red=drop | 0.8482 | 17.5334 | 0.0445 | 23.2950 |
| ret=oblivion_by_goodness | 0.8439 | 0.0775 | 0.0775 | 89.7108 |
| ret=different_class | 0.8269 | 0.0746 | 0.0746 | 101.9227 |
| ret=never | 0.8266 | 0.0668 | 0.0668 | 100.0000 |
| ret=always | 0.8254 | 0.0760 | 0.0760 | 111.1111 |
| red=eenth | 0.8166 | 0.6924 | 0.0835 | 71.7971 |

### Instance reduction on the best SVM

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9690 | 0.9650 | 0.6382 | 0.6382 | 100.0000 |
| eenth | 0.9637 | 0.9601 | 1.5360 | 0.9272 | 82.5531 |
| fcnn | 0.9618 | 0.9578 | 2.3481 | 0.2814 | 41.8320 |
| del | 0.9216 | 0.9141 | 24.5317 | 0.2851 | 42.9364 |
| drop | 0.9132 | 0.9047 | 17.6346 | 0.1457 | 29.8955 |

### Instance reduction on the best RF

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9684 | 0.9643 | 1.0487 | 1.0487 | 100.0000 |
| eenth | 0.9605 | 0.9559 | 1.5174 | 0.9086 | 82.5531 |
| fcnn | 0.9514 | 0.9453 | 2.7662 | 0.6995 | 41.8320 |
| drop | 0.9185 | 0.9079 | 18.0540 | 0.5651 | 29.8955 |
| del | 0.9053 | 0.8943 | 24.9064 | 0.6597 | 42.9364 |

### kNN vs SVM vs RF with each reduction technique

### All tests

| section | question | metric | k | test | p_value | significant | winner / post-hoc |
|---|---|---|---|---|---|---|---|
| kNN | kNN k main effect | accuracy | 3 | Friedman + Iman-Davenport | 6.87e-08 | True | k=7. Nemenyi (CD=1.05): not different from best: k=7, k=5 |
| kNN | kNN k best | accuracy | 3 | Friedman + Iman-Davenport | 3.23e-07 | True | k=7. Nemenyi (CD=1.05): not different from best: k=7, k=5 |
| kNN | kNN distance main effect | accuracy | 5 | Friedman + Iman-Davenport | 0.991 | False | distance=canberra |
| kNN | kNN distance best | accuracy | 5 | Friedman + Iman-Davenport | 1 | False | distance=canberra |
| kNN | kNN voting main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00391 | True | voting=inverse_distance. R+=0, R-=45; paired t-test p=0.000907 (reference) |
| kNN | kNN voting best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0625 | False | voting=inverse_distance. R+=5, R-=40; paired t-test p=0.0296 (reference) |
| kNN | kNN retention main effect | accuracy | 4 | Friedman + Iman-Davenport | 0.00115 | True | retention=oblivion_by_goodness. Nemenyi (CD=1.48): not different from best: retention=oblivion_by_goodness, retention=always |
| kNN | kNN retention best | accuracy | 4 | Friedman + Iman-Davenport | 0.0252 | True | retention=oblivion_by_goodness. Nemenyi (CD=1.48): not different from best: retention=oblivion_by_goodness, retention=different_class, retention=never |
| kNN | kNN top-5 | accuracy | 5 | Friedman + Iman-Davenport | 1 | False | #1 kNN(7,euclidean,inverse_distance,oblivion_by_goodness) |
| SVM | SVM kernel main effect | accuracy | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kernel=rbf. Nemenyi (CD=1.05): not different from best: kernel=rbf, kernel=linear |
| SVM | SVM kernel best | accuracy | 3 | Friedman + Iman-Davenport | 2.86e-07 | True | kernel=poly. Nemenyi (CD=1.05): not different from best: kernel=poly, kernel=rbf |
| SVM | SVM class_weight main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0371 | True | class_weight=None. R+=7, R-=48; paired t-test p=0.0179 (reference) |
| SVM | SVM class_weight best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=balanced. R+=28, R-=27; paired t-test p=0.997 (reference) |
| SVM | SVM top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.996 | False | #1 SVM(poly,0.1,scale,5,0.5,balanced) |
| RF | RF criterion main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.77 | False | criterion=gini. R+=31, R-=24; paired t-test p=0.73 (reference) |
| RF | RF criterion best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.891 | False | criterion=entropy. R+=25.5, R-=29.5; paired t-test p=0.785 (reference) |
| RF | RF n_estimators main effect | accuracy | 4 | Friedman + Iman-Davenport | 0.000298 | True | n_estimators=200. Nemenyi (CD=1.48): not different from best: n_estimators=200, n_estimators=300, n_estimators=100 |
| RF | RF n_estimators best | accuracy | 4 | Friedman + Iman-Davenport | 0.141 | False | n_estimators=200 |
| RF | RF max_features main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00391 | True | max_features=sqrt. R+=54, R-=1; paired t-test p=0.0224 (reference) |
| RF | RF max_features best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0156 | True | max_features=sqrt. R+=43.5, R-=1.5; paired t-test p=0.0226 (reference) |
| RF | RF top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.722 | False | #1 RF(entropy,200,20,sqrt,2) |
| Models | kNN vs SVM vs RF | accuracy | 3 | Friedman + Iman-Davenport | 3.81e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | kNN vs SVM vs RF: kNN vs SVM | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.00391 | True | SVM. raw p=0.001953 |
| Models | kNN vs SVM vs RF: kNN vs RF | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.00391 | True | RF. raw p=0.001953 |
| Models | kNN vs SVM vs RF | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Reduction | kNN reduction | accuracy | 5 | Friedman + Iman-Davenport | 0.000318 | True | fcnn. Bonferroni-Dunn vs none (CD=1.77): eenth worse |
| Reduction | kNN reduction | solve_time | 5 | Friedman + Iman-Davenport | 1.16e-14 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better |
| Reduction | retention vs reduction | accuracy | 8 | Friedman + Iman-Davenport | 6.73e-05 | True | red=fcnn. Nemenyi (CD=3.32): not different from best: red=fcnn, ret=oblivion_by_goodness, red=del, red=drop, ret=never, ret=different_class, ret=always |
| Reduction | SVM reduction | accuracy | 5 | Friedman + Iman-Davenport | 1.72e-13 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, drop worse |
| Reduction | SVM reduction | solve_time | 5 | Friedman + Iman-Davenport | 3.32e-23 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better |
| Reduction | RF reduction | accuracy | 5 | Friedman + Iman-Davenport | 5.44e-15 | True | none. Bonferroni-Dunn vs none (CD=1.77): drop worse, del worse |
| Reduction | RF reduction | solve_time | 5 | Friedman + Iman-Davenport | 8.83e-11 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better, del better, fcnn better |
| Models | models with fcnn | accuracy | 3 | Friedman + Iman-Davenport | 0 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with fcnn | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with eenth | accuracy | 3 | Friedman + Iman-Davenport | 1.63e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with eenth | time | 3 | Friedman + Iman-Davenport | 2.64e-06 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN |
| Models | models with drop | accuracy | 3 | Friedman + Iman-Davenport | 3.48e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with drop | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with del | accuracy | 3 | Friedman + Iman-Davenport | 3.23e-07 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with del | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
