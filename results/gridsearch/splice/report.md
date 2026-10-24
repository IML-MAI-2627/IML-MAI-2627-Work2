## splice

### kNN without reduction

#### kNN: k

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| k=3 | 40 | 0.7978 | 0.8025 | 0.0440 | 103.9407 |
| k=5 | 40 | 0.8172 | 0.8248 | 0.0491 | 102.2820 |
| k=7 | 40 | 0.8292 | 0.8439 | 0.0395 | 100.6901 |

#### kNN: distance

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| distance=canberra | 24 | 0.8147 | 0.8439 | 0.0383 | 102.3042 |
| distance=clark | 24 | 0.8147 | 0.8439 | 0.0408 | 102.3042 |
| distance=euclidean | 24 | 0.8147 | 0.8439 | 0.0499 | 102.3044 |
| distance=heom | 24 | 0.8147 | 0.8439 | 0.0523 | 102.3044 |
| distance=manhattan | 24 | 0.8147 | 0.8439 | 0.0397 | 102.3042 |

#### kNN: voting

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| voting=inverse_distance | 60 | 0.8156 | 0.8439 | 0.0417 | 102.3022 |
| voting=majority | 60 | 0.8138 | 0.8411 | 0.0467 | 102.3064 |

#### kNN: retention

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| retention=always | 30 | 0.8123 | 0.8254 | 0.0454 | 111.1111 |
| retention=different_class | 30 | 0.8121 | 0.8269 | 0.0430 | 102.0872 |
| retention=never | 30 | 0.8117 | 0.8266 | 0.0409 | 100.0000 |
| retention=oblivion_by_goodness | 30 | 0.8228 | 0.8439 | 0.0474 | 96.0188 |

#### kNN: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| kNN[distance=clark;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0398 | 89.7108 |
| kNN[distance=manhattan;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0399 | 89.7108 |
| kNN[distance=heom;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0400 | 89.7108 |
| kNN[distance=canberra;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0403 | 89.7108 |
| kNN[distance=euclidean;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.0203 | 0.8407 | 0.0409 | 89.7108 |
| kNN[distance=canberra;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0392 | 89.7108 |
| kNN[distance=clark;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0399 | 89.7108 |
| kNN[distance=heom;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0408 | 89.7108 |
| kNN[distance=euclidean;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0420 | 89.7108 |
| kNN[distance=manhattan;k=7;reduction=none;retention=oblivion_by_goodness;voting=majority] | 0.8411 | 0.0197 | 0.8377 | 0.0422 | 89.7108 |

### SVM without reduction

#### SVM: kernel

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| kernel=linear | 8 | 0.9304 | 0.9524 | 0.1466 | 100.0000 |
| kernel=poly | 144 | 0.7533 | 0.9690 | 0.4949 | 100.0000 |
| kernel=rbf | 16 | 0.9404 | 0.9659 | 0.3756 | 100.0000 |

#### SVM: class_weight

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| class_weight=None | 84 | 0.7855 | 0.9690 | 0.4328 | 100.0000 |
| class_weight=balanced | 84 | 0.7736 | 0.9690 | 0.5011 | 100.0000 |

#### SVM: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0079 | 0.9650 | 0.6066 | 100.0000 |
| Support Vector Machine[C=0.1;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0086 | 0.9654 | 0.6180 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=0.0;degree=3;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0087 | 0.9653 | 0.5718 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=0.0;degree=3;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.0087 | 0.9653 | 0.6522 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6136 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6144 | 100.0000 |
| Support Vector Machine[C=1.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6147 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=None;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6162 | 100.0000 |
| Support Vector Machine[C=3.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6325 | 100.0000 |
| Support Vector Machine[C=10.0;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9687 | 0.0083 | 0.9649 | 0.6958 | 100.0000 |

### RF without reduction

#### RF: criterion

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| criterion=entropy | 48 | 0.9606 | 0.9690 | 1.3428 | 100.0000 |
| criterion=gini | 48 | 0.9607 | 0.9671 | 1.3602 | 100.0000 |

#### RF: n_estimators

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| n_estimators=50 | 24 | 0.9591 | 0.9652 | 0.4271 | 100.0000 |
| n_estimators=100 | 24 | 0.9603 | 0.9655 | 0.8507 | 100.0000 |
| n_estimators=200 | 24 | 0.9615 | 0.9690 | 1.6655 | 100.0000 |
| n_estimators=300 | 24 | 0.9616 | 0.9671 | 2.4627 | 100.0000 |

#### RF: max_features

| level | configs | mean acc (all configs) | best acc | mean time (s) | mean storage (%) |
|---|---|---|---|---|---|
| max_features=None | 48 | 0.9572 | 0.9580 | 2.3421 | 100.0000 |
| max_features=sqrt | 48 | 0.9641 | 0.9690 | 0.3609 | 100.0000 |

#### RF: 10 best configurations

| config | accuracy | accuracy_std | macro_f1 | time | storage |
|---|---|---|---|---|---|
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9690 | 0.0100 | 0.9652 | 0.4476 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9674 | 0.0099 | 0.9635 | 0.4856 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0102 | 0.9630 | 0.6656 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0092 | 0.9631 | 0.6627 | 100.0000 |
| Random Forest[criterion=gini;max_depth=10;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9671 | 0.0096 | 0.9626 | 0.6120 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=1;n_estimators=300;n_jobs=1;random_state=42] | 0.9668 | 0.0102 | 0.9626 | 0.6902 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9668 | 0.0099 | 0.9625 | 0.6585 | 100.0000 |
| Random Forest[criterion=gini;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9668 | 0.0071 | 0.9623 | 0.4235 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9668 | 0.0069 | 0.9622 | 0.4727 | 100.0000 |
| Random Forest[criterion=gini;max_depth=20;max_features=sqrt;min_samples_leaf=2;n_estimators=300;n_jobs=1;random_state=42] | 0.9665 | 0.0076 | 0.9621 | 0.6728 | 100.0000 |

### Best kNN vs best SVM vs best RF (no reduction)

| config | accuracy | macro_f1 | time | storage |
|---|---|---|---|---|
| Support Vector Machine[C=0.1;class_weight=balanced;coef0=0.5;degree=5;gamma=scale;kernel=poly;max_iter=-1;random_state=42;tol=0.001] | 0.9690 | 0.9650 | 0.6066 | 100.0000 |
| Random Forest[criterion=entropy;max_depth=None;max_features=sqrt;min_samples_leaf=2;n_estimators=200;n_jobs=1;random_state=42] | 0.9690 | 0.9652 | 0.4476 | 100.0000 |
| kNN[distance=clark;k=7;reduction=none;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8439 | 0.8407 | 0.0398 | 89.7108 |

### Instance reduction on the best kNN

All variants on the best kNN:

| config | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| kNN[distance=clark;k=7;reduction=fcnn;retention=oblivion_by_goodness;version=2;voting=inverse_distance] | 0.8468 | 0.8334 | 1.1052 | 0.0196 | 30.0866 |
| kNN[distance=clark;k=7;reduction=drop;retention=oblivion_by_goodness;version=3;voting=inverse_distance] | 0.8467 | 0.8308 | 6.6681 | 0.0181 | 23.2392 |
| kNN[distance=clark;k=7;reduction=del;retention=oblivion_by_goodness;voting=inverse_distance] | 0.8464 | 0.8313 | 10.2466 | 0.0244 | 33.4239 |
| kNN[distance=clark;k=7;reduction=fcnn;retention=oblivion_by_goodness;version=1;voting=inverse_distance] | 0.8461 | 0.8340 | 1.1022 | 0.0205 | 29.8850 |
| kNN[distance=clark;k=7;reduction=ennth;retention=oblivion_by_goodness;threshold=0.0;voting=inverse_distance] | 0.8179 | 0.8172 | 0.3287 | 0.0350 | 71.8424 |
| kNN[distance=clark;k=7;reduction=ennth;retention=oblivion_by_goodness;threshold=0.5;voting=inverse_distance] | 0.8022 | 0.8026 | 0.3298 | 0.0328 | 66.7466 |
| kNN[distance=clark;k=7;reduction=ennth;retention=oblivion_by_goodness;threshold=0.7;voting=inverse_distance] | 0.7605 | 0.7631 | 0.3611 | 0.0293 | 51.2886 |

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| fcnn | 0.8468 | 0.8334 | 1.1052 | 0.0196 | 30.0866 |
| drop | 0.8467 | 0.8308 | 6.6681 | 0.0181 | 23.2392 |
| del | 0.8464 | 0.8313 | 10.2466 | 0.0244 | 33.4239 |
| none | 0.8439 | 0.8407 | 0.0398 | 0.0398 | 89.7108 |
| ennth | 0.8179 | 0.8172 | 0.3287 | 0.0350 | 71.8424 |

Top kNN configurations with each technique, accuracy (storage):

| rank | kNN | none | fcnn | ennth | drop | del |
|---|---|---|---|---|---|---|
| 1 | kNN(7,clark,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8468 (30%) | 0.8179 (72%) | 0.8467 (23%) | 0.8464 (33%) |
| 2 | kNN(7,manhattan,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8468 (30%) | 0.8179 (72%) | 0.8467 (23%) | 0.8464 (33%) |
| 3 | kNN(7,heom,inverse_distance,oblivion_by_goodness) | 0.8439 (90%) | 0.8364 (30%) | 0.8166 (72%) | 0.8482 (23%) | 0.8492 (34%) |

Retention vs reduction:

| variant | accuracy | time | solve_time | storage |
|---|---|---|---|---|
| red=fcnn | 0.8468 | 1.1052 | 0.0196 | 30.0866 |
| red=drop | 0.8467 | 6.6681 | 0.0181 | 23.2392 |
| red=del | 0.8464 | 10.2466 | 0.0244 | 33.4239 |
| ret=oblivion_by_goodness | 0.8439 | 0.0398 | 0.0398 | 89.7108 |
| ret=different_class | 0.8269 | 0.0392 | 0.0392 | 101.9227 |
| ret=never | 0.8266 | 0.0338 | 0.0338 | 100.0000 |
| ret=always | 0.8254 | 0.0391 | 0.0391 | 111.1111 |
| red=ennth | 0.8179 | 0.3287 | 0.0350 | 71.8424 |

### Instance reduction on the best SVM

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9690 | 0.9650 | 0.6066 | 0.6066 | 100.0000 |
| ennth | 0.9637 | 0.9601 | 0.7093 | 0.4156 | 82.5740 |
| fcnn | 0.9596 | 0.9553 | 1.2231 | 0.1376 | 41.1387 |
| del | 0.9232 | 0.9160 | 10.3643 | 0.1421 | 42.8737 |
| drop | 0.9135 | 0.9050 | 6.7260 | 0.0760 | 29.7597 |

### Instance reduction on the best RF

| technique | accuracy | macro_f1 | time | solve_time | storage |
|---|---|---|---|---|---|
| none | 0.9690 | 0.9652 | 0.4476 | 0.4476 | 100.0000 |
| ennth | 0.9599 | 0.9550 | 0.6790 | 0.3853 | 82.5740 |
| fcnn | 0.9520 | 0.9464 | 1.4487 | 0.3631 | 41.1387 |
| drop | 0.9213 | 0.9115 | 6.8266 | 0.1766 | 29.7597 |
| del | 0.9106 | 0.9004 | 10.4594 | 0.2372 | 42.8737 |

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
| kNN | kNN top-5 | accuracy | 5 | Friedman + Iman-Davenport | 1 | False | #1 kNN(7,clark,inverse_distance,oblivion_by_goodness) |
| SVM | SVM kernel main effect | accuracy | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kernel=rbf. Nemenyi (CD=1.05): not different from best: kernel=rbf, kernel=linear |
| SVM | SVM kernel best | accuracy | 3 | Friedman + Iman-Davenport | 2.86e-07 | True | kernel=poly. Nemenyi (CD=1.05): not different from best: kernel=poly, kernel=rbf |
| SVM | SVM class_weight main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.0371 | True | class_weight=None. R+=7, R-=48; paired t-test p=0.0179 (reference) |
| SVM | SVM class_weight best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | class_weight=balanced. R+=28, R-=27; paired t-test p=0.997 (reference) |
| SVM | SVM top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.996 | False | #1 SVM(poly,0.1,scale,5,0.5,balanced) |
| RF | RF criterion main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 1 | False | criterion=gini. R+=27, R-=28; paired t-test p=0.92 (reference) |
| RF | RF criterion best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.156 | False | criterion=entropy. R+=12.5, R-=42.5; paired t-test p=0.0815 (reference) |
| RF | RF n_estimators main effect | accuracy | 4 | Friedman + Iman-Davenport | 0.000426 | True | n_estimators=200. Nemenyi (CD=1.48): not different from best: n_estimators=200, n_estimators=300, n_estimators=100 |
| RF | RF n_estimators best | accuracy | 4 | Friedman + Iman-Davenport | 0.196 | False | n_estimators=200 |
| RF | RF max_features main effect | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00391 | True | max_features=sqrt. R+=45, R-=0; paired t-test p=0.023 (reference) |
| RF | RF max_features best | accuracy | 2 | Wilcoxon signed-ranks (exact) | 0.00586 | True | max_features=sqrt. R+=53.5, R-=1.5; paired t-test p=0.00338 (reference) |
| RF | RF top-5 | accuracy | 5 | Friedman + Iman-Davenport | 0.458 | False | #1 RF(entropy,200,None,sqrt,2) |
| Models | kNN vs SVM vs RF | accuracy | 3 | Friedman + Iman-Davenport | 3.81e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | kNN vs SVM vs RF: kNN vs SVM | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.00391 | True | SVM. raw p=0.001953 |
| Models | kNN vs SVM vs RF: kNN vs RF | accuracy | 2 | Wilcoxon signed-ranks + Holm | 0.00391 | True | RF. raw p=0.001953 |
| Models | kNN vs SVM vs RF | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, RF |
| Reduction | kNN reduction | accuracy | 5 | Friedman + Iman-Davenport | 0.0327 | True | none. Bonferroni-Dunn vs none (CD=1.77): ennth worse |
| Reduction | kNN reduction | solve_time | 5 | Friedman + Iman-Davenport | 2.46e-15 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better, fcnn better, del better |
| Reduction | retention vs reduction | accuracy | 8 | Friedman + Iman-Davenport | 0.00455 | True | ret=oblivion_by_goodness. Nemenyi (CD=3.32): not different from best: ret=oblivion_by_goodness, red=fcnn, red=drop, red=del, ret=never, ret=different_class, ret=always |
| Reduction | SVM reduction | accuracy | 5 | Friedman + Iman-Davenport | 1.72e-13 | True | none. Bonferroni-Dunn vs none (CD=1.77): del worse, drop worse |
| Reduction | SVM reduction | solve_time | 5 | Friedman + Iman-Davenport | 3.32e-23 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better, fcnn better, del better |
| Reduction | RF reduction | accuracy | 5 | Friedman + Iman-Davenport | 5.34e-13 | True | none. Bonferroni-Dunn vs none (CD=1.77): drop worse, del worse |
| Reduction | RF reduction | solve_time | 5 | Friedman + Iman-Davenport | 6.02e-16 | True | drop. Bonferroni-Dunn vs none (CD=1.77): drop better, del better |
| Models | models with fcnn | accuracy | 3 | Friedman + Iman-Davenport | 1.63e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with fcnn | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with ennth | accuracy | 3 | Friedman + Iman-Davenport | 2.64e-06 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with ennth | time | 3 | Friedman + Iman-Davenport | 3.87e-10 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN |
| Models | models with drop | accuracy | 3 | Friedman + Iman-Davenport | 1.63e-06 | True | RF. Nemenyi (CD=1.05): not different from best: RF, SVM |
| Models | models with drop | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
| Models | models with del | accuracy | 3 | Friedman + Iman-Davenport | 4.64e-05 | True | SVM. Nemenyi (CD=1.05): not different from best: SVM, RF |
| Models | models with del | time | 3 | Friedman + Iman-Davenport | 0 | True | kNN. Nemenyi (CD=1.05): not different from best: kNN, SVM |
