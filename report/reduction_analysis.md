# Instance reduction: kNN (#19) and SVM / RF (#20)

Draft for the report. Numbers come from `results/gridsearch_v2/<dataset>/report.md` and
`tests.csv` (grid search v2, `--jobs 1`); figures are in `results/gridsearch_v2/plots/`
(`reduction_<dataset>.png`, `tradeoff_<dataset>.png`). Method and test choices are explained in
`docs/DECISIONS.md` ("Grid search v2"). Every test uses the 10 folds of one dataset as blocks,
alpha = 0.05: Friedman (Iman-Davenport) and, if significant, Bonferroni-Dunn against the model
without reduction (CD = 1.77 for k = 5, N = 10).

Each technique is represented by its best variant on the best kNN: FCNN version 1 (each
prototype adds its nearest enemy), EENTh threshold 0.0 (plain probabilistic editing), DROP3 and
DEL. SVM and RF are trained on exactly the same reduced sets (same K, distance and voting as
the best kNN), so the three models are compared on the same prototypes.

**How to read storage.** For SVM and RF, storage is the size of the reduced training set. For
the kNN it is the final case base, so it also reflects the retention policy of the best kNN:
Oblivion by Goodness forgets cases in splice (89.7% without reduction), Always Retain adds the
solved test cases in vowel (111.1% without reduction). The pure effect of a technique is the
SVM / RF column.

**How to read time.** `time` includes the reduction; `solve time` = `time - reduction_time` is
fit + predict on the reduced set. The reduction is a one-off cost, reported separately.

---

## 1. reduction_kNNAlgorithm vs the best kNN (#19)

Best kNN: splice K = 7, Euclidean (all five distances tie on splice: with only nominal
attributes they all order neighbours by the number of mismatches), inverse-distance vote,
Oblivion by Goodness; vowel K = 3, HEOM, inverse-distance vote, Always Retain.

### splice

| technique | accuracy | macro F1 | storage kNN (%) | reduced set (%) | solve time (s) | reduction (s) |
|---|---|---|---|---|---|---|
| none | 0.8439 | 0.8407 | 89.7 | 100.0 | 0.078 | - |
| FCNN | 0.8530 | 0.8414 | 30.7 | 41.8 | 0.080 (median 0.055) | 2.1 |
| DEL | 0.8492 | 0.8347 | 33.6 | 42.9 | 0.054 | 24.2 |
| DROP3 | 0.8482 | 0.8297 | 23.3 | 29.9 | 0.045 | 17.5 |
| EENTh | 0.8166 | 0.8160 | 71.8 | 82.6 | 0.084 | 0.6 |

- **Accuracy:** significant differences (Friedman p = 3.2e-4), but only **EENTh is
  significantly worse** than the unreduced kNN. FCNN, DEL and DROP3 have a slightly *higher*
  mean than the full kNN (+0.4 to +0.9 points), not significant: they keep the accuracy while
  storing a third of the training set.
- **Efficiency:** only DROP3 is significantly faster to solve (0.045 s vs 0.078 s). FCNN is
  faster in 8 of 10 folds (median 0.055 s); its mean is inflated by one outlier fold (0.24 s),
  which also hides it from the rank test. Gains are small in absolute terms, because the
  per-query work in Python (voting and the goodness updates of the retention policy) does not
  shrink with the case base.
- **Cost:** DROP3 and DEL need 17-24 s per fold, more than 200 times a full kNN solve; FCNN
  2 s; EENTh 0.6 s. The reduction only pays off if the reduced case base serves many more
  queries than one test fold.
- **Why EENTh hurts here (hypothesis):** splice has 60 attributes with equal weight, and the
  EDA showed that only the positions around the splice junction are informative. Neighbourhoods
  are therefore noisy, and editing by neighbour agreement removes 17% of the cases, including
  correct ones near the class boundaries, which the condensed and hybrid methods keep on purpose.
- **Robustness:** the same pattern holds for the 2nd and 3rd best kNN (HEOM, Clark): FCNN
  0.841-0.853, DROP3 0.847-0.848, DEL 0.846-0.849, EENTh 0.817-0.818 (`report.md`, "Top kNN
  configurations").

### vowel

| technique | accuracy | macro F1 | storage kNN (%) | reduced set (%) | solve time (s) | reduction (s) |
|---|---|---|---|---|---|---|
| none | 0.9788 | 0.9787 | 111.1 | 100.0 | 0.0144 | - |
| EENTh | 0.9515 | 0.9512 | 108.1 | 97.0 | 0.0149 | 0.1 |
| DEL | 0.9414 | 0.9411 | 45.3 | 34.2 | 0.0081 | 2.0 |
| FCNN | 0.9404 | 0.9400 | 41.1 | 30.0 | 0.0094 (median 0.0069) | 0.2 |
| DROP3 | 0.9101 | 0.9098 | 42.8 | 31.7 | 0.0081 | 1.5 |

- **Accuracy:** every technique lowers it (Friedman p = 7.4e-12); **FCNN, DEL and DROP3 are
  significantly worse** than the full kNN (-3.7 to -6.9 points). EENTh (-2.7 points) is not
  significantly worse, but it hardly reduces anything (it removes 3% of the cases).
- **Efficiency:** FCNN and DROP3 solve significantly faster, but the gain is ~6 ms per fold,
  while reducing costs 0.2-2 s. Not worth it at this size.
- **Why reduction fails here (hypothesis):** vowel has a ceiling effect (1-NN ~99%). Each
  speaker repeats every vowel 6 times and the folds put repetitions of the same speaker in
  train and test, so a near-duplicate of most test cases is in the training set. Condensing
  removes exactly those redundant-looking interior cases, and the "lookup" that made the full
  kNN so accurate is lost.

### Which reduction technique is best for the kNN? Does reduction help?

- **splice:** reduction helps in storage and does not hurt accuracy. **FCNN** is the best
  overall choice: highest mean accuracy, 31% storage and the cheapest reduction (2 s). DROP3 is
  the choice when storage and solve time matter most (23%, the only significant speed-up), at
  8 times the reduction cost; it is statistically tied with FCNN in accuracy. DEL gives no
  advantage over DROP3 and is the slowest to compute. EENTh is the worst.
- **vowel:** reduction does not help: it costs accuracy for a saving that is irrelevant at
  990 cases. Keep the full training set.
- The answer depends on the data: reduction pays off when the training set is redundant *and*
  noisy (splice: many cases, uninformative attributes), not when every case is a useful
  near-duplicate of a test case (vowel).

### Retention vs reduction

Friedman + Nemenyi over the best configuration of each retention policy and the reduced best
kNN (`cd_<dataset>_retention_vs_reduction_accuracy.png`):

- **splice:** FCNN, DEL, DROP3 and all four retention policies are in the same Nemenyi group
  (only EENTh is behind). Oblivion by Goodness, the best retention policy, already forgets 10%
  of the cases at no preprocessing cost; reduction reaches the same accuracy with 23-34% of the
  cases but needs 2-24 s per fold beforehand.
- **vowel:** the four retention policies are the top group (with EENTh); FCNN, DEL and DROP3
  are significantly behind. Retention keeps (or adds) cases, which is what this dataset needs.
- So improving retention is the cheaper way to keep accuracy, and the right choice when the case
  base is small; reduction is the way to cut storage substantially, and it only keeps accuracy
  when the training set is redundant. Both act on the same case base, so they can be combined:
  in splice FCNN + Oblivion by Goodness gives 30.7% storage, below FCNN alone (41.8%), with the
  best mean accuracy of all kNN variants.

---

## 2. SVM and RF on the reduced training sets (#20)

Best SVM: polynomial kernel, degree 5, in both datasets (splice C = 0.1 with balanced class
weights, vowel C = 3). Best RF: entropy criterion, sqrt features (splice 200 trees, max depth
20, min 2 per leaf; vowel 300 trees, full depth). Control = the same model on the full
training set.

### splice

| technique | SVM acc | SVM solve (s) | RF acc | RF solve (s) | reduced set (%) |
|---|---|---|---|---|---|
| none | 0.9690 | 0.64 | 0.9684 | 1.05 | 100.0 |
| EENTh | 0.9637 | 0.93 | 0.9605 | 0.91 | 82.6 |
| FCNN | 0.9618 | 0.28 | 0.9514 | 0.70 | 41.8 |
| DEL | 0.9216 | 0.29 | 0.9053 | 0.66 | 42.9 |
| DROP3 | 0.9132 | 0.15 | 0.9185 | 0.57 | 29.9 |

### vowel

| technique | SVM acc | SVM solve (s) | RF acc | RF solve (s) | reduced set (%) |
|---|---|---|---|---|---|
| none | 0.9909 | 0.022 | 0.9798 | 1.22 | 100.0 |
| EENTh | 0.9697 | 0.021 | 0.9505 | 1.37 | 97.0 |
| FCNN | 0.9455 | 0.008 | 0.8960 | 0.67 | 30.0 |
| DEL | 0.9313 | 0.008 | 0.8980 | 0.73 | 34.2 |
| DROP3 | 0.8889 | 0.007 | 0.8697 | 0.83 | 31.7 |

### SVM: did reduction change the results?

- **splice:** yes for the hybrids. DEL and DROP3 are significantly worse (-4.7 and -5.6
  points); EENTh and FCNN are not (-0.5 and -0.7 points). Only DROP3 makes the SVM
  significantly faster (0.15 s vs 0.64 s).
- **vowel:** FCNN, DEL and DROP3 are significantly worse (-4.5 to -10.2 points); EENTh is not.
  FCNN, DEL and DROP3 are significantly faster, but from 22 ms to 7-8 ms.
- **Interpretation (hypothesis):** FCNN keeps the cases near the class boundaries, which is
  where the support vectors are, so the SVM loses little on splice. DROP3 and DEL start with a
  noise filter that removes every case misclassified by its neighbours, and many of those lie
  on the boundary; what they keep is what a K = 7 (splice) / K = 3 (vowel) kNN needs, not the
  margin of an SVM.

### RF: did reduction change the results?

- **splice:** DROP3 and DEL are significantly worse (-5.0 and -6.3 points); EENTh and FCNN are
  not (-0.8 and -1.7 points). DROP3, DEL and FCNN make it significantly faster to train.
- **vowel:** FCNN, DEL and DROP3 are significantly worse (-8.2 to -11.0 points), the largest
  losses of the three models; EENTh is not. Only FCNN is significantly faster.
- **Interpretation (hypothesis):** RF suffers more than SVM from condensed and hybrid reductions (vowel FCNN:
  -8.4 points for RF vs -4.5 for SVM). A forest estimates class regions from the density of
  cases in each leaf and needs many cases for its bootstrap samples; removing the interior of
  the classes, which is what condensing does, takes that away. The SVM only needs the
  boundary.

### Overall

- **Editing (EENTh) is the only technique that never significantly hurts SVM or RF**, but it
  barely reduces the training set (83% on splice, 97% on vowel), so it does not make them
  faster either.
- **Condensed and hybrid techniques cut the training set to 30-43%** and make SVM / RF faster to
  fit, but cost accuracy, except FCNN on splice with SVM and RF. They were designed for the kNN
  and select the cases a kNN needs.
- **kNN vs SVM vs RF on the same reduced set** (`report.md`, "models with ..."): in splice SVM
  and RF stay ahead of the kNN with every technique. In vowel the reduction levels the field:
  with EENTh, DROP3 and DEL there are no significant differences between the three models, and
  with DROP3 and DEL the kNN has the highest mean. The reduced sets favour the model they were
  built for.

---

## Limitations

- The variant of each technique and the best kNN / SVM / RF were selected on the same 10 folds
  the tests use (no validation set), so the p-values of these comparisons are optimistic
  (`docs/DECISIONS.md`, "Double dipping").
- Times are one measurement per fold; isolated spikes (FCNN fold 5 in both datasets) move the
  means, so medians are given where it matters.
- Reductions were computed once and shared by kNN, SVM and RF (reduction cache); their time is
  the one measured when they were computed.
