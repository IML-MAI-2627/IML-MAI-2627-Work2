# Design decisions and notes

Living log of decisions that affect the code, the results or the report.
Add an entry whenever the team decides something. Format: what we do, and why.

Status: **decided** / **pending**.

---

## Preprocessing (issue #4, `src/preprocessing.py`)

### Overall approach
- **Fit on train, apply to train and test** (`Preprocessor.fit` / `transform`). All statistics
  (min, max, mean, mode) are computed in `fit` using only the train fold.
  *Why:* using the test fold to compute anything is data leakage and inflates the accuracy.
  `transform` only reads the stored statistics, it never computes them. (decided)
- **Folds are never re-split or modified.** We use the 10 predefined stratified folds as they are. (decided)

### Numeric attributes
- **Min-max scaling to [0,1]** with the train min/max: `x' = (x - min) / (max - min)`.
  *Why:* required by the assignment; also keeps all attributes on a comparable scale so no
  attribute dominates the distance. (decided)
- **Test values outside the train range are clipped to [0,1].**
  *Why:* the train min/max can leave a test value slightly outside [0,1]. Clipping keeps
  every distance term bounded and satisfies "all numeric attributes lie in [0,1]". (decided)
- **Constant column (max == min) becomes 0.**
  *Why:* avoids a division by zero; such a column carries no information anyway. (decided)
- **Own implementation (numpy), not `sklearn.MinMaxScaler`.**
  *Why:* scikit-learn is only allowed for SVM and RF; the kNN part must be our own code. (decided)

### Nominal attributes
- **Integer codes, ONE column per attribute** (`X_nom`), for the kNN.
  *Why:* HEOM compares nominal values by equality (0 if equal, 1 if different), so a single
  column of labels is enough. The numbers are labels, not an order. (decided)
- **Code = position of the value in the ARFF header** (`nominal_values`), not in the data.
  *Why:* the codes are identical in every fold, and a value that appears only in the test fold
  still has a valid code. The header is part of the file definition, so it is not leakage. (decided)
- **One-hot version only for SVM, RF and the numeric-only distances** (`Preprocessed.to_onehot()`).
  *Why:* those methods would interpret integer codes as ordered quantities (a false order).
  Our nominal attributes have no natural order (A/C/G/T, Male/Female), so no ordinal encoding. (decided)
- **Width of the one-hot:** splice gives 287 columns (60 attributes; most have 4 values
  A/C/G/T, some also have the ambiguous codes D/N/S/R). vowel gives 12 (10 numeric + `Sex` x 2). (note)
- **Ambiguous nucleotide codes in splice (D, N, S, R) are valid values, not missing values.** (note)

### Missing values
- **Numeric: train mean. Nominal: train mode.** Statistics ignore the missing cells.
  *Why:* simple, deterministic, computed on train only. (decided)
- **Mode ties go to the first value in header order** (deterministic).
- **A column with every train value missing** gets mean 0 / code 0, so no NaN appears.
- **splice and vowel contain no missing values** (checked with `grep` on the raw files and with
  `missing.sum() == 0`). The code handles them anyway, as the assignment requires. (note)
- **Limitation:** imputing removes the information that a value was missing, so the HEOM rule
  "missing -> distance 1" is not applied. It has no effect on these two datasets. (limitation)
- No missing-indicator columns and no discretization (out of scope).

### Class
- `y` is kept unchanged; `y_codes` gives the integer version (position in the header `class_values`).
- The class is never part of `X_num` / `X_nom`, so it cannot leak into the distance. (decided)

### Dropped columns (vowel)
- **`Train_or_Test` is dropped.** It only records which half of the original UCI split a row came
  from; it does not describe the sound and is unrelated to our 10 folds. (decided)
- **`Speaker_Number` is dropped.** Each speaker says every vowel equally often, so it carries no
  class information, and it would pull neighbours towards rows of the same speaker. (decided)
- **`Sex` is kept.** It has a physical meaning (voice frequency) and only 2 values. (decided)
- The drop list is configurable (`Preprocessor(drop=...)`, defaults in `DEFAULT_DROP`). An unknown
  column name raises `ValueError`, so a typo is not silently ignored.
- **Pending:** confirm with a comparison once the kNN exists: (A) 10 numeric, (B) A + `Sex`,
  (C) B + `Speaker_Number`. Report it in the analysis. (pending)

### How the preprocessing was checked
- Unit tests (synthetic, run in CI): scaling, clipping, constant column, imputation with train
  stats, statistics ignore missing cells, codes, one-hot shape, class unchanged, drop.
- Leakage test: statistics do not change after transforming a very different test set, and
  fitting on train+test gives different statistics.
- Real-data tests (skipped if `datasets/` is empty): all 10 folds of splice and vowel have
  numeric values in [0,1] with no NaN, codes in range, expected one-hot width, class unchanged.
- Independent check: the scaled values of vowel fold 0 match a min-max computed directly with
  `scipy.io.arff` (all 10 numeric columns, train and test).
- End-to-end sanity check: a throwaway 1-NN (Manhattan, numeric attributes only) on vowel gives
  about 98.9% over the 10 folds, far above chance (9%), so data and labels are aligned.

### Observations for the analysis / report
- **vowel has a ceiling effect.** 1-NN already reaches ~98.9%. Likely reason: the dataset has
  15 speakers x 11 vowels x 6 repetitions, and the random stratified folds put repetitions of the
  same speaker in both train and test, so a near-duplicate of each test row is usually in train.
  Expected consequences: very small differences between metrics / K / voting schemes, possibly
  few significant results in the statistical tests, and room for heavy instance reduction without
  losing accuracy. (hypothesis, to be confirmed)

- **splice is imbalanced:** class N is ~52% of the train, which is also the majority-class baseline. (note)
- **splice: informative positions are concentrated in the middle of the sequence (~27-35);** the rest look close to uniform, so an unweighted distance (HEOM) may be diluted by noisy attributes. (hypothesis)
---

## Pending decisions
| Topic | Proposal | Needs |
|---|---|---|
| Distance chosen as "one you may choose" | HEOM (simple, no train statistics) | Team decision before the kNN (#6) |
| Nominal attributes in Euclidean / Manhattan / Clark-Camberra | Use the one-hot version | Team decision before the kNN (#6) |
| Clark / Camberra with 0/0 | Define the term as 0 | Implement in the kNN |
| vowel columns | See "Dropped columns" (A/B/C comparison) | After the kNN exists |
