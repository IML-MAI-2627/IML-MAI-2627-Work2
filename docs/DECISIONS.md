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
- **Constant column (max == min, or almost) becomes 0.**
  *Why:* such a column carries no information. `MinMaxScaler` only shifts it (a different test
  value would become 1 after clipping), so we set it to 0 ourselves. (decided)
- **Implemented with scikit-learn** (`SimpleImputer`, `MinMaxScaler(clip=True)`, `OrdinalEncoder`,
  `make_pipeline`), behind our own `Preprocessor` class.
  *Why:* the professor confirmed in writing that scikit-learn may be used for SVM, RF and
  preprocessing; the kNN must be our own code. Standard, well-tested transformers are easier to
  justify than hand-written ones. The first version was plain numpy; on splice and vowel the
  sklearn version gives the same output, except for rounding differences (~1e-16). (decided)

### Nominal attributes
- **Integer codes, ONE column per attribute** (`X_nom`), for the kNN.
  *Why:* HEOM compares nominal values by equality (0 if equal, 1 if different), so a single
  column of labels is enough. The numbers are labels, not an order. (decided)
- **Code = position of the value in the ARFF header** (`nominal_values`), not in the data.
  *Why:* the codes are identical in every fold, and a value that appears only in the test fold
  still has a valid code. The header is part of the file definition, so it is not leakage. (decided)
- **One-hot version only for SVM, RF and the numeric-only distances** (`Preprocessed.to_onehot()`).
  *Why:* those methods would interpret integer codes as ordered quantities (a false order).
  Our nominal attributes have no natural order (A/C/G/T, Male/Female): the integer codes
  are only labels, and `OrdinalEncoder` is just the tool that assigns them. (decided)
- **Width of the one-hot:** splice gives 287 columns (60 attributes; most have 4 values
  A/C/G/T, some also have the ambiguous codes D/N/S/R). vowel gives 12 (10 numeric + `Sex` x 2). (note)
- **Ambiguous nucleotide codes in splice (D, N, S, R) are valid values, not missing values.** (note)

### Missing values
- **Numeric: train mean. Nominal: train mode.** Statistics ignore the missing cells.
  *Why:* simple, deterministic, computed on train only. (decided)
- **Mode ties go to the first value in alphabetical order** (deterministic).
  *Why:* it is what `SimpleImputer(strategy="most_frequent")` does on the text values (it returns
  the smallest one), so we need no extra rule. The order is the plain string order, where
  uppercase letters come before lowercase ones. (decided)
- **A numeric column with every train value missing** gets mean 0 and becomes 0, so no NaN
  appears (`keep_empty_features=True`).
- **A nominal column with every train value missing raises `ValueError`** in `fit`.
  *Why:* there is no mode to impute, and sklearn would silently remove the column. Drop the
  column with `Preprocessor(drop=...)` instead. (decided)
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
- Equivalence with the first (numpy) version: same codes and modes on all 10 folds of
  splice and vowel, numeric differences only at rounding level. On 900 random synthetic
  cases the only differences were the two intended ones (mode ties, empty nominal column).
  The synthetic cases were checked with a throwaway script that is not in the repo.

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

## Methods and libraries used

| Area | Method / strategy | Implemented with | Where |
|---|---|---|---|
| Reading ARFF | Parse header and rows; keep features, missing mask and class apart | `scipy.io.arff.loadarff` + own wrapper (`TrainMatrix` / `TestMatrix`) | `src/arff_parser.py` |
| Fold handling | The 10 predefined train/test pairs, read as they are | Own code (`config.fold_paths`, `load_fold`) | `src/config.py`, `src/arff_parser.py` |
| Dropped columns | Configurable list, defaults per dataset | Own code (`DEFAULT_DROP`) | `src/preprocessing.py` |
| Numeric missing values | Train mean | `sklearn.impute.SimpleImputer(strategy="mean", keep_empty_features=True)` | `src/preprocessing.py` |
| Numeric scaling | Min-max to [0,1] with the train range, test clipped | `sklearn.preprocessing.MinMaxScaler(clip=True)` | `src/preprocessing.py` |
| Constant numeric column | Set to 0 (range below `10 * eps`, the same threshold as `MinMaxScaler`) | Own code (numpy), after the scaler | `src/preprocessing.py` |
| Nominal missing values | Train mode on the text values; ties go to the first in alphabetical order | `sklearn.impute.SimpleImputer(strategy="most_frequent")` | `src/preprocessing.py` |
| Nominal column with no train value | `ValueError` in `fit` | Own code (check before the pipeline) | `src/preprocessing.py` |
| Nominal encoding | Integer code = position in the ARFF header, one column per attribute | `sklearn.preprocessing.OrdinalEncoder(categories=header)` | `src/preprocessing.py` |
| Value not declared in the header | `ValueError` naming the attribute and the value, in `fit` and `transform` | Own code (check before the pipeline) | `src/preprocessing.py` |
| No columns of one type | Skip that pipeline (sklearn rejects 0 columns; splice has no numeric ones) | Own code (`if`) | `src/preprocessing.py` |
| Chaining the steps | Fit on train, transform train and test | `sklearn.pipeline.make_pipeline` (one pipeline for numeric, one for nominal) | `src/preprocessing.py` |
| One-hot for SVM / RF | One 0/1 column per header value, built from the integer codes | Own code (`np.eye`) in `Preprocessed.to_onehot()` | `src/preprocessing.py` |
| Class labels | Kept unchanged; integer code = position in the header | Own code (dict lookup) | `src/preprocessing.py` |
| EDA figures | Class counts, histograms, nucleotide heatmaps | `matplotlib`, `seaborn` | `experiments/eda.py` |
| EDA projection | PCA in 2D | Own code (`numpy.linalg.svd`) | `experiments/eda.py` |
| EDA export | Preprocessed fold as CSV (one-hot + class) | `pandas` | `experiments/eda.py` |

Not used (on purpose):
- **scikit-learn in the kNN or in the instance reduction.** Both must be our own code.
- **`sklearn.preprocessing.OneHotEncoder`.** The one-hot is built with `np.eye` from the integer
  codes we already have, so both encodings always agree.
- **`sklearn.preprocessing.LabelEncoder` for the class.** It sorts the labels alphabetically;
  we want the header order.
- **`sklearn.compose.ColumnTransformer`.** Numeric and nominal columns go to separate arrays
  (`X_num`, `X_nom`), so two small pipelines are simpler.
- **`sklearn.decomposition.PCA`.** The EDA needs one projection; a numpy SVD is enough.
- **`StandardScaler` / z-score.** The assignment asks for [0,1].
- **Discretization (`pandas.qcut`) and missing-indicator columns.** Optional and not needed here.
- **Re-splitting (`train_test_split`, `KFold`).** The folds are predefined.

---

## Pending decisions
| Topic | Proposal | Needs |
|---|---|---|
| Distance chosen as "one you may choose" | HEOM (simple, no train statistics) | Team decision before the kNN (#6) |
| Nominal attributes in Euclidean / Manhattan / Clark-Camberra | Use the one-hot version | Team decision before the kNN (#6) |
| Clark / Camberra with 0/0 | Define the term as 0 | Implement in the kNN |
| vowel columns | See "Dropped columns" (A/B/C comparison) | After the kNN exists |
