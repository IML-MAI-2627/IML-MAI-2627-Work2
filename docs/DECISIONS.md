# Design decisions and notes

Living log of decisions that affect the code, the results or the report.
Add an entry whenever the team decides something. Format: what we do, and why.

Status: **decided** / **pending**.

---

## Preprocessing (issue #4, `src/preprocessing.py`)

### Overall approach

- **Fit on train, apply to train and test** (`Preprocessor.fit` / `transform`). All statistics (min, max, mean, mode)
  are computed in `fit` using only the train fold.
    - *Why:* using the test fold to compute anything is data leakage and inflates the accuracy.
      `transform` only reads the stored statistics, it never computes them. (decided)
- **Folds are never re-split or modified.** We use the 10 predefined stratified folds as they are. (decided)

### Numeric attributes

- **Min-max scaling to [0,1]** with the train min/max: `x' = (x - min) / (max - min)`.
    - *Why:* required by the assignment; also keeps all attributes on a comparable scale so no
      attribute dominates the distance. (decided)
- **Test values outside the train range are clipped to [0,1].**
    - *Why:* the train min/max can leave a test value slightly outside [0,1]. Clipping keeps
      every distance term bounded and satisfies "all numeric attributes lie in [0,1]". (decided)
- **Constant column (max == min) becomes 0.**
    - *Why:* such a column carries no information. `MinMaxScaler` only shifts it (a different test
      value would become 1 after clipping), so we set it to 0 ourselves. (decided)
- **Implemented with scikit-learn** (`SimpleImputer`, `MinMaxScaler(clip=True)`, `OrdinalEncoder`,
  `make_pipeline`), behind our own `Preprocessor` class.
    - *Why:* the professor confirmed in writing that scikit-learn may be used for SVM, RF and
      preprocessing; the kNN must be our own code. Standard, well-tested transformers are easier to
      justify than hand-written ones. The first version was plain numpy; on splice and vowel the
      sklearn version gives the same output, except for rounding differences (~1e-16). (decided)

### Nominal attributes

- **Integer codes, ONE column per attribute** (`X_nom`), for the kNN.
    - *Why:* HEOM compares nominal values by equality (0 if equal, 1 if different), so a single
      column of labels is enough. The numbers are labels, not an order. (decided)
- **Code = position of the value in the ARFF header** (`nominal_values`), not in the data.
    - *Why:* the codes are identical in every fold, and a value that appears only in the test fold
      still has a valid code. The header is part of the file definition, so it is not leakage. (decided)
- **One-hot version for SVM, RF and for comparing nominal values in the kNN distances** (`Preprocessed.to_onehot()`).
    - *Why:* those methods would interpret integer codes as ordered quantities (a false order).
    - Our nominal attributes have no natural order (A/C/G/T, Male/Female): the integer codes
      are only labels, and `OrdinalEncoder` is just the tool that assigns them. (decided)
- **Width of the one-hot:** splice gives 287 columns (60 attributes; most have 4 values
  A/C/G/T, some also have the ambiguous codes D/N/S/R). vowel gives 12 (10 numeric + `Sex` x 2). (note)
- **Ambiguous nucleotide codes in splice (D, N, S, R) are valid values, not missing values.** (note)

### Missing values

- **Numeric: train mean. Nominal: train mode.** Statistics ignore the missing cells.
    - *Why:* simple, deterministic, computed on train only. (decided)
- **Mode ties go to the first value in alphabetical order** (deterministic).
    - *Why:* it is what `SimpleImputer(strategy="most_frequent")` does on the text values (it returns
      the smallest one), so we need no extra rule. The order is the plain string order, where
      uppercase letters come before lowercase ones. (decided)
- **A numeric column with every train value missing** gets mean 0 and becomes 0, so no NaN
  appears (`keep_empty_features=True`).
- **Columns with more than `max_missing` (default 50%) of their train values missing** are
  handled by a flag, `on_missing`: `"error"` (default) stops and lists ALL of them at once;
  `"drop"` removes them (stored in `dropped_missing_`). Measured on the train fold only. *Limitation:* since it is
  decided per fold, in theory different folds could drop different
  columns. (decided)
- **A nominal column with every train value missing** still raises `ValueError` if
  `max_missing` is set to 1.0: there is no mode to impute. (decided)
- **splice and vowel contain no missing values** (checked with `grep` on the raw files and with
  `missing.sum() == 0`). The code handles them anyway, as the assignment requires. (note)
- **Limitation:** imputing removes the information that a value was missing, so the HEOM rule
  "missing -> distance 1" is not applied. It has no effect on these two datasets. (limitation)
- No missing-indicator columns and no discretization (out of scope).

### Class

- `y` is kept unchanged; `y_codes` gives the integer version (position in the header `class_values`).
- The class is never part of `X_num` / `X_nom`, so it cannot leak into the distance. (decided)
- **Rows with an unknown class (`?`) are removed with a warning** that reports how many. *Why:* they cannot be learned
  from or evaluated. If they are in the test fold, that fold is
  evaluated on fewer cases. splice and vowel have none. (decided)

### Dropped columns (vowel)

- **`Train_or_Test` is dropped.** It only records which half of the original UCI split a row came
  from; it does not describe the sound and is unrelated to our 10 folds. (decided)
- **`Speaker_Number` is dropped.** Each speaker says every vowel equally often, so it carries no
  class information, and it would pull neighbours towards rows of the same speaker. (decided)
- **`Sex` is kept.** It has a physical meaning (voice frequency) and only 2 values. (decided)
- The drop list is configurable (`Preprocessor(drop=...)`, defaults in `DEFAULT_DROP`). An unknown
  column name raises `ValueError`, so a typo is not silently ignored.
- **Pending:** confirm with a comparison once the kNN exists: (A) 10 numeric, (B) A + `Sex`, (C) B + `Speaker_Number`.
  Report it in the analysis. (pending)

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

### Observations for the analysis / report

- **vowel has a ceiling effect.** 1-NN already reaches ~98.9%. Likely reason: the dataset has
  15 speakers x 11 vowels x 6 repetitions, and the random stratified folds put repetitions of the
  same speaker in both train and test, so a near-duplicate of each test row is usually in train.
  Expected consequences: very small differences between metrics / K / voting schemes, possibly
  few significant results in the statistical tests, and room for heavy instance reduction without
  losing accuracy. (hypothesis, to be confirmed)

- **splice is imbalanced:** class N is ~52% of the train, which is also the majority-class baseline. (note)
- **splice: informative positions are concentrated in the middle of the sequence (~27-35);** the rest look close to
  uniform, so an unweighted distance (HEOM) may be diluted by noisy attributes. (hypothesis)

---

## Experiment runner (issue #5, `src/experiment_runner.py`)

### Unified CLI and dynamic parameter parsing

- **The project uses a single entry point (`experiments/run.py`) for all ML experiments.** Model-specific
  hyperparameters are captured dynamically via `parse_known_args()` and passed as a dictionary to the classifier's
  constructor. Exploratory Data Analysis is deliberately kept in a separate standalone script (`experiments/eda.py`) to
  separate visualization logic from the rigorous evaluation pipeline.
    - *Why:* Centralizes experiment execution, keeps the CLI clean, and delegates parameter validation to the specific
      model that uses them, adhering to the single responsibility principle. (decided)

### Common classifier interface

- **All classifiers use the same interface:** they receive a preprocessed training set and test set and return a
  `ClassificationResult` containing predictions and the number of retained training instances.
    - *Why:* the experiment runner should not depend on whether the classifier is kNN, SVM, Random Forest, or a reduced
      kNN. A common return structure lets the same evaluation and CSV-writing code handle all models. (decided)
- **Classifier contract:** a classifier receives preprocessed `train` and `test` data and returns
  `ClassificationResult(predictions, storage)`. `predictions` is a `list[int]` using the class-code mapping defined by
  preprocessing, with one prediction per test instance.
  `storage` is the number of retained training instances. The runner handles metric calculation, storage percentage and
  CSV export. (decided)
- **Object-Oriented Classifier Contract:** All classifiers inherit from a BaseClassifier abstract class. They receive
  their parameters upon instantiation and encapsulate the training and prediction logic inside a fit_predict (self,
  train,
  test) method.
    - *Why:* Eliminates duplicate logic, prevents passing unused params dictionaries through the runner pipeline, and
      gives each
      model absolute control over its own internal state and scikit-learn bindings. (decided)
- **Application example (OOP interface):** A Random Forest classifier processes its parameters, prepares the one-hot
  encoded data, and returns a ClassificationResult:
    ```python
    class RandomForest(BaseClassifier):
        def parse_params(self):
            # Model-specific validation and type casting
            pass

        def fit_predict(self, train, test) -> ClassificationResult:
            # Prepare data (e.g., to_onehot)
            # Fit sklearn model and predict
            return ClassificationResult(
                predictions=predictions,
                storage=len(train.y),
            )
    ```
  If instance reduction is applied, `storage` records the number of retained training instances. (decided)
- The `predictions`values must be integer class codes (Python or numpy integers); anything else, such as
  `1.7`, raises `ValueError` instead of being silently truncated by `int()`. (decided)
- **`storage` in `ClassificationResult` is the retained training-instance count, not a percentage.** The experiment
  runner converts it to the percentage of the original training set before writing the CSV.
    - *Why:* a reduction method can report how many instances it retained, while the runner has the original
      training-set size needed to calculate the required storage percentage. Standard kNN therefore reports 100%
      storage. (decided)

### Fold execution and results

- **`run_fold()` handles one predefined fold:** it loads the preprocessed fold, runs the classifier, measures classifier
  execution time, validates the prediction count and retained-instance count, and computes accuracy, correct
  predictions, incorrect predictions, and storage percentage. The timed section is the complete classifier call, so it
  includes a model fit when the classifier performs one internally, while parsing, preprocessing and CSV writing are
  excluded. (decided)
- **`run_experiment()` runs all 10 predefined folds** by repeatedly calling `run_fold()`; it does not create or modify
  the cross-validation split. (decided)
- **Each 10-fold run starts by resetting the project seed** (`config.SEED`, currently 42).
    - *Why:* stochastic models receive the same random-state sequence when a configuration is
      repeated, while all folds in that configuration remain reproducible. (decided)
- **Results use one common CSV schema:**
  `dataset, fold, model, params, accuracy, n_correct, n_incorrect, time, storage, reduction_time, macro_f1, confusion_matrix`.
    - *Why:* all kNN, SVM, Random Forest, and reduced-model experiments can be stored and compared using the same
      structure. (decided)
- **The confusion matrix of every fold is stored** (`confusion_matrix`, JSON list of lists; row = true class,
  column = predicted class, both in the order of the ARFF header `class_values`; read it back with
  `experiment_runner.parse_confusion_matrix`).
    - *Why:* any other metric (per-class recall, precision, balanced accuracy, ...) and the confusion-matrix plots can
      be derived later from the CSVs without re-running anything, so we do not need to add more metric columns. (decided)
- **Two extra metrics, no more:**
    - `macro_f1`: unweighted mean of the per-class F1, over the classes present in the test fold or in the predictions
      (same rule as scikit-learn's default; a class never predicted gets F1 = 0). *Why:* splice is imbalanced (class N
      ~52%); accuracy alone can hide a model that neglects the minority classes EI/IE. (decided)
    - `reduction_time`: seconds of the timed call spent reducing the training set (0 without reduction). `time` still
      covers the whole call (see "Reduction inside `prepare_data`"); `time - reduction_time` is the time to solve the
      test cases. *Why:* without it the reduced models are always "slower" and the question "does reduction improve
      efficiency?" cannot be answered: reduction is a one-off cost, the gain is in classifying. (decided)
- **Optional per-class scores** (`ClassificationResult.scores`, only when `classifier.return_scores = True`): kNN vote
  shares (majority) or normalised 1/d weights (inverse distance), SVM `decision_function`, RF `predict_proba`. Used
  only for ROC / PR curves. Off by default so the timed experiments do not pay for them; predictions are identical
  with or without them (tested). (decided)
- **The experiment runner is model-agnostic.** Model-specific algorithms and parameters stay in their own classifier
  implementations; the runner only orchestrates execution and collects common metrics. (decided)
- **Preprocessed folds are cached** (`load_fold`, `functools.lru_cache`). Each fold is parsed and preprocessed once per
  process and reused by every configuration.
    - *Consequence:* classifiers must never modify `train` or `test` in place. (decided)
- **Units in the CSV:** `accuracy` is a fraction in [0,1]; `storage` is a percentage in [0,100].
    - *Why:* storage is reported as a percentage of the original training set, accuracy keeps the usual fraction.
      Convert when building tables. (decided)
- **Stochastic classifiers pass `random_state=config.SEED` explicitly** (e.g. Random Forest).
    - *Why:* the seed reset in `run_experiment` only reaches models that use numpy's global random state; an explicit
      `random_state` is reproducible regardless. (decided)

---

## Statistical analysis (issue #14, `src/stats.py`)

### Framework

- **We follow Demšar (2006)**, the mandatory reading of the assignment: Wilcoxon signed-ranks for two configurations;
  Friedman plus a post-hoc test for more; CD diagrams to present them.
    - *Why:* The assignment explicitly asks for "a T-Test or another statistical method [2]", and the course slides
      summarize Demšar's recommendations. (decided)
- **Datasets are evaluated completely independently.** We do not average or aggregate scores across the two datasets.
    - *Why:* The assignment explicitly asks to determine the best parameters and algorithms "at each dataset." (decided)
- **Blocks are the 10 folds of one dataset.** The Friedman test is applied to a $10 \times k$ table where the 10 folds
  act as the datasets/blocks, and the $k$ extracted Champions act as the algorithms.
    - *Why:* Demšar's tests assume blocks are different independent datasets, but the assignment only provides two
      datasets and requires computing performance over the 10-fold cross-validation sets. Treating the 10 folds as
      blocks ($N=10$) is the standard academic adaptation to fulfill the assignment requirements.
    - *Limitation:* Folds are not perfectly independent (their training sets overlap). Demšar (Section 3.2.3) notes that
      no test natively accounts for this overlapping dependency, which stands as a known theoretical limitation of this
      approach. (decided / limitation)
- **Significance level alpha = 0.05** for every test unless stated otherwise. (decided)

### Three or more configurations (Multiple Hypothesis Testing)

- **Friedman is computed with Demšar's formula, and the decision uses the Iman-Davenport F_F.**
    - *Why:* `scipy.stats.friedmanchisquare` applies a tie correction the paper does not use; the chi-square
      approximation is conservative and needs N > 10 and k > 5, while we have N = 10 (Demšar, Section 3.2.2). (decided)
- **Post-hoc only after a significant Friedman test:**
    - **Nemenyi** is used for all-against-all comparisons. We apply this to our elite subset of 4 (RF) or 6 (SVM)
      Champions.
        - *Why we test a subset and not the whole grid:* Demšar's Nemenyi test controls the family-wise error rate
          across $k (k-1)/2$ comparisons. If we tested all 168 SVM configurations simultaneously ($k=168$), the test
          would make ~14,000 comparisons. This triggers the **Nemenyi Power Loss**, pushing the Critical Difference (CD)
          so high that the test fails to separate any models. Testing only the 6 core Champions ensures the CD remains
          tightly scoped and mathematically powerful. (decided)
    - **Bonferroni-Dunn** is used when there is a control (e.g., comparing the baseline "Best kNN" without reduction
      against the reduction techniques). It is more powerful because it makes $k - 1$ comparisons instead
      of $k (k - 1)/2$ (Demšar, p. 12). (decided)
- **Critical values are computed, not copied:** Nemenyi q = studentized range / sqrt (2) with infinite degrees of
  freedom; Bonferroni-Dunn q = z_{1 - alpha / (2 (k - 1))}. Both reproduce Demšar's Table 5 and also work for k > 10.
  Table 5 (b) prints 2.724 for k = 9, alpha = 0.05; the formula and the neighbouring entries give 2.734, so we treat it
  as a typo. (decided)

### Two configurations (Pairwise testing)

- **Wilcoxon signed-ranks decides; the paired t-test is reported only as a reference.**
    - *Why:* Demšar (Section 3.1.2) shows the t-test needs normal differences and is statistically unsafe with fewer
      than ~30 blocks. Wilcoxon is the recommended non-parametric alternative. We use this strictly for head-to-head
      comparisons (e.g., Best overall RF vs Best overall SVM). (decided)
- **Zeros follow Demšar's rule** (split evenly between R+ and R-, one ignored if their number is odd) and the **p-value
  is exact** (all sign patterns).
    - *Why:* `scipy.stats.wilcoxon` handles zeros differently and switches to a normal approximation with ties or zeros,
      diverging from Demšar's exact method. (decided)

### Presentation

- **CD diagrams follow Demšar's Figure 1:** best rank on the right; Nemenyi groups joined by a thick line; for
  Bonferroni-Dunn, the interval of one CD around the control is drawn exactly (no padding) because its limits decide
  significance. (decided)

### How the module was checked

- Demšar Table 2: R+ = 93, R- = 12. Table 6: mean ranks 3.143 / 2.000 / 2.893 / 1.964,
  chi2_F = 9.28, F_F = 3.69. Table 5 (a) and 5 (b) critical values. CD = 1.25 and 1.12 (Nemenyi)
  and 1.16 (Bonferroni-Dunn), with the conclusions of p. 14 and the groups of Figure 1 (a).
- The exact Wilcoxon p-value matches `scipy.stats.wilcoxon(method="exact")` when there are no
  ties or zeros.

---

## Random Forest (issue #13, `src/models/random_forest.py`)

### Scikit-learn integration

- **Uses `sklearn.ensemble.RandomForestClassifier`.**
    - *Why:* The assignment explicitly allows using the scikit-learn library implementation for the Random Forest and
      SVM classifiers. (decided)

### Data preparation

- **Nominal variables are One-Hot Encoded (`train.to_onehot()`).** The Random Forest is fed the numeric columns followed
  by binary columns for each nominal category, rather than the raw integer codes.
    - *Why:* Scikit-learn's Random Forest interprets single-column integer labels as continuous/ordinal values. Since
      our nominal categories (e.g., A/C/G/T or Male/Female) have no natural mathematical order, feeding them as raw
      integers would force the trees to make arbitrary splits (e.g., `value <= 2`). One-hot encoding ensures independent
      and correct evaluation of categorical splits. (decided)
- **Empty matrix validation.** The `prepare_data` method raises a `ValueError` if the generated matrices or label arrays
  are empty or misaligned.
    - *Why:* "Fail-fast" philosophy. Catching data corruption or preprocessing errors manually before feeding the data
      to scikit-learn prevents cryptic backend errors and guarantees data integrity. (decided)

### Parameter handling and validation (Fail-Fast philosophy)

- **Core hyperparameters exposed (`n_estimators`, `criterion`, `max_depth`, `max_features`, `min_samples_leaf`).** These
  control the ensemble size, splitting logic, and individual tree regularization, matching the core parameters
  highlighted in the theory slides.
    - *Why:* They form the minimal and sufficient set to properly tune a Random Forest's bias-variance tradeoff.
      (decided)
- **Strict mathematical constraints (No `None` for strict ints).** Parameters like `n_estimators` and `min_samples_leaf`
  are rigorously cast to `int`, and explicitly reject the string `"None"`.
    - *Why:* These parameters represent hard counts in the scikit-learn backend. Passing a `None` value mathematically
      breaks the algorithm. Rejecting them at the parsing stage provides immediate, clear feedback. (decided)
- **Nullable constraints for specific limits (`max_depth`).** The parser allows `"None"` for `max_depth`.
    - *Why:* In a decision tree, a `max_depth` of `None` is mathematically valid and implies "expand nodes until all
      leaves are pure or contain less than min_samples_split samples." (decided)
- **Flexible parsing for `max_features`.** Accepts strings (`"sqrt"`, `"log2"`, `"None"`), floats (e.g., `0.5`), and
  integers (e.g., `5`).
    - *Why:* Scikit-learn has very specific behaviors depending on the type: a float acts as a percentage of total
      features, an integer acts as an absolute count, and strings apply specific math functions. The parser perfectly
      preserves this flexibility. (decided)
- **Explicit string validation for `criterion`.** Typographical errors (e.g., `--criterion patata`) immediately raise a
  `ValueError`.
    - *Why:* Fail-fast principle. It prevents the model from silently falling back to `"gini"`, ensuring the user is
      actually testing the split logic they intended. (decided)
- **Rejection of unknown parameters.** Providing any parameter via CLI that is not in the model's `default_params`
  dictionary immediately raises a `ValueError`.
    - *Why:* Fail-fast principle. It guarantees that typos (e.g., `--n_estimtors 50`) break execution instantly instead
      of being silently ignored by the parameter parser, ensuring user intent matches execution perfectly. (decided)
- **Default injection of `random_state`.** If omitted via CLI, the model automatically reads and injects `config.SEED`.
    - *Why:* Random Forest relies heavily on randomness (bootstrapping samples and randomly selecting feature subsets at
      each split). Injecting the seed guarantees 100% reproducible execution across all folds. (decided)
- **Exposed `n_jobs`.** Allows setting the number of parallel jobs via CLI.
    - *Why:* Training 100+ trees is trivially parallelizable. Exposing this allows utilizing all CPU cores
      (`--n_jobs -1`) to drastically reduce experiment time on large datasets without altering mathematical results.
      (decided)
- **Default values explicitly added**: If some params are not passed through the CLI, they are explicitly added into
  `params`.
    - *Why:* We do this in order to visualize them in the csv, as the default param values wouldn't be captured and
      thus, we facilitate experiment discerning. (decided)

## Support Vector Machine (issue #12, `src/models/svm.py`)

### Scikit-learn integration

- **Uses `sklearn.svm.SVC`.**
    - *Why:* The assignment explicitly allows using the scikit-learn library implementation for the Random Forest and
      SVM classifiers. (decided)

### Hyperparameter selection and validation

- **Core hyperparameters exposed (`C`, `kernel`, `gamma`, `degree`).** These are the fundamental parameters that dictate
  the SVM's regularization and decision boundary shape, as highlighted in the theoretical slides.
    - *Why:* They form the minimal required set to properly tune an SVM across different datasets. (decided)
- **Included `class_weight` validation (`'balanced'` vs `None`).**
    - *Why:* The EDA revealed that the `splice` dataset is imbalanced (majority class ~52%). Standard SVMs optimize for
      overall accuracy and can collapse into predicting only the majority class. Exposing `class_weight="balanced"`
      allows the algorithm to heavily penalize misclassifications of minority classes. (decided)
- **Included convergence safeguards (`max_iter`, `tol`).**
    - *Why:* SVM training time scales roughly at $O (N^3)$. With complex datasets or very high values of `C`, the
      optimization solver can hang indefinitely. Exposing `max_iter` allows us to enforce a hard time limit on
      experiments. (decided)
- **Included `coef0` for kernel shifting.**
    - *Why:* When experimenting with `poly` or `sigmoid` kernels, the independent term is mathematically necessary to
      control the influence of higher-order polynomials vs lower-order ones. (decided)
- **Default injection of `random_state`.** If omitted via CLI, the model automatically reads and injects `config.SEED`.
    - *Why:* Even though standard SVM is deterministic, `scikit-learn` uses randomness under the hood for probability
      estimations and data shuffling during optimization. Injecting the seed guarantees 100% reproducible execution
      across all folds. (decided)

### Data Preparation and Fail-Fast philosophy

- **Strict mathematical constraints (No `None` for floats/ints).** Parameters like `C`, `degree`, or `max_iter` are
  rigorously cast to `float` or `int`, and explicitly reject the string `"None"`.
    - *Why:* These parameters represent mathematical constants or hard limits in the scikit-learn backend. Passing a
      `None` value crashes the underlying C/Cython solver. Rejecting them at the parsing stage provides immediate, clear
      feedback to the user. (decided)
- **Explicit string validation for `kernel` and `class_weight`.** Typographical errors (e.g., `--kernel patata`)
  immediately raise a `ValueError`.
    - *Why:* Fail-fast principle. It prevents the model from silently falling back to a default configuration, ensuring
      the user is actually testing the parameters they intended to test. (decided)
- **Rejection of unknown parameters.** Providing any parameter via CLI that is not in the model's `default_params`
  dictionary immediately raises a `ValueError`.
    - *Why:* Fail-fast principle. It guarantees that typos (e.g., `--kernal poly`) break execution instantly instead of
      being silently ignored by the parameter parser, ensuring user intent matches execution perfectly. (decided)
- **Kernel-based pruning of inactive hyperparameters.** After parsing, the algorithm dynamically removes variables from
  `self.params` that have no mathematical effect on the chosen kernel (e.g., `degree` is dropped if kernel is `rbf`;
  `gamma` is dropped if kernel is `linear`).
    - *Why:* This prevents the `experiments/run.py` CSV exporter from recording mathematically identical executions as
      distinct configuration strings. It heavily optimizes and cleans the output of exhaustive Grid Search processes.
      (decided)
- **Data prepared via One-Hot Encoding (`to_onehot()`).** The SVM receives the numeric columns followed by binary
  columns for nominals.
    - *Why:* Like the Random Forest, standard SVMs interpret distance geometrically. Feeding raw integer categorical
      codes would create false distances and orderings between nominal values (e.g., A/C/G/T). One-hot encoding ensures
      orthogonal geometric distances between categories. (decided)

## kNN (issue #6, #7-#11, #16-#18: `src/models/knn/`)

### Architecture and Modular Organization

- **Modular package architecture in `src/models/knn/` with unified utilities.**
  Instead of scattering logic across multiple files (`knn_base.py`, `knn.py`, `distances.py`, etc.), the kNN subsystem is organized into specialized, cohesive modules:
  - `knn_utils.py`: Shared data containers (`Neighbor`, `Neighbors`, `KNNRun`), validation functions (`check_k`, `_check_distances`, `_check_vote`), dataset slicing, distance matrix operations, nearest-neighbor sorting, sequential retention processing (`_run_with_retention`), and the unified execution entry point `run_kNN()`.
  - `knn.py`: The `KNN(BaseClassifier)` adapter orchestrating parameter parsing, data preparation, reduction, and delegating execution to `run_kNN()`.
  - `distances.py`: Own distance metric implementations (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`) in pure NumPy with symmetric optimization.
  - `voting.py`: Voting strategies (`majority`, `inverse_distance`).
  - `retention.py`: Dynamic test-time retention policies (`never`, `always`, `different_class`, `oblivion_by_goodness`).
  - `reduction.py`: Training set instance reduction algorithms (`fcnn`, `drop`, `eenth`) with object-oriented validation (`InstanceReduction`).
    - *Why:* Eliminates code duplication, avoids circular dependencies, and provides a single, unified execution path (`run_kNN`) for batch and retention classification, maximizing code reuse across kNN, reduction, and retention. (decided)
- **Single unified execution engine: `run_kNN()`.**
  Replaces separate algorithm runners. When `retention_function is None`, `run_kNN()` executes the fast batch path using vectorized NumPy broadcasting and stable sorting. When a retention policy is active, it delegates directly to `_run_with_retention()` to sequentially classify and update the case base.
    - *Why:* Simplifies the adapter to a single execution call, avoiding branching complexity while keeping execution speed optimal. (decided)
- **Strictly own code with NumPy, no scikit-learn.**
  All distance evaluations, prototype selection, associate graph updates, neighbor sorting, voting, and retention are implemented from scratch using NumPy.
    - *Why:* Strict requirement of the course assignment; custom vectorization yields comparable or superior performance to scikit-learn on our datasets. (decided)
- **`kNNAlgorithm(...)` is the algorithm, `KNN(BaseClassifier)` is the adapter.**
  `kNNAlgorithm(train, test, k, distance_function, voting_function, retention_function=None)` in
  `src/models/knn/knn_utils.py` returns one prediction per test case (it calls `run_kNN`). `KNN` in
  `src/models/knn/knn.py` follows the same structure as `SVM` and `RandomForest` (`parse_params`,
  `prepare_data`, `fit_predict`), so `experiments/run.py --classifier knn` runs it like any other model.
    - *Why:* the assignment requires the name `kNNAlgorithm(...)`, and the runner needs the common
      `BaseClassifier` interface. (decided)
- **The other required names follow the same pattern:** `svmAlgorithm(train, test, **params)`
  (`src/models/svm.py`), `rfAlgorithm(train, test, **params)` (`src/models/random_forest.py`) and
  `reduction_kNNAlgorithm(train, test, k, distance, voting, reduction, retention=None, **options)`
  (`src/models/knn/reduction.py`: reduce `train` with the kNN's own k, distance and voting, then
  `kNNAlgorithm`). They return the predictions; a test checks they equal the adapters' predictions, so the
  experiments (run through the adapters) are the results of these functions. (decided)
- **Checked against scikit-learn (`KNeighborsClassifier`, brute force) on all 10 folds** (only in `tests/`):
  identical predictions for k=1 on both datasets; for k>1 every difference comes from a tie (at the k-th
  neighbour or in the vote), which we break deterministically and sklearn does not. (note)

### Distance Calculations and Symmetric Optimization (`distances.py`)

- **Single-argument calls (`dist_fn(data)`) give the symmetric train x train matrix through the normal path.**
  The first version evaluated only the upper triangle with `np.triu_indices`. Measured on splice fold 0
  (2871 train cases): **2.5 s and 4.3 GB peak memory**, because `X_nom[i]` / `X_nom[j]` gather two
  (4.1M x 60) int64 arrays. The normal path counts nominal mismatches with one product of 0/1 one-hot matrices:
  **0.26 s and ~0.4 GB**, same values (test `test_distance_single_argument_matches_symmetric`).
    - *Why:* half the arithmetic does not help when the gathering dominates; a BLAS product is ~10x faster and the
      memory peak made parallel runs on splice impossible. (decided, replaces the triu version)
- **Reconstruction and retention of the full $(N, N)$ symmetric 2D matrix.**
  `compute_distance_matrix` returns the full $(N, N)$ NumPy matrix with exact zeros on the diagonal.
    - *Why:* All neighbor queries in FCNN (nearest enemy search), DROP (associates search and sorting), EENTh (consensus checking), and retention (candidate filtering) rely on **$O(1)$ vectorized row slicing** (`D[p, :]` and `D[p, candidates]`). Storing a condensed 1D distance array would require complex index arithmetic and prevent NumPy's optimized C-level row slicing, drastically slowing down neighbor lookups in Python. The full 2D float64 matrix takes minimal storage (~8 MB for $N=1000$), so keeping the 2D matrix is optimal. (decided)
- **Five registered distance metrics (`DISTANCES`):**
  - `euclidean`: Minkowski $r=2$ with nominal overlap on one-hot representation.
  - `manhattan`: Minkowski $r=1$ with nominal overlap on one-hot representation.
  - `canberra`: Relative difference $\frac{|x - y|}{|x| + |y|}$ with nominal overlap; $0/0$ terms count as 0.
  - `clark`: Squared relative difference without square root; $0/0$ terms count as 0.
  - `heom`: Heterogeneous Euclidean-Overlap Metric (Wilson & Martinez, 1997) combining scaled continuous distances with categorical equality overlap. (decided)
- **Neighbour order is deterministic:** stable argsort, so ties in distance go to the lower training
  index. Voting policies receive neighbours ordered by `(distance, index)`. (decided)
- **All distances must be finite and non-negative, and the matrix must have the right shape.**
  Otherwise `ValueError` before voting. (decided)
- **`euclidean` on one-hot is computed as two exact parts:** numeric squared differences, plus
  `2 x (number of different nominal values)`, because two one-hot rows of one attribute differ in 0 or 2
  columns. The number of matches comes from a product of 0/1 matrices, which is exact, so equal rows give
  exactly 0 and ties remain ties.
    - *Why:* a direct one-hot difference needs an `(n_test, n_train, n_onehot)` array (~2 GB on splice), and a
      row-by-row loop took 0.69 s per splice fold against 0.027 s. (decided)
- **Sanity check:** 1-NN euclidean on vowel gives 0.9889 mean accuracy over the 10 folds, the same as the
  1-NN check of the preprocessing (~98.9%). (note)

### Decoupled Training Reduction and Test Evaluation (`knn.py`)

- **Decoupled distance computation between training reduction and test prediction.**
  Instead of computing an upfront $(Train + Test) \times (Train + Test)$ matrix, distances are computed on `train` for reduction, and the test phase computes distances only against retained prototypes ($N_{test} \times N_{reduced}$).
    - *Why:* In instance reduction, 50% to 90% of training instances are pruned. Precomputing distances from test instances to training instances that get discarded was completely wasted computation. Furthermore, computing $N_{test} \times N_{test}$ distances when retention is `"never"` was 100% redundant. Decoupling eliminates all redundant distance calculations. (decided)
- **Parameters: `k` (default 3), `distance` (default `euclidean`), `voting` (default `majority`),
  `retention` (default `never`), `reduction` (default `none`).** Unknown keys, a non-integer or non-positive
  `k` (including `"None"`), and unknown strategy names raise `ValueError`. Names are case-insensitive.
  Defaults are injected into `params` so they appear in the CSV. (decided)
- **Strategies are selected by name** through the registries `DISTANCES`, `VOTING` and `RETENTION`
  (`src/models/knn/`). `self.params` keeps only the names; the callables are looked up in `fit_predict`.
    - *Why:* the CLI passes strings, and the CSV `params` column must show readable names, not
      `<function ...>`. (decided)
- **No `random_state`:** the kNN is deterministic. (decided)
- **`storage` is the size of the case base at the end of the test phase:** the (reduced) training set
  without retention; more or less with AR, DF and OG (see *Retention*). (decided)

### Case-Base Retention Policies (`retention.py`)

- **Sequential processing over a dynamic case base.**
  Unlike batch kNN, dynamic retention evaluates test instances sequentially ($i = 0, \dots, N_{test}-1$). The case base begins with the training prototypes and dynamically expands (or contracts) as test instances are classified. (decided)
- **Four registered retention policies (`RETENTION`):**
  - `never`: Static case base; standard batch evaluation.
  - `always`: Retains every test instance upon classification.
  - `different_class`: Retains a test instance only if the prediction was incorrect.
  - `oblivion_by_goodness`: Reinforcement learning case retention. (decided)
- **Encapsulation of `OblivionByGoodness` in `retention.py`.**
  Encapsulates neighbor goodness tracking, reinforcement adjustments, and threshold-based forgetting:
  - Neighbors contributing to a correct classification receive positive reinforcement; neighbors voting incorrectly are penalized.
  - Cases whose goodness score drops below the threshold are pruned from the active mask.
  - Enforces a minimum case floor (`min_cases=k`) so that pruning never reduces active cases below the neighborhood size $k$. (decided)

### Instance Reduction Techniques (`reduction.py`)

- **Four registered reduction techniques (`REDUCTION_TECHNIQUES`):**
  - **FCNN (Fast Condensed Nearest Neighbor, Angiulli 2007):**
    Initializes prototype set $S$ with class centroids/medoids. Iteratively queries representatives for nearest enemies using precomputed distances. Guaranteed 100% training-set consistent (Theorem 3.1).
    - Governed by `--version` (Angiulli 2007, with medoids instead of centroids): `1` every prototype adds its
      nearest Voronoi enemy; `2` every prototype adds the medoid of its enemies; `3` and `4` add only one instance
      per iteration (the nearest enemy / the enemy medoid of the prototype with most enemies), slower but smaller.
    - *Why:* The centroid rule provides clean class anchors; enemy selection rules balance boundary tightness vs storage compression. (decided)
  - **DROP (Decremental Reduction Optimization Procedure, Wilson & Martinez 2000):**
    Prunes instances whose removal does not reduce the leave-one-out classification accuracy of their *associates* (instances that have $p$ in their $k+1$ nearest neighbors).
    - Governed by `--version`: `1` (DROP1), `2` (DROP2), `3` (DROP3 with ENN pre-filtering), `4` (DROP4), `5` (DROP5).
    - In DROP1, removed points are discarded from neighbor lists of associates so risk evaluations never reference pruned instances.
    - *Why:* Associate-based risk checks prevent cascading classification errors; DROP3's initial ENN pass eliminates noisy border instances before pruning internal prototypes. (decided)
  - **ENNTh (Edited Nearest Neighbor estimating class probabilities and Threshold; Vázquez, Sánchez & Pla 2005,
    "WilsonTh"; called `eenth` in the code):** for each instance, its $k$ nearest neighbours (itself excluded) give
    class probabilities $P_i(x) = \sum_j p_i^j / (1 + d(x, x_j))$, normalised to sum 1. The instance is removed if the
    most probable class is not its label, **or** if that maximum probability is $\le$ `--threshold` $\mu$
    (default 0.0, which is the paper's WilsonProb). Checked against the paper (Section 3).
    - *Why:* Provides flexible border smoothing and noise cleaning; unified under a single threshold parameter. (decided)
  - **DEL (Decremental Encoding Length, Wilson & Martinez 2000 Section 4.6 & 3.2.8; Cameron-Jones 1995):**
    Decremental instance pruning governed by the Minimum Description Length (MDL) encoding length heuristic:
    $$Cost(m, n, x) = F(m, n) + m \log_2(C) + F(x, n - m) + x \log_2(C - 1)$$
    where $m$ is the number of retained instances, $n$ is total training set size, $x$ is the number of misclassified training instances (exceptions), and $C$ is the number of target classes.
    - **Log-Space Combinatorial Sums ($F(a, b)$):**
      $$F(a, b) = \log^* \left(\sum_{j=0}^a \binom{b}{j}\right)$$
      In naive arithmetic, evaluating $\sum_{j=0}^a \binom{b}{j}$ overflows standard 64-bit IEEE 754 floating point numbers for $b \ge 1000$ (e.g., $\binom{1000}{500} \approx 2.7 \times 10^{299}$, while `float64` max is $\approx 1.8 \times 10^{308}$). To ensure exact numerical stability:
      1. Precompute cumulative log-factorials $\ln(i!) = \sum_{k=1}^i \ln(k)$ up to $N$.
      2. Express each binomial coefficient in log-space: $\ln \binom{b}{j} = \ln(b!) - \ln(j!) - \ln((b-j)!)$.
      3. Sum terms using the standard Log-Sum-Exp trick: $\ln \sum \exp(z_j) = z_{\max} + \ln \sum \exp(z_j - z_{\max})$.
      4. Convert to base-2 logarithm ($\log_2(z) = \frac{\ln(z)}{\ln(2)}$) before evaluating Rissanen's universal prior for integers $\log^*(z) = \log_2(c) + \log_2(z) + \log_2(\log_2(z)) + \dots$
      - *Literature references:* Cameron-Jones (1995), Wilson & Martinez (2000), Quinlan & Rivest (1989), Rissanen (1983), Press et al. (2007, *Numerical Recipes* Section 6.1), and Goodfellow et al. (2016, *Deep Learning* Section 4.1). (decided)
    - **DEL Pruning Rules:**
      - An instance is removed during the noise-filtering pass if it is misclassified by its $k$ nearest neighbors **and** dropping it does not increase description length cost ($Cost_{new} \le Cost_{curr}$).
      - During the decremental pass, candidate prototypes sorted by distance to their nearest enemy are removed if the associate error delta and self-exception check maintain or lower the system encoding length.
      - Implemented by reusing `drop()` with an `is_del: bool = True` branch. (decided)
      - *Interpretation of the paper:* DEL has a noise-filtering pass and then a decremental pass ordered by
        distance to the nearest enemy (Wilson & Martinez 2000, Section 4.6). The paper says the instances are removed
        "as long as any improvement is being made", which can be read as one ordered pass (our choice, as in DROP3)
        or as repeating that pass until nothing is removed; extra passes could only remove more instances. The number
        of exceptions x after a removal is estimated from the associates and the removed instance itself, not by
        reclassifying the whole training set. (decided)

- **Object-Oriented Reduction Hierarchy (`reduction.py`):**
  Each reduction algorithm is implemented as a dedicated subclass inheriting from an abstract `InstanceReduction` base class:
  - `NoReduction` (method `"none"`): identity wrapper.
  - `FCNNReduction` (method `"fcnn"`): handles version 1..4.
  - `DROPReduction` (method `"drop"`): handles version 1..5.
  - `EENThReduction` (method `"eenth"`): handles threshold $\ge 0.0$.
  - `DELReduction` (method `"del"`): handles encoding length reduction.
  - `InstanceReduction.__new__` serves as a polymorphic factory so that `InstanceReduction(params)` or `InstanceReduction("del", k=3)` transparently returns the specialized subclass instance satisfying `isinstance(obj, InstanceReduction)`.
  - Each subclass overrides `reduce(train, distance_matrix=None)` and validates/prunes its specific hyperparameters. (decided)

- **Model-Agnostic Reduction across Classifiers:**
  Instance reduction operates directly on `Preprocessed` training data after feature preprocessing (`train.reduce(reducer)` or `reduce_preprocessed(train, reducer)`).
  - All classifiers (`KNN`, `SVM`, `RandomForest`, `MajorityBaseline`) support reduction and report the retained prototype count in `storage`.
  - **Reduction inside `prepare_data` instead of `Preprocessor.fit`:**
    Instance reduction is performed inside each classifier's `prepare_data(train, test)` method, rather than during dataset feature preprocessing (`Preprocessor.fit`):
    - *Timing Integrity:* `run_fold` times `classifier(train, test)` (which calls `fit_predict`, which delegates to `prepare_data`). This ensures prototype selection runtime (DROP3, FCNN, DEL) is fully captured in the benchmarked `time` metric.
    - *Caching Independence:* `load_fold` caches feature preprocessing (`@cache`) per `(dataset, fold)`. Keeping reduction model-side avoids cache poisoning across different reduction configurations.
    - *Storage Percentage Correctness:* `run_fold` receives the unreduced `train` and calculates `storage = 100.0 * result.storage / len(train.y)`. If reduction were in `load_fold`, `len(train.y)` would already be reduced, falsely yielding 100% storage.
  - **kNN parameter separation:** In `KNN`, the hyperparameters `--k`, `--distance`, and `--voting` are shared between classification and reduction. Passing reduction-specific aliases (`--reduction_k`, `--reduction_distance`, `--reduction_voting`) is strictly rejected with `ValueError` to eliminate ambiguity.
  - **Non-kNN models parameter separation:** In `SVM`, `RandomForest`, and `MajorityBaseline`, reduction uses `--reduction_k` (default 3), `--reduction_distance` (default `"euclidean"`), and `--reduction_voting` (default `"majority"`); passing `--k` raises `ValueError` instructing the user to pass `--reduction_k`.
  - Downstream models fit directly on the reduced prototype subset, correctly shrinking training storage and training runtime. (decided)

### Parameters and Validation (Fail-Fast Philosophy)

- **Strict, fail-fast hyperparameter validation.**
  `KNN.parse_params`, `SVM.parse_params`, `RandomForest.parse_params`, and `InstanceReduction` validate all configuration parameters upfront:
  - Unknown parameters raise `ValueError: Unknown <Model> parameters provided: [...]`.
  - Incompatible reduction parameters raise immediate `ValueError`: passing `version` to EENTh, passing `threshold` to FCNN or DROP, or passing either when reduction is `"none"`.
  - **Passing reduction parameters without specifying `--reduction` raises `ValueError`:**
    If any reduction parameter (`reduction_k`, `reduction_distance`, `reduction_voting`, `version`, `threshold`, `distance`, `voting`) is passed when `--reduction` is omitted, execution halts immediately with `ValueError` stating that `--reduction` must be specified, instead of silently dropping the parameter or ignoring user intent.
  - Incompatible parameters passed to `MajorityBaseline` also raise `ValueError`.
    - *Why:* Prevents configuration errors and typos from being silently ignored or causing silent misbehavior during long multi-fold runs. (decided)
- **Default parameter injection and CSV synchronization.**
  Default values are injected for omitted parameters (`k=3`, `distance="euclidean"`, `voting="majority"`, `retention="never"`, `reduction="none"`). Active reduction parameters (`version` for FCNN/DROP, `threshold` for EENTh, `reduction_k` for non-kNN models) are synchronized into `self.params` and serialized in the CSV results table. (decided)

### Contracts for strategies

- **Distances:** `def metric(train, test=None) -> np.ndarray` returning the `(n_test, n_train)` matrix (or
  `(n, n)` with one argument), registered in `DISTANCES` under its CLI name.
- **Voting:** `def policy(neighbors: Sequence[Neighbor]) -> int` with a deterministic tie-break, registered
  in `VOTING` (plus its score function in `VOTE_SCORES` for ROC / PR).
- **Retention:** `def policy(prediction: int, label: int) -> bool` (retain the solved case?), registered in
  `RETENTION`; OG is an `OblivionByGoodness` object because it also forgets cases.
- Strategies must never modify `train` or `test` (folds are cached by the runner). (decided)

### Distances

- **Five metrics: euclidean (Minkowski r=2), manhattan (r=1), canberra, clark and heom.**
  Numeric attributes come from `X_num`; nominal values are compared by equality through their
  one-hot columns. All feature weights are 1, as the assignment asks. (decided)
- **A different nominal value adds 2** in euclidean (squared), manhattan, canberra and clark
  (two one-hot columns differ by 1 each), **and 1 in heom** (overlap). (note)
- **Canberra and Clark use the term |x - y| / (|x| + |y|).**
    - *Why:* it equals the slides' |x - y| / |x + y| for our non-negative, [0,1]-scaled data. When
      both values are 0 the term is 0/0, which we count as 0: equal values add no distance. (decided)
- **Clark is implemented as the slides write it, without the usual square root.**
    - *Why:* the square root is monotonic, so the k nearest neighbours are the same either way;
      only the inverse-distance weights change. (decided)
- **Canberra and Clark exaggerate small differences near 0.** They measure relative differences,
  and min-max scaling puts every attribute's train minimum at 0, so 0.05 vs 0.0 already gives the
  maximum term (1). (limitation)
- **Canberra, not Clark, entered the 96-configuration v1 grid.** The assignment asks for one of them;
  Canberra has an unambiguous formula in Wilson & Martinez (1997). Grid search v2 includes Clark as well
  (120 configurations). (decided)
- **heom** (Wilson & Martinez, 1997, section 2.3): numeric |x - y| / range, where range = 1 after
  our min-max scaling, plus overlap for nominal values. The "unknown -> 1" rule never applies
  because missing values are imputed first (see *Missing values*). (decided)
- **On splice, all metrics give identical predictions.** Every attribute is nominal, so each
  metric is an increasing function of the number m of different values (sqrt(2m), 2m, sqrt(m))
  and picks the same k nearest neighbours. With inverse distance the weights could differ, but in
  the grid the four metrics agree fold by fold in all 24 configurations each. (note)

### Voting

- **Inverse distance weighted vote: weight 1/d (p = 1).** Each class sums the weights of its
  neighbours. (decided)
- **Neighbours at distance 0 vote alone, with equal weight.**
    - *Why:* 1/0 is undefined, and an identical case should decide. Among them the vote is a
      majority vote. (decided)
- **Ties in total weight go to the tied class of the nearest neighbour**, the same rule as in
  majority voting, so both schemes always have a single winner. Ties are exact equality of the
  sums. (decided)

### Retention

- **Policies: never (NR), always (AR), different class (DF, like IB2) and oblivion by goodness
  (OG)**, from the course slides. (decided)
- **With retention, the test cases are solved one by one, in file order, and retained with their
  true class.**
    - *Why:* DF needs the true class to know that a case was misclassified; we use it in every
      policy for consistency. *Limitation:* the results depend on the order of the test cases. (decided)
- **NR uses the batch path (all distances at once).** A test checks that the one-by-one loop with
  NR gives exactly the same predictions. (decided)
- **Storage is the size of the case base at the end of the test phase**, reported as % of the
  training set, so AR, DF and OG can exceed 100%. The runner accepts up to train + test cases. (decided)
- **OG: every solved case is retained; a case is forgotten when its goodness drops below the
  threshold.** Initial goodness 0.5, alpha 0.2, threshold 0.5 (issue #11). After each query only
  its k retrieved neighbours update: G <- G + alpha * (r - G), with r = 1 if the neighbour's class
  equals the query's true class. (decided)
- **With initial goodness = threshold, a single wrong retrieval (0.5 -> 0.4) forgets a case.**
  This is what the assignment specifies ("a case is removed whenever Goodness_t(c) < Goodness_0(c)",
  Section 1.2, 3c). (decided)
- **OG never forgets a case if that would leave fewer than k cases.** (decided)

## Hyperparameter Tuning (Grid Search v1, removed)

- **`experiments/grid_search.py` (v1) and its results were removed on 23 Oct**; grid search v2 (below) replaces
  them and is the only one delivered (two grid searches would confuse the reader). v1 stays in the git history.
  The rules of this section that are not specific to v1 still apply to v2: datasets are never merged, the 10
  folds are the blocks, and the optimization-bias limitation. (decided)

### Manual Implementation (`experiments/grid_search.py`, v1)

- **Implemented a custom Grid Search script instead of using `sklearn.model_selection.GridSearchCV`.** The script
  generates a Cartesian product of parameters, prunes redundant configurations, and executes the CLI `run.py` command
  for each valid combination.
    - *Why:* The assignment strictly requires evaluation across the 10 predefined train/test folds, exporting results to
      a specific 9-column CSV schema. A custom script ensures total control over the fold execution while maintaining
      the project's single-responsibility pipeline intact. (decided)
- **Kernel-based parameter pruning.** The script simulates the SVM kernel logic to prune inactive parameters (e.g.,
  dropping `gamma` when the kernel is `linear`) before running them.
    - *Why:* Prevents the grid from evaluating mathematically identical models multiple times, massively reducing
      computation time and keeping the output logs clean of duplicate configurations. (decided)
- **Resilience and Checkpointing.** Each configuration is saved as an individual CSV. The script checks if a
  configuration file already exists before running it.
    - *Why:* If the intensive Grid Search process is interrupted, it can resume exactly where it left off without
      re-computing past folds. (decided)

### Champion Extraction and P-Hacking Avoidance

- **Configurations are grouped into fundamental "families" before statistical testing.** Instead of comparing all 168
  SVM or 96 RF configurations simultaneously, we define architectural families and extract only the absolute best
  configuration (the "Champion") from each family.
    - **Random Forest (4 Champions):** Grouped by `criterion` (gini/entropy) and `max_features` (sqrt/None). *Why:*
      These parameters fundamentally alter the geometry and split logic of the trees. The rest (`n_estimators`,
      `max_depth`...) are treated as internal variance stabilizers.
    - **Support Vector Machine (6 Champions):** Grouped by `kernel` (linear/rbf/poly) and `class_weight`
      (None/balanced). *Why:* `kernel` fundamentally alters the mathematical mapping space, and `class_weight`
      completely alters the loss function. The rest (`C`, `gamma`...) are internal regularizers. (decided)
- **Champions are extracted using raw Mean Accuracy across the 10 folds.**
    - *Why:* **To strictly avoid p-hacking (double-dipping).** Statistical tests mathematically assume that the
      hypotheses are formulated *a priori*. If we used a statistical test to filter out bad models, and then ran another
      statistical test on the survivors, the family-wise error calculation would be completely invalidated. Selecting
      champions based purely on arithmetic mean guarantees the mathematical purity of the subsequent inferential
      statistics. (decided)
    -

### Cross-Model Comparison (The Inter-Model Tournament)

- **Direct Wilcoxon head-to-head for absolute bests.** The absolute #1 RF and absolute #1 SVM can be compared directly
  using the Wilcoxon signed-ranks test.
    - *Why:* When isolating exactly two models, Wilcoxon is the most powerful and appropriate non-parametric test.
      (decided)

### Why datasets are NEVER merged for statistical testing

- **Splice and Vowel folds are never concatenated into an $N=20$ block table.** All statistical tests are strictly
  scoped to $N=10$ blocks of a single dataset.
    - *Why (Assignment Rules):* The guidelines explicitly require hyperparameter and algorithm evaluation "at each
      dataset" (e.g., "Which is the best kernel for the SVM at each dataset?").
    - *Why (Statistical Purity):* Demšar explicitly warns that error rates across unrelated domains are incommensurable.
      The Friedman test assumes blocks (rows) are exchangeable repetitions. Ten folds from a continuous, 11-class
      acoustic problem (Vowel) and ten folds from a categorical, imbalanced genomic problem (Splice) share entirely
      different error variances. Merging them violates the foundational assumptions of the test.
    - *Why (Sample Size):* Demšar's true cross-dataset methodology requires treating each dataset as a single block.
      With only 2 datasets available, $N=2$ is mathematically too small to run Wilcoxon or Friedman (which require at
      least $N=5$ or $N=6$ just to reach a theoretical $p < 0.05$). (decided)

### Known Limitations regarding Evaluation

- **Optimization Bias (Hyperparameter Leakage).** Because we are running our Grid Search across the 10 predefined test
  folds and selecting the hyperparameters that yield the highest mean accuracy, the test sets are effectively acting as
  validation sets.
    - *Consequence:* This introduces a slight optimization bias. The reported maximum accuracies represent an optimistic
      upper bound rather than completely unseen generalizability. Since the assignment forbids re-splitting or modifying
      the provided 10 folds, this bias is an unavoidable constraint of the experimental design. (limitation)
- **Double Dipping**: When applying the Wilcoxon test upon the top models extracted with the Nemenyi test, we are
  actually performing p-hacking. It cannot be avoided as we cannot divide the dataset into train-val-test and we are
  forced to do it this way.

### Best kNN (issue #15, grid search v1)

- *Superseded by grid search v2* (120 configurations, see below); kept as the record of the v1 run.
- **Grid: 96 configurations** = k {3, 5, 7} x distance {euclidean, manhattan, canberra, heom}
  x voting {majority, inverse_distance} x retention {never, always, different_class,
  oblivion_by_goodness}. Champions for the statistical tests: the best configuration of each
  retention x voting pair (8 champions). (decided)
- **Selection rule, fixed before looking at the winner:** among the configurations statistically
  equivalent to the best one (the Nemenyi top group, or every champion when Friedman does not
  reject), take the lowest storage, then the highest mean accuracy, then the first distance in
  grid order.
    - *Why:* without a significant accuracy difference, the cheaper model is preferred. (decided)
- **Result:** vowel -> euclidean, k=3, inverse_distance, never (0.976, storage 100%; Friedman
  p = 0.81, all champions equivalent). splice -> euclidean, k=7, inverse_distance,
  oblivion_by_goodness (0.844, storage 89.7%): best mean rank, best accuracy and lowest storage
  of the Nemenyi top group. (decided)
- **k works in opposite directions:** accuracy falls with k on vowel (near-duplicate repetitions
  of the same speaker) and rises with k on splice (noisy attributes). (note)
- **OG acts like an editing method on splice:** it forgets more cases than it retains (final
  case base 89.7% of the train) and improves accuracy. On vowel it does not help. (note)
- **The champions are chosen on the same 10 folds used for the tests**, so the selected
  configuration's accuracy is an optimistic estimate (selection bias). (limitation)

---

## Grid search v2 (`experiments/grid_search_v2.py`)

One script runs every experiment of the assignment, with and without instance reduction, and answers each
question of Section 1.3 with a pre-defined test. It replaces `experiments/grid_search.py` (v1, removed).

### Ground rules (decided)

- **Each dataset on its own; the 10 folds are the blocks** of every test (N = 10). splice and vowel are never
  merged (see "Why datasets are NEVER merged").
- **Selection rule fixed before seeing results:** the winner of any group is the highest mean accuracy over the
  10 folds; ties -> lower mean `time` -> configuration name. Tests never pick the winner; they say whether it can
  be distinguished from the alternatives.
- **Test by number of configurations** (Demšar 2006): 2 -> Wilcoxon signed-ranks (exact, t-test as reference);
  >= 3 -> Friedman with Iman-Davenport, and only if p < 0.05 a post-hoc: Nemenyi (all pairs) or Bonferroni-Dunn
  when there is a natural control (the unreduced model). Control comparisons made with Wilcoxon are Holm-corrected.
  alpha = 0.05. A CD diagram is saved for every significant Friedman.
- **Equal conditions for time:** default `--jobs 1` (parallel processes compete for the CPU and inflate times) and
  RF with `n_jobs=1`, like our kNN. `--jobs N` is fine to get accuracies fast, but then do not use the times.
- **Reproducible and resumable:** seed reset per configuration (`run_experiment`), one CSV per configuration
  written atomically (temp file + rename), existing CSVs are skipped.
- **Reduction cache** (`results/gridsearch_v2/reduction_cache/`, `reduce_with_cache` in `reduction.py`): the
  reduced set depends only on the technique, its variant, K, distance, voting and the training fold, never on the
  model trained on it or on the kNN retention. Phase 3 reuses the four reductions of the best kNN for SVM and RF
  (on splice ~35 s per fold each time, DROP3 and DEL dominate). The key is a hash of those settings and of the
  training arrays; all techniques are deterministic, so a hit gives exactly the same instances.
    - *Time stays comparable:* on a hit, `reduction_time` is the time measured when the reduction was computed and
      `time` is charged with it (`run_fold` adds the difference), so the CSVs mean the same with or without the
      cache. Delete the folder to force recomputation. (decided)
    - *No global state:* the folder is an explicit attribute of each classifier (`reduction_cache_dir`, default
      `None` = no cache), set by `grid_search_v2.py` when it creates the classifier; `reduce_with_cache` receives it
      as an argument. (decided)
- **In-process execution** (`run_experiment` directly, optional process pool) instead of one subprocess per
  configuration as in v1: each process parses and preprocesses every fold once (`load_fold` cache).

### Pipeline (per dataset)

`--stage run` (three phases, each needs the winners of the previous one), then `--stage evaluate`.

| Phase | What runs | Size |
|---|---|---|
| 1. base | kNN full grid: K {3,5,7} x 5 distances x 2 votes x 4 retentions; SVM grid (C, kernel linear/rbf/poly, gamma, class_weight, coef0/degree for poly); RF grid (n_estimators {50..300}, gini/entropy, max_depth, max_features, min_samples_leaf) | 120 + 168 + 96 |
| 2. kNN reduction | every reduction variant on the **top-`--reduce-top` kNN** (default 3): FCNN v1/v2 (condensed), EENTh threshold 0/0.5/0.7 (edited), DROP3 and DEL (hybrids) | 7 x 3 |
| 3. SVM/RF reduction | best SVM and best RF trained on the reduced set of each technique (its best variant on the best kNN, with the best kNN's K, distance and voting) | 4 x 2 |

- **Why the full kNN, SVM and RF grids and not only top candidates:** they are cheap (kNN ~0.1 s per fold, SVM/RF
  seconds), so pruning is only needed for the *tests*, not for running.
- **Why reduction only on the top kNN configurations:** it is what the assignment asks (`reduction_kNNAlgorithm` =
  best kNN + reduction) and reducing is expensive (splice fold, K=3: FCNN 1.4 s, EENTh 0.5 s, DROP3 12 s, DEL 21 s).
  The top-3 instead of only the winner (both modes in one run; `--reduce-top 1` gives only the winner) checks
  whether the ranking of the kNN configurations survives the reduction (descriptive table, no test).
- **Why the SVM/RF reduced set uses the kNN winner's settings and the variant chosen on the kNN:** the three
  models are compared on the same prototypes, and the variant is not chosen on SVM/RF results (no second
  selection on the data we test them on).
- **DROP4 is not in the grid:** our implementation is O(m n^2 log n) in Python (16 s per fold on vowel, far more on
  splice). DROP3 and DEL cover the hybrid family. **RNN, GCNN, RENN, RNGE** are not implemented.

### Tests and the question each one answers (`--stage evaluate`)

| # | Question (Section 1.3) | Columns compared | Test |
|---|---|---|---|
| 1 | Best K / similarity function / is voting useful / retention, for kNN | levels of each factor: (a) **main effect** = per fold, mean accuracy over all configurations of the level; (b) **best configuration of each level** | Wilcoxon (2 levels) or Friedman + Nemenyi |
| 2 | Best kernel / class_weight for SVM | same (a) and (b) for `kernel`, `class_weight` | idem |
| 3 | Best criterion / n_estimators / max_features for RF | same (a) and (b) | idem |
| 4 | Is the best kNN / SVM / RF distinguishable from the runner-ups? | top-5 configurations of the model | Friedman + Nemenyi |
| 5 | SVM and RF in front of the best kNN (accuracy and efficiency) | best kNN, best SVM, best RF | Friedman + Nemenyi on accuracy and on time; Wilcoxon kNN vs SVM and kNN vs RF with Holm |
| 6 | Best reduction technique for kNN, or reduction does not help | best kNN without reduction (control) + best variant of FCNN, EENTh, DROP3, DEL | Friedman + Bonferroni-Dunn vs control, on accuracy and on `solve_time`; storage as a table |
| 7 | Retention vs reduction | best configuration of each retention policy + reduced best kNN (4 techniques) | Friedman + Nemenyi on accuracy; storage and time as a table |
| 8 | Does reduction change SVM / RF? | best SVM (or RF) full (control) + the 4 reduced versions | Friedman + Bonferroni-Dunn on accuracy and `solve_time` |
| 9 | kNN vs SVM vs RF when all are reduced | per technique: kNN, SVM, RF on the same reduced set | Friedman + Nemenyi on accuracy and time |

Feature selection vs sample reduction (and combining them) is answered by reasoning only, as the assignment says.

- **Main effect and best-per-level, both:** the best-per-level column is what "best K" literally means, but it is
  chosen on the same folds it is tested on, and a level with more configurations has more chances to win (e.g.
  the poly kernel has 144 SVM configurations, linear 8). The main effect involves no selection at all, so it is the
  clean test; when both agree the conclusion is solid, when they disagree we report it (the factor helps only
  when tuned together with the others).
- **Why not Friedman over the whole grid:** with k = 120 (kNN) or 168 (SVM) and N = 10 the Nemenyi CD is far larger
  than any rank gap (power loss); the factor tests and the top-5 test keep k small.
- **Efficiency:** `time` (fit + predict, including the reduction) for the model comparisons, because an eager model
  pays its training at solve time too; `solve_time = time - reduction_time` for the reduction comparisons, because
  the reduction is a one-off cost and the question is whether the reduced case base is faster to use.
- **Metric for the tests: accuracy** (as the assignment). `macro_f1` is reported in every table; for splice
  (imbalanced) we look at it when accuracy is close.

### Double dipping: where it is avoided and where it is not

- **Unavoidable:** with no validation set, the configurations are selected on the same 10 test folds that the tests
  use. The tests 4-9 compare *selected* winners, so their p-values are optimistic (the best of many looks better
  than it is) and the reported accuracies are an upper bound (see "Optimization Bias").
- **Avoided where possible:** (1) the main-effect tests involve no selection; (2) the reduction variant is chosen
  on kNN and then applied, not re-chosen, to SVM and RF; (3) the selection rule is fixed in advance and the tests
  never choose the winner, so we do not test, filter by p-value and test again on the survivors; (4) the top-5
  test is reported as "can the winner be told apart from its runner-ups", which is the conservative reading.
- **Considered and rejected:** select on folds 0-4 and test on folds 5-9. It is honest, but with N = 5 the exact
  Wilcoxon cannot reach p < 0.05 (smallest two-sided p = 2/32 = 0.0625) and Friedman loses most of its power.

### Outputs (`results/gridsearch_v2/`)

- `<dataset>/base/<model>/*.csv`, `<dataset>/reduction/<model>/*.csv`: one CSV per configuration (the name is a
  hash of the parameters; the parameters are inside the CSV).
- `<dataset>/summary.csv`: one row per configuration (mean accuracy, std, macro F1, time, solve time, storage).
- `<dataset>/tests.csv` and `<dataset>/report.md`: every test (question, k, statistic, p-value, winner, post-hoc)
  and the tables per question. `<dataset>/figures/cd_*.png`: CD diagrams.
- `winners.json`: parameters of the winners (best kNN / SVM / RF, best variant of each technique, best reduction,
  and the reduced winners) for `plot_winners.py`.

### Winner plots (`experiments/plot_winners.py`)

- **Retrain the winners instead of saving every model during the grid.** Retraining a winner on 10 folds takes
  seconds and is exact (fixed seeds; the script checks that the accuracy equals the grid search). Saving every
  fitted model of the grid (~400 configurations x 10 folds, RF up to 300 trees) would take GBs to keep ~6 of them.
  What the plots need are the scores, so those are cached per winner (`plots/cache/*.npz`); plots can be redrawn
  without retraining. The kNN "model" is just its (reduced) case base anyway. (decided)
- **ROC / PR pool the out-of-fold scores of the 10 test folds** (each instance scored once). Multiclass is
  one-vs-rest: ROC macro-averaged, PR micro-averaged; legends give macro AUC / macro AP. *Limitation:* SVM decision
  values of different folds are not on exactly the same scale; pooling is the usual simplification. kNN scores
  are vote shares (few distinct values for small K), so its curves have few steps. (decided)
- `sklearn.metrics` is used only to draw the curves of every model, never inside the kNN or the reduction.
- Plots: `roc_pr_<dataset>.png`, `confusion_<dataset>.png` (row-normalised colour, counts for splice),
  `tradeoff_<dataset>.png` (accuracy vs storage and vs solve time for each model and technique),
  `reduction_<dataset>.png` (per technique: accuracy ± std over folds, solve time, reduction time, storage;
  read from the CSVs, no retraining). The written analysis of #19 / #20 is `report/reduction_analysis.md`.

---

## Methods and libraries used


| Area                                 | Method / strategy                                                                                                   | Implemented with                                                                                 | Where                                 |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------|
| Reading ARFF                         | Parse header and rows; keep features, missing mask and class apart                                                  | `scipy.io.arff.loadarff` + own wrapper (`TrainMatrix` / `TestMatrix`)                            | `src/arff_parser.py`                  |
| Fold handling                        | The 10 predefined train/test pairs, read as they are                                                                | Own code (`config.fold_paths`, `load_fold`)                                                      | `src/config.py`, `src/arff_parser.py` |
| Dropped columns                      | Configurable list, defaults per dataset                                                                             | Own code (`DEFAULT_DROP`)                                                                        | `src/preprocessing.py`                |
| Numeric missing values               | Train mean                                                                                                          | `sklearn.impute.SimpleImputer(strategy="mean", keep_empty_features=True)`                        | `src/preprocessing.py`                |
| Numeric scaling                      | Min-max to [0,1] with the train range, test clipped                                                                 | `sklearn.preprocessing.MinMaxScaler(clip=True)`                                                  | `src/preprocessing.py`                |
| Constant numeric column              | Set to 0                                                                                                            | Own code (numpy), after the scaler                                                               | `src/preprocessing.py`                |
| Nominal missing values               | Train mode on the text values; ties go to the first in alphabetical order                                           | `sklearn.impute.SimpleImputer(strategy="most_frequent")`                                         | `src/preprocessing.py`                |
| Columns with too many missing values | More than `max_missing` (50%) in train: `ValueError` listing all of them, or drop with `on_missing="drop"`          | Own code (check before the pipelines)                                                            | `src/preprocessing.py`                |
| Rows with an unknown class           | Removed with a warning                                                                                              | Own code (boolean mask)                                                                          | `src/preprocessing.py`                |
| Nominal encoding                     | Integer code = position in the ARFF header, one column per attribute; a value not in the header raises `ValueError` | `sklearn.preprocessing.OrdinalEncoder(categories=header)`                                        | `src/preprocessing.py`                |
| No columns of one type               | Skip that pipeline (sklearn rejects 0 columns; splice has no numeric ones)                                          | Own code (`if`)                                                                                  | `src/preprocessing.py`                |
| Chaining the steps                   | Fit on train, transform train and test                                                                              | `sklearn.pipeline.make_pipeline` (one pipeline for numeric, one for nominal)                     | `src/preprocessing.py`                |
| One-hot for SVM / RF                 | One 0/1 column per header value, built from the integer codes                                                       | Own code (`np.eye`) in `Preprocessed.to_onehot()`                                                | `src/preprocessing.py`                |
| Class labels                         | Kept unchanged; integer code = position in the header                                                               | Own code (dict lookup)                                                                           | `src/preprocessing.py`                |
| Single ML Entry point                | Created a single entry point for all the ML model runs                                                              | Own code                                                                                         | `experiments/run.py`                  |
| Experiment runner                    | Run the 10 predefined folds and collect common evaluation metrics                                                   | Own code                                                                                         | `src/experiment_runner.py`            |
| Result export                        | Save experiment results using the common 9-column CSV schema                                                        | Python `csv`                                                                                     | `src/experiment_runner.py`            |
| Random Forest model                  | Train an ensemble of decision trees using one-hot encoded nominals                                                  | `sklearn.ensemble.RandomForestClassifier`                                                        | `src/models/random_forest.py`         |
| Support Vector Machine model         | Train a Support Vector Machine using one-hot encoded nominals                                                       | `sklearn.svm.SVC`                                                                                | `src/models/svm.py`                   |
| kNN utilities & runner               | `kNNAlgorithm` / `run_kNN`: vectorized distance matrix, stable tie-breaking, batch and retention modes | Own code + `numpy`                                                                               | `src/models/knn/knn_utils.py`        |
| kNN adapter                          | `KNN(BaseClassifier)`, validates hyperparameters, prunes inactive keys, integrates reduction and retention        | Own code                                                                                         | `src/models/knn/knn.py`              |
| kNN distances                        | Euclidean, Manhattan, Canberra, Clark, HEOM (registry `DISTANCES`); one argument gives train x train   | Own code + `numpy`                                                                               | `src/models/knn/distances.py`        |
| kNN voting                           | Majority (nearest-tied tie-break) and Inverse Distance (registry `VOTING`)                                         | Own code                                                                                         | `src/models/knn/voting.py`           |
| kNN retention                        | Never, Always, Different Class, Oblivion by Goodness with neighbor score reinforcement (registry `RETENTION`)        | Own code                                                                                         | `src/models/knn/retention.py`        |
| Instance reduction                   | FCNN (rules 1..4), EENTh (threshold), DROP (versions 1..5) (registry `REDUCTION_TECHNIQUES`)                        | Own code + `numpy`                                                                               | `src/models/knn/reduction.py`        |
| Grid search v2                       | Full pipeline with and without reduction, tests per question, winners                                              | Own code + `concurrent.futures` (stdlib)                                                         | `experiments/grid_search_v2.py`       |
| Winner plots                         | ROC (macro OvR), PR (micro), confusion matrices, accuracy vs storage / time                                          | `sklearn.metrics` (curves only, not in the kNN), `matplotlib`                                    | `experiments/plot_winners.py`         |
| Holm correction                      | Step-down adjusted p-values for control comparisons                                                                 | Own code                                                                                         | `src/stats.py`                        |
| kNN sanity check (tests only)        | Compare predictions with a reference kNN                                                                            | `sklearn.neighbors.KNeighborsClassifier` (only in `tests/`)                                      | `tests/test_knn.py`                   |
| EDA figures                          | Class counts, histograms, nucleotide heatmaps                                                                       | `matplotlib`, `seaborn`                                                                          | `experiments/eda.py`                  |
| EDA projection                       | PCA in 2D                                                                                                           | Own code (`numpy.linalg.svd`)                                                                    | `experiments/eda.py`                  |
| EDA export                           | Preprocessed fold as CSV (one-hot + class)                                                                          | `pandas`                                                                                         | `experiments/eda.py`                  |
| Statistical tests                    | Wilcoxon (exact), Friedman + Iman-Davenport, Nemenyi, Bonferroni-Dunn, paired t-test                                | Own code + `scipy.stats` distributions (`chi2`, `f`, `norm`, `studentized_range`, `ttest_1samp`) | `src/stats.py`                        |
| CD diagram                           | Demšar (2006), Figure 1                                                                                             | `matplotlib`                                                                                     | `src/stats.py`                        |

Not used (on purpose):

- **scikit-learn in the kNN or in the instance reduction.** Both must be our own code.
- **`sklearn.preprocessing.OneHotEncoder`.** The one-hot is built with `np.eye` from the integer
  codes we already have, so both encodings always agree.
- **`sklearn.preprocessing.LabelEncoder` for the class.** It sorts the labels alphabetically;
  we want the header order.
- **`sklearn.compose.ColumnTransformer`.** Numeric and nominal columns go to separate arrays (`X_num`, `X_nom`), so two
  small pipelines are simpler.
- **`sklearn.decomposition.PCA`.** The EDA needs one projection; a numpy SVD is enough.
- **`StandardScaler` / z-score.** The assignment asks for [0,1].
- **Discretization (`pandas.qcut`) and missing-indicator columns.** Optional and not needed here.
- **Re-splitting (`train_test_split`, `KFold`).** The folds are predefined.
- **`scipy.stats.friedmanchisquare` and `scipy.stats.wilcoxon` for the decisions.** They handle
  ties and zeros differently from Demšar (2006); `wilcoxon` is only used in a test as a check.
- **`scikit-posthocs`.** Not in the allowed packages; Nemenyi, Bonferroni-Dunn and the CD
  diagram are a few lines of our own code.

---

## Pending decisions

Closed (23 Oct): **HEOM** is the metric "you may choose" (implemented and in both grids; best kNN distance on
vowel). **OG parameters** follow the assignment (Section 1.2, 3c): G_0 = 0.5, alpha = 0.2, and a case is removed
whenever G_t < G_0, so one wrong retrieval (0.5 -> 0.4) forgets it; that is the specified behaviour.

| Topic                                    | Proposal                                                                                  | Needs                                   |
|------------------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------|
| vowel columns                            | See "Dropped columns" (A/B/C comparison)                                                  | Team: run it, or state in the report that it is out of scope |
