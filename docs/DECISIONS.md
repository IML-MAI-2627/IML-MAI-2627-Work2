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
- **One-hot version only for SVM, RF and the numeric-only distances** (`Preprocessed.to_onehot()`).
    - *Why:* those methods would interpret integer codes as ordered quantities (a false order).
      Our nominal attributes have no natural order (A/C/G/T, Male/Female): the integer codes
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
  `dataset, fold, model, params, accuracy, n_correct, n_incorrect, time, storage`.
    - *Why:* all kNN, SVM, Random Forest, and reduced-model experiments can be stored and compared using the same
      structure. (decided)
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

## Methods and libraries used

| Area                                 | Method / strategy                                                                                                   | Implemented with                                                             | Where                                 |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------------------------|
| Reading ARFF                         | Parse header and rows; keep features, missing mask and class apart                                                  | `scipy.io.arff.loadarff` + own wrapper (`TrainMatrix` / `TestMatrix`)        | `src/arff_parser.py`                  |
| Fold handling                        | The 10 predefined train/test pairs, read as they are                                                                | Own code (`config.fold_paths`, `load_fold`)                                  | `src/config.py`, `src/arff_parser.py` |
| Dropped columns                      | Configurable list, defaults per dataset                                                                             | Own code (`DEFAULT_DROP`)                                                    | `src/preprocessing.py`                |
| Numeric missing values               | Train mean                                                                                                          | `sklearn.impute.SimpleImputer(strategy="mean", keep_empty_features=True)`    | `src/preprocessing.py`                |
| Numeric scaling                      | Min-max to [0,1] with the train range, test clipped                                                                 | `sklearn.preprocessing.MinMaxScaler(clip=True)`                              | `src/preprocessing.py`                |
| Constant numeric column              | Set to 0                                                                                                            | Own code (numpy), after the scaler                                           | `src/preprocessing.py`                |
| Nominal missing values               | Train mode on the text values; ties go to the first in alphabetical order                                           | `sklearn.impute.SimpleImputer(strategy="most_frequent")`                     | `src/preprocessing.py`                |
| Columns with too many missing values | More than `max_missing` (50%) in train: `ValueError` listing all of them, or drop with `on_missing="drop"`          | Own code (check before the pipelines)                                        | `src/preprocessing.py`                |
| Rows with an unknown class           | Removed with a warning                                                                                              | Own code (boolean mask)                                                      | `src/preprocessing.py`                |
| Nominal encoding                     | Integer code = position in the ARFF header, one column per attribute; a value not in the header raises `ValueError` | `sklearn.preprocessing.OrdinalEncoder(categories=header)`                    | `src/preprocessing.py`                |
| No columns of one type               | Skip that pipeline (sklearn rejects 0 columns; splice has no numeric ones)                                          | Own code (`if`)                                                              | `src/preprocessing.py`                |
| Chaining the steps                   | Fit on train, transform train and test                                                                              | `sklearn.pipeline.make_pipeline` (one pipeline for numeric, one for nominal) | `src/preprocessing.py`                |
| One-hot for SVM / RF                 | One 0/1 column per header value, built from the integer codes                                                       | Own code (`np.eye`) in `Preprocessed.to_onehot()`                            | `src/preprocessing.py`                |
| Class labels                         | Kept unchanged; integer code = position in the header                                                               | Own code (dict lookup)                                                       | `src/preprocessing.py`                |
| Single ML Entry point                | Created a single entry point for all the ML model runs                                                              | Own code                                                                     | `experiments/run.py`                  |
| Experiment runner                    | Run the 10 predefined folds and collect common evaluation metrics                                                   | Own code                                                                     | `src/experiment_runner.py`            |
| Result export                        | Save experiment results using the common 9-column CSV schema                                                        | Python `csv`                                                                 | `src/experiment_runner.py`            |
| Random Forest model                  | Train an ensemble of decision trees using one-hot encoded nominals                                                  | `sklearn.ensemble.RandomForestClassifier`                                    | `src/models/random_forest.py`         |
| Support Vector Machine model         | Train a Support Vector Machine using one-hot encoded nominals                                                       | `sklearn.svm.SVC`                                                            | `src/models/svm.py`                   |
| EDA figures                          | Class counts, histograms, nucleotide heatmaps                                                                       | `matplotlib`, `seaborn`                                                      | `experiments/eda.py`                  |
| EDA projection                       | PCA in 2D                                                                                                           | Own code (`numpy.linalg.svd`)                                                | `experiments/eda.py`                  |
| EDA export                           | Preprocessed fold as CSV (one-hot + class)                                                                          | `pandas`                                                                     | `experiments/eda.py`                  |

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

---

## Pending decisions

| Topic                                                        | Proposal                                 | Needs                             |
|--------------------------------------------------------------|------------------------------------------|-----------------------------------|
| Distance chosen as "one you may choose"                      | HEOM (simple, no train statistics)       | Team decision before the kNN (#6) |
| Nominal attributes in Euclidean / Manhattan / Clark-Camberra | Use the one-hot version                  | Team decision before the kNN (#6) |
| Clark / Camberra with 0/0                                    | Define the term as 0                     | Implement in the kNN              |
| vowel columns                                                | See "Dropped columns" (A/B/C comparison) | After the kNN exists              |
