# IML Work 2 — kNN vs SVM vs Random Forest

Course project for *Introduction to Machine Learning* (MAI, 2026-2027). We study
kNN (implemented from scratch) and compare it with SVM and Random Forest,
including the effect of instance reduction on accuracy, efficiency and storage.

Datasets: `splice` (3190 cases, nominal) and `vowel` (990 cases, numeric + nominal),
both with predefined 10-fold train/test splits in `datasets/`.

## Requirements

- Python 3.11
- Only the packages in `requirements.txt` (pinned versions).

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the tests

All unit and integration tests are written for **`pytest`** (using pytest fixtures and parameterized test functions). Run the full test suite using:

```bash
pytest
```

> **Note**: Do not use `python -m unittest discover`, as `unittest` only recognizes `unittest.TestCase` subclasses and will discover 0 tests. Always run tests using `pytest`.

## Running the experiments

Experiments use the predefined 10-fold train/test splits provided in
`datasets/`. The folds must not be modified.

The experiment runner preprocesses each fold, runs the selected classifier,
and records the classification accuracy, number of correct and incorrect
predictions, execution time (excluding data preprocessing), and storage. It
resets the project seed (`42`) at the beginning of every 10-fold experiment.
The time covers the classifier call, including model fitting when a classifier
does it internally; parsing, preprocessing and CSV writing are excluded.

Results can be saved using the following common CSV schema:

```text
dataset,fold,model,params,accuracy,n_correct,n_incorrect,time,storage,reduction_time,macro_f1,confusion_matrix
```

For reduction experiments, storage is the percentage of the original
training instances retained by the classifier. A classifier using the complete
training set therefore has a storage of 100%.

- `reduction_time`: seconds of `time` spent reducing the training set (0 without reduction);
  `time - reduction_time` is the time to classify the test fold.
- `macro_f1`: mean of the per-class F1 (useful on the imbalanced splice).
- `confusion_matrix`: JSON list of lists, rows = true class, columns = predicted class, in the
  order of the ARFF header. Read it with `experiment_runner.parse_confusion_matrix`.

### Running the Models

The main entry point for all experiments is `experiments/run.py`. It uses a single command-line interface
to execute any implemented model across the 10 folds of a given dataset.

**General Command Structure:**

```bash
python experiments/run.py --dataset <name> --classifier <model_id> [model_hyperparameters...]
```

**Required Core Parameters:**

- --dataset (string): The dataset to use. Valid values: vowel, splice.

- --classifier (string): The model to execute. Valid values: baseline, knn, rf, svm.

- --out (optional, path): Destination for the CSV results. Defaults to results/<classifier>_<dataset>.csv.

**Model-Specific Hyperparameters:**

These parameters are passed as optional flags. If omitted, the model will fall back to its internal defaults.

- Majority Baseline (--classifier baseline)
    - No additional hyperparameters required.

  Example:
    ```bash
    python experiments/run.py --dataset vowel --classifier baseline
    ```

- Random Forest (`--classifier rf`)

    - `--n_estimators` (integer): Number of trees in the forest. Default: Scikit-learn default (100).

    - `--criterion` (string): The function to measure the quality of a split. Must be one of: `gini`, `entropy`, or
      `log_loss`. Default: Scikit-learn default ("gini").

    - `--max_depth` (integer or 'None'): Maximum depth of the trees. Default: Scikit-learn default (None).

    - `--max_features` (integer, float, 'sqrt', 'log2', or 'None'): The number of features to consider when looking for
      the
      best split. Float values represent a fraction. Default: Scikit-learn default ("sqrt").

    - `--min_samples_leaf` (integer): Minimum number of samples required to be at a leaf node. Default: Scikit-learn
      default (1).

    - `--random_state` (integer, None): Seed for the random forest estimator to ensure reproducibility. Default: The
      project's fixed seed (`config.SEED`).

    - `--n_jobs` (integer, None): The number of jobs to run in parallel. fit, predict, decision_path and apply are all
      parallelized over the trees. Default: None

  Example:
    ```bash
      python experiments/run.py --dataset splice --classifier rf --n_estimators 200 --max_depth 10
    ```
- Support Vector Machine (`--classifier svm`)

    - `--C` (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
      Default: Scikit-learn default (1.0).

    - `--kernel` (string): Specifies the kernel type to be used. Must be one of: `linear`, `poly`, `rbf`, or `sigmoid`.
      Default: Scikit-learn default ("rbf").

    - `--gamma` (float, 'scale', or 'auto'): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Default: Scikit-learn
      default ("scale").

    - `--degree` (integer): Degree of the polynomial kernel function ('poly'). Ignored by all other kernels. Default:
      Scikit-learn default (3).

    - `--class_weight` ('balanced' or 'None'): Set to 'balanced' to automatically adjust weights inversely proportional
      to class frequencies (useful for imbalanced datasets like `splice`). Default: None.

    - `--coef0` (float): Independent term in kernel function. Only significant in 'poly' and 'sigmoid'. Default: 0.0.

    - `--max_iter` (integer): Hard limit on iterations within solver, or -1 for no limit. Default: -1.

    - `--tol` (float): Tolerance for stopping criterion. Default 1e-3.

    - `--random_state` (integer or 'None'): Seed for probability estimation and shuffling. Default: The project's fixed
      seed (`config.SEED`).

      Example:
        ```bash
        python experiments/run.py --dataset splice --classifier svm --C 1.5 --max_iter 1000
        ```
- k-Nearest Neighbours (`--classifier knn`), own implementation (`run_kNN` in `src/models/knn/knn_utils.py`)

    - `--k` (positive integer): Number of neighbours. Default: 3.

    - `--distance` (string): Metric for neighbor evaluation, selected from `DISTANCES` in `src/models/knn/distances.py`.
      A single-argument call (`dist(train)`) returns the symmetric train x train matrix.
      Must be one of:
      - `euclidean`: Minkowski $r=2$ with nominal overlap on one-hot representation.
      - `manhattan`: Minkowski $r=1$ with nominal overlap on one-hot representation.
      - `canberra`: Relative difference metric with nominal overlap.
      - `clark`: Squared relative difference metric with nominal overlap.
      - `heom`: Heterogeneous Euclidean-Overlap Metric (Wilson & Martinez, 1997).
      Default: `euclidean`.

    - `--voting` (string): Voting policy, selected from `VOTING` in `src/models/knn/voting.py`.
      Must be one of:
      - `majority`: Standard majority vote (ties go to the nearest tied neighbor).
      - `inverse_distance`: Distance-weighted voting with epsilon smoothing for identical instances.
      Default: `majority`.

    - `--retention` (string): Dynamic case-base retention policy during testing, selected from `RETENTION` in `src/models/knn/retention.py`.
      Must be one of:
      - `never`: Static case base; standard batch classification across folds.
      - `always`: Retains every evaluated test case.
      - `different_class`: Retains test cases only when the prediction does not match the true class.
      - `oblivion_by_goodness`: Neighbor-tracked reinforcement learning retention with threshold-based forgetting (`min_cases=k`).
      Default: `never`.

    Examples:
    ```bash
    # Standard batch kNN with Canberra distance and inverse distance voting
    python experiments/run.py --dataset vowel --classifier knn --k 5 --distance canberra --voting inverse_distance

    # kNN with dynamic Oblivion by Goodness retention
    python experiments/run.py --dataset splice --classifier knn --k 3 --retention oblivion_by_goodness
    ```

### Instance Reduction Parameters (Supported Across All Classifiers)

Instance reduction algorithms select a representative subset of training instances $S \subset T$ before model training, drastically reducing memory footprint and prediction latency while pruning noise. Supported across **all classifiers** (`knn`, `svm`, `rf`, `baseline`). Reduction is applied during preprocessing on the training partition before the classifier fits or predicts.

> [!IMPORTANT]
> **Parameter Compatibility & Exclusivity Rules:**
> - **In `knn` (`--classifier knn`)**: Instance reduction automatically uses the classifier's own hyperparameters (`--k`, `--distance`, `--voting`) for both classification and reduction. Passing reduction-specific aliases (`--reduction_k`, `--reduction_distance`, `--reduction_voting`) to `knn` is **strictly forbidden and raises a `ValueError`**.
> - **In non-kNN models (`svm`, `rf`, `baseline`)**: Passing `--k` is **strictly forbidden and raises a `ValueError`** (use `--reduction_k` instead). The neighbor distance metric and voting rule for reduction can be configured with `--distance` (or `--reduction_distance`) and `--voting` (or `--reduction_voting`).
> - **Fail-fast validation**: Passing any reduction parameter (`--reduction_k`, `--version`, `--threshold`, `--distance`, `--voting`, `--reduction_distance`, `--reduction_voting`) without explicitly specifying `--reduction` is **strictly forbidden and raises a `ValueError`** requiring `--reduction`.

**Reduction Hyperparameters:**

- `--reduction` (string): Training set instance reduction technique, selected from `REDUCTION_TECHNIQUES` in `src/models/knn/reduction.py`.
  - `none`: Use the full training set without reduction (default).
  - `fcnn`: Fast Condensed Nearest Neighbor (Angiulli, 2007). Order-independent, training-set consistent condensation.
  - `eenth`: Extended Edited Nearest Neighbor with Threshold (Vázquez & Sánchez, 2005). Prunes noisy and ambiguous borderline instances.
  - `drop`: Decremental Reduction Optimization Procedure (Wilson & Martínez, 2000). Decrementally prunes internal and noisy prototypes.
  - `del`: Decremental Encoding Length (Cameron-Jones, 1995; Wilson & Martinez, 2000 Sections 3.2.8 & 4.6). Minimum Description Length pruning using numerically stable log-space combinatorial sums.
  Default: `none`.

- `--reduction_k` (positive integer): Number of nearest neighbors used by instance reduction when executing non-kNN models (`svm`, `rf`, `baseline`). Default: 3. *(Forbidden for `knn`, which uses `--k`)*.

- `--version` (integer): Algorithm variant/rule. Exclusive to `--reduction fcnn` and `--reduction drop`:
  - For `fcnn`: `1` (each prototype adds its nearest enemy), `2` (each prototype adds the medoid of its enemies), `3` / `4` (as 1 / 2 but one instance per iteration). Default: 1.
  - For `drop`: `1` (DROP1), `2` (DROP2), `3` (DROP3 with ENN filtering), `4` (DROP4), `5` (DROP5). Default: 3.
  *(Forbidden with `--reduction eenth`, `--reduction del`, or `--reduction none`)*.

- `--threshold` (float): Acceptance consensus threshold $\in [0, 1]$. Exclusive to `--reduction eenth`:
  - An instance is removed if its distance-weighted class probability (weights 1/(1+d) over its $k$ nearest neighbours) does not predict its label, or if that maximum probability is <= this threshold (Vázquez, Sánchez & Pla, 2005). Default: 0.0.
  *(Forbidden with `--reduction fcnn`, `--reduction drop`, `--reduction del`, or `--reduction none`)*.

- `--distance` / `--reduction_distance` (string, optional): Distance metric for neighbor identification during reduction in non-kNN models. Defaults to `euclidean`. *(Forbidden as a separate reduction parameter in `knn`, which uses kNN's classification metric)*.

- `--voting` / `--reduction_voting` (string, optional): Voting rule for neighbor consensus during reduction in non-kNN models. Defaults to `majority`. *(Forbidden as a separate reduction parameter in `knn`, which uses kNN's classification voting rule)*.

**Instance Reduction CLI Examples:**

```bash
# kNN with DROP3 reduction (shares --k for classification and reduction)
python experiments/run.py --dataset vowel --classifier knn --k 3 --reduction drop --version 3

# kNN with DEL (Decremental Encoding Length) reduction
python experiments/run.py --dataset vowel --classifier knn --k 3 --reduction del

# kNN with FCNN rule 2 (nearest enemy) and dynamic Oblivion by Goodness retention
python experiments/run.py --dataset splice --classifier knn --k 3 --reduction fcnn --version 2 --retention oblivion_by_goodness

# kNN with EENTh filtering at consensus threshold 0.6
python experiments/run.py --dataset vowel --classifier knn --k 5 --reduction eenth --threshold 0.6

# SVM with DROP3 instance reduction (uses --reduction_k)
python experiments/run.py --dataset vowel --classifier svm --kernel rbf --reduction drop --version 3 --reduction_k 3

# Random Forest with DEL reduction (MDL compression)
python experiments/run.py --dataset vowel --classifier rf --n_estimators 100 --reduction del --reduction_k 3

# Majority Baseline with FCNN reduction
python experiments/run.py --dataset splice --classifier baseline --reduction fcnn --version 1 --reduction_k 3
```

The commands create `results/baseline_vowel.csv` and `results/baseline_splice.csv`.
Use `--out <path>` to select another destination.

### Functions named in the assignment

The experiments use the classifier classes, but the functions the assignment asks for exist
and give the same predictions (tested in `tests/test_required_names.py`):

```python
from experiment_runner import load_fold
from models.knn import kNNAlgorithm, reduction_kNNAlgorithm
from models.knn.distances import heom
from models.knn.voting import majority
from models.random_forest import rfAlgorithm
from models.svm import svmAlgorithm

train, test, _ = load_fold("vowel", 0)  # TrainMatrix / TestMatrix, preprocessed
kNNAlgorithm(train, test, 3, heom, majority)  # optional retention function as 6th argument
reduction_kNNAlgorithm(train, test, 3, "heom", "majority", "fcnn", version=1)
svmAlgorithm(train, test, kernel="poly", degree=5, C=3.0)
rfAlgorithm(train, test, n_estimators=300, criterion="entropy")
```

Each returns one predicted class code per test case.

### Create a model

Each classifier now inherits from the BaseClassifier class. This object-oriented design
allows each model to validate its own hyperparameters and encapsulate its execution logic.

The model must implement the fit_predict method, which receives a preprocessed train/test
pair and returns its class-code predictions along with the number of training instances it
retained. The runner calculates metrics, timing, and storage percentage, then writes the common CSV schema.

This is the interface used by kNN, SVM, Random Forest, and their reduced-data variants:

```python
from models.classifier_base import BaseClassifier
from experiment_runner import ClassificationResult, run_experiment, save_results
from preprocessing import Preprocessed


class MyModel(BaseClassifier):
    def parse_params(self):
        """Optional: Validate and cast hyperparameters from the CLI."""
        # Example: self.params is populated from the command line
        self.k = int(self.params.get("k", 3))

    def prepare_data(self, train: Preprocessed, test: Preprocessed):
        """Prepare data for fit and predict, applying model-configured instance reduction."""
        if hasattr(self, "reduction_instance") and self.reduction_instance.method != "none":
            train = train.reduce(self.reduction_instance)
        # Format features (e.g., train.to_onehot() or return Preprocessed)
        return train_X, train_y, test_X, test_y

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Core training and prediction logic."""
        # 1. Prepare data (including model-level instance reduction)
        train_X, train_y, test_X, test_y = self.prepare_data(train, test)
        # 2. Fit model and predict
        predictions = ...  # one integer class code per test row
        return ClassificationResult(predictions=predictions, storage=len(train_y))


# ---------------------------------------------------------
# Execution (handled automatically inside run.py)
# ---------------------------------------------------------
params = {"k": 5}
model_instance = MyModel(params)

results = run_experiment(
    dataset="vowel",
    classifier=model_instance.fit_predict,  # Inject the bound method
    model_name="my_model",
    params=params,
)
save_results(results, "results/my_model_vowel.csv")
```

See `docs/DECISIONS.md` for the full classifier and storage contract.

## EDA

Figures and a text summary of the preprocessed datasets (class distribution, vowel
histograms and PCA, splice nucleotide frequencies). Run from the project root:

```bash
python experiments/eda.py --dataset all --fold 0 --csv
```

Output goes to `results/eda/` (not committed). `--dataset` accepts `splice`, `vowel` or
`all`; `--csv` also saves the preprocessed fold (one-hot + `class` column); `--out`
changes the output folder.

## Grid Search v2 (final experiments, with and without reduction)

`experiments/grid_search_v2.py` runs every experiment of the assignment and the statistical tests
that answer each question. The reasoning and the list of tests are in `docs/DECISIONS.md`
("Grid search v2"). Run from the project root:

```bash
# everything: run the grids, then the tests (both datasets)
python experiments/grid_search_v2.py --dataset all --stage all

# only one dataset, or only one stage
python experiments/grid_search_v2.py --dataset splice --stage run
python experiments/grid_search_v2.py --dataset splice --stage evaluate
```

Options:

- `--dataset` `splice`, `vowel` or `all` (default `all`). Datasets are always analysed separately.
- `--stage` `run` (experiments), `evaluate` (tests and reports from the CSVs) or `all` (default).
- `--out` output folder (default `results/gridsearch_v2`).
- `--jobs N` parallel processes (default 1). Use 1 for the final times: parallel processes
  compete for the CPU and inflate them. `N > 1` is fine to get accuracies quickly.
- `--reduce-top N` how many of the best kNN configurations get every reduction technique
  (default 3; the best one is the `reduction_kNNAlgorithm`, the others check robustness).

Expected time with `--jobs 1`: about 2 hours (splice ~1.5 h, vowel ~20 min; sum of the measured
times of our run). The results of our run are committed in `results/gridsearch_v2/`, so
`--stage evaluate` and `plot_winners.py` can be run directly on them (a few minutes).

`run` has three phases: (1) full kNN / SVM / RF grids without reduction; (2) every
reduction variant (FCNN, EENTh, DROP3, DEL) on the best kNN configurations; (3) the best SVM and RF trained on the reduced set of each technique. It is
resumable: a configuration whose CSV exists is skipped, so an interrupted run can simply be
started again.

Each reduced training set is computed once and stored in `results/gridsearch_v2/reduction_cache/`
(not committed; delete it to recompute), because the best SVM and RF reuse the reductions of the
best kNN. The reported times still include the reduction time measured when it was computed. The
cache is off by default: a classifier only uses it when its `reduction_cache_dir` attribute is
set, which `grid_search_v2.py` does for every classifier it creates:

```python
model = SVM({"kernel": "rbf", "reduction": "drop", "version": 3})
model.reduction_cache_dir = Path("results/gridsearch_v2/reduction_cache")  # optional
```

Output (`results/gridsearch_v2/`):

```text
results/gridsearch_v2/
├── winners.json                 # parameters of the winners (used by plot_winners.py)
└── <dataset>/
    ├── report.md                # tables and every test, per question of the assignment
    ├── tests.csv                # one row per test (question, test, p-value, winner, post-hoc)
    ├── summary.csv              # one row per configuration (means over the 10 folds)
    ├── figures/cd_*.png         # CD diagrams of the significant Friedman tests
    ├── base/<model>/*.csv       # one CSV per configuration without reduction
    └── reduction/<model>/*.csv  # one CSV per configuration with reduction
```

### Plots of the winners (ROC, PR, confusion matrices, trade-offs)

After `--stage evaluate`:

```bash
python experiments/plot_winners.py --results results/gridsearch_v2
```

It retrains the winners in `winners.json` (best kNN, SVM and RF, without and with the best
reduction technique) on the 10 folds with per-class scores, checks that the accuracy is the
one of the grid search, caches the scores in `plots/cache/` and writes to
`results/gridsearch_v2/plots/`: `roc_pr_<dataset>.png`, `confusion_<dataset>.png` and
`tradeoff_<dataset>.png` (accuracy vs storage and vs solve time) and `reduction_<dataset>.png`
(accuracy, solve time, reduction time and storage per technique for kNN, SVM and RF). `--retrain` ignores the
cache; `--dataset` limits it to one dataset.

## Building the delivery ZIP

```bash
python scripts/make_submission.py --report path/to/report.pdf
```

Creates `dist/submission.zip` with `report.pdf` at the root and the project under `code/`
(README, requirements, `src/`, `experiments/`, `datasets/`), as the course instructions ask.
Only files tracked by git are packed. It fails if `requirements.txt` lists a package outside the
allowed ones or if any packed file imports one. Options: `--with-results` (committed results),
`--with-tests` (tests, need pytest), `--draft` (no report, for a dry run), `--out <zip>`.

## Layout

- `src/` — source code
- `tests/` — tests
- `experiments/` — experiment scripts
- `results/` — CSV and MD results
- `report/` — report sources
- `datasets/` — ARFF folds (do not modify)

---

## Development (team)

### One-time setup

After the basic setup above, every team member runs once:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

What gets installed:

| Command                               | Installs                                                | Purpose                                                                                                                               |
|---------------------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `pip install -r requirements.txt`     | numpy, scipy, pandas, scikit-learn, matplotlib, seaborn | Packages allowed by the assignment; the only ones the final code may use.                                                             |
| `pip install -r requirements-dev.txt` | ruff, pytest, pre-commit                                | Development tools only (lint/format, tests, git hooks). Not needed to run the experiments.                                            |
| `pre-commit install`                  | A git hook in `.git/hooks/`                             | Runs the checks below automatically on every `git commit`. The first commit downloads the hook environments, so it can take a minute. |

The pre-commit hooks:

- `ruff` and `ruff format`: lint and format the code.
- Block files larger than 2 MB (avoids committing results or data by accident).
- Fix missing end-of-file newlines and trailing whitespace.
- Detect leftover merge-conflict markers.
- Refuse commits made directly on `main`.

If a hook modifies files, the commit is aborted: review the changes, `git add` them
and commit again.

### Checks

Run locally before pushing (the same checks run in GitHub Actions on every pull request):

```bash
ruff check .
ruff format --check .
pytest -q
pre-commit run --all-files
```

### Workflow

1. Pick an issue and assign it to yourself: comment `request` on it (see Automation).
2. Create a branch from an up-to-date `main`:

```bash
   git switch main && git pull
   git switch -c feat/<short-name>
```

3. Commit small, focused changes. The pre-commit hooks run automatically.
4. Push and open a pull request that references the issue (`Closes #n`).
5. CI must pass before merging. Asking a teammate to take a look is welcome but not required.

`main` is protected: do not push to it directly.

### Rules of the project

- The kNN is implemented from scratch (no ML libraries in that part).
- scikit-learn is used for SVM, Random Forest and preprocessing. The kNN and the
  instance reduction are our own code.
- Never modify the dataset folds in `datasets/`.
- No data leakage: normalization ranges and any statistic are computed on the
  training set of each fold only.
- Use fixed random seeds so results are reproducible.
- Keep this README up to date when the way to install or run the project changes.

### Automation (GitHub Actions)

| What                  | When                        | Behaviour                                                                                                                                                                                                    |
|-----------------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI**                | Every pull request          | Installs dependencies with Python 3.11, runs `ruff` and `pytest`. Required to merge.                                                                                                                         |
| **PR checks**         | Every pull request          | Warns (red check, not blocking) if the description does not link an issue. Adds labels by folder (`knn`, `reduction`, `models`, `stats`, `report`, `infra`) and a size label (`size/S`, `size/M`, `size/L`). |
| **Stale PR reminder** | Weekdays                    | Comments on pull requests with no activity for 2 days. Never closes anything. Add the `wip` label or use a draft PR to be ignored.                                                                           |
| **Daily digest**      | Weekdays, 09:00 (Spain)     | Updates the "Team digest (automated)" issue with open PRs and the current milestone. Adds a comment mentioning people only when a PR is stalled or a milestone is due within 2 days.                         |
| **Self-assign**       | When commenting on an issue | Comment `request` (or `/request`, `/assign`) to assign the issue to yourself, `/unassign` to remove yourself. The comment must contain only that word.                                                       |

Milestones: M1 base (9 Oct), M2 kNN and models (16 Oct), M3 reduction and
experiments (21 Oct), M4 report (25 Oct).
