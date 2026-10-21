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
dataset,fold,model,params,accuracy,n_correct,n_incorrect,time,storage
```

For reduction experiments, storage is the percentage of the original
training instances retained by the classifier. A classifier using the complete
training set therefore has a storage of 100%.

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
      Supports single-parameter calls exploiting symmetry via `np.triu_indices` (cutting operations and peak memory by 50%).
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

    - `--reduction` (string): Training set instance reduction technique, selected from `REDUCTION_TECHNIQUES` in `src/models/knn/reduction.py`.
      Must be one of:
      - `none`: Use the full training set without reduction.
      - `fcnn`: Fast Condensed Nearest Neighbor (Angiulli, 2007). Guaranteed training-set consistent.
      - `eenth`: Extended Edited Nearest Neighbor with Threshold (Wilson & Martinez, 2000). Removes noisy/border instances.
      - `drop`: Decremental Reduction Optimization Procedure (Wilson & Martinez, 2000). Prunes internal and noisy prototypes.
      Default: `none`.

    - `--version` (integer): Version rule/variant. Exclusive to `--reduction fcnn` and `--reduction drop`:
      - For `fcnn`: `1` (Centroid rule), `2` (Nearest enemy rule), `3` (Furthest enemy rule), `4` (Random enemy rule). Default: 1.
      - For `drop`: `1` (DROP1), `2` (DROP2), `3` (DROP3 with ENN filtering), `4` (DROP4), `5` (DROP5). Default: 3.
      *(Incompatible with `--reduction eenth` or `--reduction none`)*.

    - `--threshold` (float): Pruning acceptance threshold $\in [0, 1]$. Exclusive to `--reduction eenth`:
      - Instances are pruned if the consensus among their $k$ nearest neighbors is below this threshold. Default: 0.0.
      *(Incompatible with `--reduction fcnn`, `--reduction drop`, or `--reduction none`)*.

    Examples:
    ```bash
    # Standard batch kNN with Canberra distance and inverse distance voting
    python experiments/run.py --dataset vowel --classifier knn --k 5 --distance canberra --voting inverse_distance

    # kNN with DROP3 reduction (retains boundary instances, cuts storage)
    python experiments/run.py --dataset vowel --classifier knn --k 3 --reduction drop --version 3

    # kNN with FCNN rule 2 (nearest enemy) and dynamic Oblivion by Goodness retention
    python experiments/run.py --dataset splice --classifier knn --k 3 --reduction fcnn --version 2 --retention oblivion_by_goodness

    # kNN with EENTh filtering at threshold 0.6
    python experiments/run.py --dataset vowel --classifier knn --k 5 --reduction eenth --threshold 0.6
    ```

The commands create `results/baseline_vowel.csv` and `results/baseline_splice.csv`.
Use `--out <path>` to select another destination.

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

    def prepare_data(self):
        """We prepare the data for the model fit and predict."""
        return train_X, train_y, test_X, test_y

    def fit_predict(self, train: Preprocessed, test: Preprocessed) -> ClassificationResult:
        """Core training and prediction logic."""
        # 1. Train the model using the prepared data from self.prepare_data()
        # 2. Predict the test instances
        predictions = ...  # one integer class code per test row
        return ClassificationResult(predictions=predictions, storage=len(train.y))


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

## Hyperparameter Tuning (Grid Search)

Extracts the approximated optimal hyperparameters for each model and each dataset performing a Grid Search and
statistical tests.
Run from the project root:

```bash
python experiments/grid_search.py --dataset all --classifier all --out results/gridsearch
```

Output goes to `results/gridsearch/` (not committed). `--dataset` accepts `splice`, `vowel` or `all`; `--classifier`
accepts `rf`, `svm`, or `all`; `--out` changes the output folder.

The output is organized as a nested directory structure to keep thousands of configurations perfectly categorized:

```text
results/gridsearch/
├── optimal_results.txt                     # Master summary report with stats and best configs
├── splice/                                 # Results specific to the Splice dataset
│   ├── rf/                                 # Random Forest configurations
│   │   ├── cd_diagram_splice_rf_champions.png  # Intra-model CD diagram
│   │   ├── splice_rf_criterion-gini...csv      # Fold-by-fold results for a specific config
│   │   └── ... (96 CSV files)
│   └── svm/                                # Support Vector Machine configurations
│       ├── cd_diagram_splice_svm_champions.png
│       ├── splice_svm_C-1.0_kernel-rbf...csv
│       └── ... (168 CSV files)
└── vowel/                                  # Results specific to the Vowel dataset
    ├── rf/
    └── svm/
```

### Generated Artifacts

* **`optimal_results.txt`**: The definitive human-readable report. It details the absolute best configuration for each
  model and dataset, lists the statistically equivalent top models (Nemenyi post-hoc test), and displays the results of
  the Head-to-Head Wilcoxon signed-ranks test between the absolute best RF and the absolute best SVM.
* **`cd_diagram_*.png`**: Critical Difference diagrams following Demšar (2006). These visually group statistically tied
  configurations within a model family using a thick black line, separating the true winners from the rest.
* **`*.csv` Files**: Each evaluated configuration saves its 10-fold results to its own CSV file. This enables aggressive
  checkpointing: if the script is interrupted, it will instantly resume where it left off without re-running completed
  configurations.

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
