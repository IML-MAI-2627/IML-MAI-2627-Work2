# IML Work 2 — kNN vs SVM vs Random Forest

Course project for *Introduction to Machine Learning* (MAI, 2026-2027). We study
kNN (implemented from scratch) and compare it with SVM and Random Forest,
including the effect of instance reduction on accuracy, efficiency and storage.

Datasets: `splice` (3190 cases, nominal) and `vowel` (990 cases, numeric + nominal),
both with predefined 10-fold train/test splits in `datasets/`.

## Team

- Pol Marsol Torras
- David Fernández Gómez
- Ran Fan
- Mariam Shehadeh

## Environment

- Python: 3.11 (tested with 3.11.16)
- Tested operating system: MacOS Sequoia 15.6.1 (24G90)
- Packages: only those in `requirements.txt` (pinned versions): numpy, scipy, pandas,
  scikit-learn, matplotlib, seaborn.
- Hardware: a normal CPU is enough (no GPU). The full grid search takes about 2 hours on one
  process; every other command takes seconds to a few minutes.

## Installation

Open a terminal in the code folder (the project root, `code/` in the delivery ZIP):

```bash
python3.11 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the commands below from the same folder with the environment active, and `deactivate` at
the end.

## Project structure

The delivery archive (`submission.zip`) contains the report PDF and the `code/` folder:

```text
submission.zip
├── report.pdf                   # Complete project report (<= 20 pages)
└── code/                        # Executable project root
    ├── README.md                # Operational reproduction guide for the instructor
    ├── requirements.txt         # Pinned production dependencies (allowed libraries only)
    ├── datasets/                # Predefined 10-fold train/test splits (Splice and Vowel)
    ├── src/                     # Core library code
    │   ├── config.py            # Global paths, dataset definitions, seed (42)
    │   ├── arff_parser.py       # Reads ARFF folds into train/test matrices
    │   ├── preprocessing.py     # Data scaling (min-max [0,1]), encoding, imputation (train fold only)
    │   ├── experiment_runner.py # Evaluates a model on 10 folds, tracking accuracy, times, storage
    │   ├── stats.py             # Friedman, Nemenyi, Bonferroni-Dunn, Wilcoxon, Holm tests
    │   └── models/              # Model implementations
    │       ├── knn/             # From-scratch pure NumPy kNN and instance reduction
    │       │   ├── knn.py       # Main kNN classifier interface
    │       │   ├── distances.py # Euclidean, Manhattan, Canberra, Clark, HEOM
    │       │   ├── voting.py    # Majority and inverse-distance weighted voting
    │       │   ├── retention.py # Never, Always, Different Class, Oblivion by Goodness
    │       │   └── reduction.py # FCNN (v1/v2), ENNTH (mu=0, 0.5, 0.7), DROP3, DEL
    │       ├── svm.py           # Scikit-learn Support Vector Machine adapter
    │       ├── random_forest.py # Scikit-learn Random Forest adapter
    │       └── majority_baseline.py # Majority baseline classifier
    ├── experiments/             # Executable entry points for reproduction
    │   ├── run.py               # Runs any single configuration across 10 folds
    │   ├── eda.py               # Generates Section 2 dataset analysis & results/eda/report.md
    │   ├── grid_search.py       # Runs the complete grids & Demšar statistical evaluations
    │   ├── plot_winners.py      # Generates evaluation plots and CD diagrams
    │   ├── vowel_columns.py     # Analysis on Vowel column variations
    │   └── selection_bias.py    # Leave-one-fold-out estimation of selection bias
    └── results/                 # Evaluation outputs (CSV metrics, test statistics, plots)
```

## Execution order

1. `python experiments/eda.py --dataset all --fold 0` — generates `results/eda/report.md`
   with all Section 2 statistics and EDA figures. Independent (reads raw `datasets/`).
2. `python experiments/grid_search.py --dataset all --stage run` — the experiments
   (about 2 h); writes one CSV per configuration in `results/gridsearch/`.
3. `python experiments/grid_search.py --dataset all --stage evaluate` — tests and reports;
   **needs the CSVs of step 2** and writes `results/gridsearch/winners.json`.
4. `python experiments/plot_winners.py --results results/gridsearch` — **needs
   `winners.json`** (step 3).
5. `python experiments/vowel_columns.py` — **needs `winners.json`** (step 3).
6. `python experiments/selection_bias.py` — **needs the CSVs of step 2**.

The results of steps 2-3 are committed, so steps 3-6 can be run directly without repeating
step 2. `experiments/run.py` runs one configuration on its own and depends on nothing.

## Entry points

| File | Purpose | Command (from the project root) |
|---|---|---|
| `experiments/run.py` | One classifier and parameter set on the 10 folds | `python experiments/run.py --dataset vowel --classifier knn --k 3 --distance heom` |
| `experiments/eda.py` | Exploratory figures and statistical summary (`results/eda/report.md`) | `python experiments/eda.py --dataset all --fold 0` |
| `experiments/grid_search.py` | All experiments and statistical tests | `python experiments/grid_search.py --dataset all --stage all` |
| `experiments/plot_winners.py` | ROC/PR, confusion, trade-off & timing plots, and Table 7 timing report | `python experiments/plot_winners.py --results results/gridsearch` |
| `experiments/vowel_columns.py` | Which vowel columns to keep (A/B/C comparison) | `python experiments/vowel_columns.py` |
| `experiments/selection_bias.py` | Leave-one-fold-out estimate of the selection bias | `python experiments/selection_bias.py` |

Each one is described below with all its options and outputs.

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
  - `ennth`: Edited Nearest Neighbor with Threshold (ENNTH; Vázquez & Sánchez, 2005). Prunes noisy and ambiguous borderline instances.
  - `drop`: Decremental Reduction Optimization Procedure (Wilson & Martínez, 2000). Decrementally prunes internal and noisy prototypes.
  - `del`: Decremental Encoding Length (Cameron-Jones, 1995; Wilson & Martinez, 2000 Sections 3.2.8 & 4.6). Minimum Description Length pruning using numerically stable log-space combinatorial sums.
  Default: `none`.

- `--reduction_k` (positive integer): Number of nearest neighbors used by instance reduction when executing non-kNN models (`svm`, `rf`, `baseline`). Default: 3. *(Forbidden for `knn`, which uses `--k`)*.

- `--version` (integer): Algorithm variant/rule. Exclusive to `--reduction fcnn` and `--reduction drop`:
  - For `fcnn`: `1` (each prototype adds its nearest enemy), `2` (each prototype adds the medoid of its enemies), `3` / `4` (as 1 / 2 but one instance per iteration). Default: 1.
  - For `drop`: `1` (DROP1), `2` (DROP2), `3` (DROP3 with ENN filtering), `4` (DROP4), `5` (DROP5). Default: 3.
  *(Forbidden with `--reduction ennth`, `--reduction del`, or `--reduction none`)*.

- `--threshold` (float): Acceptance consensus threshold $\in [0, 1]$. Exclusive to `--reduction ennth`:
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

# kNN with ENNTH filtering at consensus threshold 0.6
python experiments/run.py --dataset vowel --classifier knn --k 5 --reduction ennth --threshold 0.6

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

## EDA (Exploratory Data Analysis)

Extracts all dataset properties, distributions, and domain statistics cited in Section 2 of the report, and generates exploratory figures. EDA operates directly on the raw benchmark datasets without dependencies on downstream models or grid search runs. Run from the project root:

```bash
python experiments/eda.py --dataset all --fold 0
```

Outputs:
- **`results/eda/report.md`**: Text summary containing all exploratory statistics quoted in Section 2:
  - Class frequencies and majority baseline.
  - Ambiguity codes (IUPAC) in Splice.
  - Position-wise mutual information and canonical splice motifs (GT at 31–32, AG at 29–30).
  - Sequence duplicates and exact copies across folds (9.2% on Splice).
  - Speaker breakdown and nearest-neighbor speaker overlap (98.2%) on Vowel.
  - Min-max scaling range clipping statistics.
  - Baseline 1-NN accuracy.
- **Figures in `results/eda/`**:
  - `splice_fold0_classes.png`: Class distributions for Splice.
  - `splice_fold0_positions.png`: Nucleotide frequency per position and class (Figure 1 in report).
  - `vowel_fold0_classes.png`: Class distributions for Vowel.
  - `vowel_fold0_hist.png`: Feature distributions per acoustic attribute.
  - `vowel_fold0_pca.png`: 2D PCA projection of the vowel space.
- **Options**:
  - `--dataset`: `splice`, `vowel`, or `all` (default `all`).
  - `--fold`: Index of the fold to analyze (default `0`).
  - `--csv`: Also saves the preprocessed fold as CSV (one-hot encoded).
  - `--out`: Changes the output directory (default `results/eda`).

## Grid Search (final experiments, with and without reduction)

`experiments/grid_search.py` runs every experiment of the assignment and the statistical tests
that answer each question. The reasoning and the list of tests are in `docs/DECISIONS.md`
("Grid search"). Run from the project root:

```bash
# everything: run the grids, then the tests (both datasets)
python experiments/grid_search.py --dataset all --stage all

# only one dataset, or only one stage
python experiments/grid_search.py --dataset splice --stage run
python experiments/grid_search.py --dataset splice --stage evaluate
```

Options:

- `--dataset` `splice`, `vowel` or `all` (default `all`). Datasets are always analysed separately.
- `--stage` `run` (experiments), `evaluate` (tests and reports from the CSVs) or `all` (default).
- `--out` output folder (default `results/gridsearch`).
- `--jobs N` parallel processes (default 1). Use 1 for the final times: parallel processes
  compete for the CPU and inflate them. `N > 1` is fine to get accuracies quickly.
- `--reduce-top N` how many of the best kNN configurations get every reduction technique
  (default 3; the best one is the `reduction_kNNAlgorithm`, the others check robustness).

Expected time with `--jobs 1`: about 2 hours (splice ~1.5 h, vowel ~20 min; sum of the measured
times of our run). The results of our run are committed in `results/gridsearch/`, so
`--stage evaluate` and `plot_winners.py` can be run directly on them (a few minutes).

`run` has three phases: (1) full kNN / SVM / RF grids without reduction; (2) every
reduction variant (FCNN, ENNTH, DROP3, DEL) on the best kNN configurations; (3) the best SVM and RF trained on the reduced set of each technique. It is
resumable: a configuration whose CSV exists is skipped, so an interrupted run can simply be
started again.

Each reduced training set is computed once and stored in `results/gridsearch/reduction_cache/`
(not committed; delete it to recompute), because the best SVM and RF reuse the reductions of the
best kNN. The reported times still include the reduction time measured when it was computed. The
cache is off by default: a classifier only uses it when its `reduction_cache_dir` attribute is
set, which `grid_search.py` does for every classifier it creates:

```python
model = SVM({"kernel": "rbf", "reduction": "drop", "version": 3})
model.reduction_cache_dir = Path("results/gridsearch/reduction_cache")  # optional
```

Output (`results/gridsearch/`):

```text
results/gridsearch/
├── winners.json                 # parameters of the winners (used by plot_winners.py)
└── <dataset>/
    ├── report.md                # tables and every test, per question of the assignment
    ├── tests.csv                # one row per test (question, test, p-value, winner, post-hoc)
    ├── summary.csv              # one row per configuration (means over the 10 folds)
    ├── figures/cd_*.png         # CD diagrams of the significant Friedman tests
    ├── base/<model>/*.csv       # one CSV per configuration without reduction
    └── reduction/<model>/*.csv  # one CSV per configuration with reduction
```

### Plots of the winners (ROC, PR, confusion matrices, trade-offs, timings)

After `--stage evaluate`:

```bash
python experiments/plot_winners.py --results results/gridsearch
```

It retrains the winners in `winners.json` (best kNN, SVM and RF, without and with the best
reduction technique) on the 10 folds with per-class scores, checks that the accuracy matches
the grid search, caches the scores in `plots/cache/`, and outputs:

1. **Figures in `results/gridsearch/plots/`**:
   - `roc_pr_<dataset>.png`: ROC and Precision-Recall curves.
   - `confusion_<dataset>.png`: Multiclass confusion matrices.
   - `tradeoff_<dataset>.png`: Accuracy vs storage and accuracy vs solve time.
   - `reduction_<dataset>.png`: Accuracy, solve time, reduction time, and storage per reduction technique across models.
   - `timing_<dataset>.png`: Fit and predict wall-clock times per sample across models.

2. **Timing & Sanity Report (`results/gridsearch/timing_report.md`)**:
   - Fit and predict time per instance for the winners (Table 7 in the report).
   - Mean ranks across 10 folds for kNN, SVM, and RF.
   - Pairwise distance disagreement checks on Splice (e.g., verifying differences between Manhattan, Clark, and Canberra on one-hot features).

Options:
- `--results`: Path to grid search directory containing `winners.json` (default `results/gridsearch`).
- `--out`: Directory where plots are saved (default `results/gridsearch/plots`).
- `--report`: Path where the timing report is written (default `results/gridsearch/timing_report.md`).
- `--retrain`: Ignores cached scores and re-evaluates all winning configurations.
- `--dataset`: Limits plotting and timing analysis to one dataset (`splice` or `vowel`).

## vowel columns

```bash
python experiments/vowel_columns.py
```

Runs the best vowel kNN of the grid search (read from `winners.json`) with three column sets: (A) the 10
numeric features, (B) A + `Sex` (the current preprocessing) and (C) B + `Speaker_Number`, and
compares them with Friedman and pairwise Wilcoxon (Holm). Writes `vowel_columns.md` (tables, tests
and how often the nearest training case is the same speaker) and `vowel_columns_accuracy.csv`
(per fold) to `results/vowel_columns/`. Options (both flags, optional): `--results` grid folder
with `winners.json` (default `results/gridsearch`), `--out` output folder (default
`results/vowel_columns`). About 3 s.

## Selection bias

```bash
python experiments/selection_bias.py
```

Leave-one-fold-out estimate of the bias of selecting and testing on the same 10 folds: for each
fold, the best configuration on the other 9 (same rule as the grid) is scored on it. One row per
dataset and model (kNN, SVM, RF, without reduction). Writes `selection_bias.md` and
`selection_bias.csv` to `results/selection_bias/`. Options (both flags, optional): `--results`
grid folder (default `results/gridsearch`), `--out` output folder (default
`results/selection_bias`). About 3 s.

## Implementation Guarantees and Rules

- **From-Scratch kNN & Instance Reduction:** Our kNN classifier, all 5 distance metrics (`euclidean`, `manhattan`, `canberra`, `clark`, `heom`), voting rules (`majority`, `inverse_distance`), retention policies (`never`, `always`, `different_class`, `oblivion_by_goodness`), and all four instance reduction algorithms (`fcnn` versions 1 & 2, `ennth`, `drop3`, and `del`) are implemented entirely from scratch in pure NumPy. No third-party ML libraries are used in `src/models/knn/`.
- **Allowed Scikit-Learn Use:** As authorized by the assignment specification and course instructions, `scikit-learn` is used strictly where permitted: for the eager baseline models (`src/models/svm.py`, `src/models/random_forest.py`) and preprocessing transformations (`src/preprocessing.py`).
- **No Data Leakage:** Preprocessing transformations (min-max normalization, category encodings) and instance reduction selections are strictly fitted on the training split of each fold alone. Test instances are never used to compute normalization bounds or select prototypes.
- **Deterministic Reproducibility:** Fixed random seeds (`seed = 42`) and deterministic tie-breaking (`np.argsort(kind="stable")`) are enforced throughout the pipeline.
- **Unmodified Folds:** The original 10-fold stratified cross-validation folds in `datasets/` are read as provided and never altered.

## Packaging Utility (Optional)

The submission archive `submission.zip` can be inspected or rebuilt using:

```bash
python scripts/make_submission.py --report path/to/report.pdf --with-results --with-tests
```

This utility validates that `requirements.txt` contains only course-approved dependencies and verifies that `src/models/knn/` does not import any external machine learning packages.
