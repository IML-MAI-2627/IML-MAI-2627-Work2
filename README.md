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
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments
_TODO: placeholder. Commands will be added once the experiment runner exists._

## Layout
- `src/` — source code
- `tests/` — tests
- `experiments/` — experiment scripts
- `results/` — CSV results
- `report/` — report sources
- `datasets/` — ARFF folds (do not modify)
- `docs/` — assignment and planning documents
