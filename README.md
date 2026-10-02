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

## Running the experiments
_TODO: placeholder. Commands will be added once the experiment runner exists._

## Layout
- `src/` — source code
- `tests/` — tests
- `experiments/` — experiment scripts
- `results/` — CSV results
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

| Command | Installs | Purpose |
|---|---|---|
| `pip install -r requirements.txt` | numpy, scipy, pandas, scikit-learn, matplotlib, seaborn | Packages allowed by the assignment; the only ones the final code may use. |
| `pip install -r requirements-dev.txt` | ruff, pytest, pre-commit | Development tools only (lint/format, tests, git hooks). Not needed to run the experiments. |
| `pre-commit install` | A git hook in `.git/hooks/` | Runs the checks below automatically on every `git commit`. The first commit downloads the hook environments, so it can take a minute. |

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
- scikit-learn is used only for SVM and Random Forest.
- Never modify the dataset folds in `datasets/`.
- No data leakage: normalization ranges and any statistic are computed on the
  training set of each fold only.
- Use fixed random seeds so results are reproducible.
- Keep this README up to date when the way to install or run the project changes.

### Automation (GitHub Actions)
| What | When | Behaviour |
|---|---|---|
| **CI** | Every pull request | Installs dependencies with Python 3.11, runs `ruff` and `pytest`. Required to merge. |
| **PR checks** | Every pull request | Warns (red check, not blocking) if the description does not link an issue. Adds labels by folder (`knn`, `reduction`, `models`, `stats`, `report`, `infra`) and a size label (`size/S`, `size/M`, `size/L`). |
| **Stale PR reminder** | Weekdays | Comments on pull requests with no activity for 2 days. Never closes anything. Add the `wip` label or use a draft PR to be ignored. |
| **Daily digest** | Weekdays, 09:00 (Spain) | Updates the "Team digest (automated)" issue with open PRs and the current milestone. Adds a comment mentioning people only when a PR is stalled or a milestone is due within 2 days. |
| **Self-assign** | When commenting on an issue | Comment `request` (or `/request`, `/assign`) to assign the issue to yourself, `/unassign` to remove yourself. The comment must contain only that word. |

Milestones: M1 base (9 Oct), M2 kNN and models (16 Oct), M3 reduction and
experiments (21 Oct), M4 report (25 Oct).
