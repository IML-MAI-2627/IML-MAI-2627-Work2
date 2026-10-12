import itertools
import subprocess
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# 1. Configuration & Hyperparameter Grids
# ---------------------------------------------------------
GRIDS = {
    "rf": {
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [10, 20, "None"],
        "max_features": ["sqrt", "None"],
        "min_samples_leaf": [1, 2],
    },
    "svm": {
        "C": [0.1, 1.0, 3.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "class_weight": ["None", "balanced"],
        "coef0": [-0.5, 0.0, 0.5],
        # We Leave tol and max_iter fixed to avoid combinatorial explosion
        "tol": [0.001],
        "max_iter": [-1],
        "degree": [2, 3, 5],
    },
}

DATASETS = ["vowel", "splice"]
MODELS = ["rf", "svm"]

OUT_DIR = Path("results/gridsearch")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def generate_combinations(grid_dict):
    """Generates all possible combinations (Cartesian product) for a given grid."""
    keys = grid_dict.keys()
    values = grid_dict.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance, strict=False))


def prune_params(model, params):
    """
    Simulates the internal pruning of the models so we can detect duplicates
    before running them in the CLI.
    """
    p = params.copy()
    if model == "svm":
        kernel = p.get("kernel")
        if kernel == "linear":
            p.pop("gamma", None)
        if kernel != "poly":
            p.pop("degree", None)
        if kernel not in ["poly", "sigmoid"]:
            p.pop("coef0", None)
    return p


# ---------------------------------------------------------
# 3. Execution Phase
# ---------------------------------------------------------
def run_grid_search():
    print("--- Starting Manual Grid Search CV ---")
    print(f"CSV results will be saved in: {OUT_DIR}\n")

    total_runs = 0

    for dataset in DATASETS:
        for model in MODELS:
            grid = GRIDS[model]
            combinations = list(generate_combinations(grid))

            # This set will keep track of configurations we have already scheduled
            tested_configs = set()
            skipped_count = 0

            print(
                f"[{dataset.upper()} - {model.upper()}] "
                f"Found {len(combinations)} raw combinations to evaluate."
            )

            for i, params in enumerate(combinations):
                # 1. Prune the inactive parameters
                pruned_params = prune_params(model, params)

                # 2. Check if we already tested this exact mathematical model
                # We use frozenset because normal dicts cannot be added to a set
                config_key = frozenset(pruned_params.items())
                if config_key in tested_configs:
                    skipped_count += 1
                    continue

                tested_configs.add(config_key)

                # Unique output file based on the loop index
                out_file = OUT_DIR / f"{dataset}_{model}_comb_{i}.csv"

                if out_file.exists():
                    continue

                # Build the CLI command using the PRUNED params
                cmd = [
                    "python",
                    "experiments/run.py",
                    "--dataset",
                    dataset,
                    "--classifier",
                    model,
                    "--out",
                    str(out_file),
                ]

                for param_name, param_value in pruned_params.items():
                    cmd.extend([f"--{param_name}", str(param_value)])

                if model == "rf":
                    cmd.extend(["--n_jobs", "-1"])

                print(f"  -> Running valid config: {pruned_params}")

                # Execute the command
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                total_runs += 1

            print(f"  * Skipped {skipped_count} redundant combinations due to kernel pruning.\n")

    print(f"\n--- Grid Search Execution Completed ({total_runs} distinct combinations run) ---\n")


# ---------------------------------------------------------
# 4. Evaluation Phase
# ---------------------------------------------------------
def evaluate_results():
    print("--- Evaluating Grid Search Results ---")

    csv_files = list(OUT_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the output directory.")
        return

    df_list = []
    columns = [
        "dataset",
        "fold",
        "model",
        "params",
        "accuracy",
        "n_correct",
        "n_incorrect",
        "time",
        "storage",
    ]

    for file in csv_files:
        try:
            df = pd.read_csv(
                file, names=columns, header=0 if "dataset" in open(file).readline() else None
            )
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        print("No valid data could be extracted from the CSV files.")
        return

    full_df = pd.concat(df_list, ignore_index=True)

    aggregated = (
        full_df.groupby(["dataset", "model", "params"])
        .agg(mean_accuracy=("accuracy", "mean"), mean_time=("time", "mean"))
        .reset_index()
    )

    report_file = OUT_DIR / "optimal_results.txt"

    print("\nOPTIMAL HYPERPARAMETERS FOUND\n")

    with open(report_file, "w") as f:
        f.write("OPTIMAL HYPERPARAMETERS FOUND\n")
        f.write("=======================================\n\n")

        for dataset in full_df["dataset"].unique():
            for model in full_df["model"].unique():
                subset = aggregated[
                    (aggregated["dataset"] == dataset) & (aggregated["model"] == model)
                ]
                if subset.empty:
                    continue

                subset = subset.sort_values(by="mean_accuracy", ascending=False)
                best_row = subset.iloc[0]

                result_block = (
                    f"Dataset: {dataset.upper()} | Model: {model.upper()}\n"
                    f"  Best Accuracy:  {best_row['mean_accuracy']:.4f}\n"
                    f"  Mean Time/Fold: {best_row['mean_time']:.4f} seconds\n"
                    f"  Optimal Params: {best_row['params']}\n"
                    f"{'-' * 60}\n"
                )

                print(result_block, end="")
                f.write(result_block)

    print(f"\nSummary report successfully saved to: {report_file}")


if __name__ == "__main__":
    run_grid_search()
    evaluate_results()
