import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when the script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
        # We leave tol and max_iter fixed to avoid combinatorial explosion
        "tol": [0.001],
        "max_iter": [-1],
        "degree": [2, 3, 5],
    },
}


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
def run_grid_search(datasets: list[str], models: list[str], out_dir: Path):
    print("--- Starting Manual Grid Search CV ---")
    print(f"CSV results will be saved in: {out_dir}\n")

    total_runs = 0

    for dataset in datasets:
        for model in models:
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
                pruned_params = prune_params(model, params)

                # Check if we already tested this exact mathematical model
                config_key = frozenset(pruned_params.items())
                if config_key in tested_configs:
                    skipped_count += 1
                    continue

                tested_configs.add(config_key)

                # Unique output file based on the loop index
                out_file = out_dir / f"{dataset}_{model}_comb_{i}.csv"

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
def evaluate_results(out_dir: Path):
    print("--- Evaluating Grid Search Results ---")

    csv_files = list(out_dir.glob("*.csv"))
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

    report_file = out_dir / "optimal_results.txt"

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manual Grid Search CV for hyperparameter tuning.")
    parser.add_argument("--dataset", choices=["vowel", "splice", "all"], default="all")
    parser.add_argument("--classifier", choices=["rf", "svm", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path("results/gridsearch"))

    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    datasets = ["vowel", "splice"] if args.dataset == "all" else [args.dataset]
    models = ["rf", "svm"] if args.classifier == "all" else [args.classifier]

    # 1. Run all combinations
    run_grid_search(datasets, models, args.out)

    # 2. Read all files, find the best, and save to optimal_results.txt
    evaluate_results(args.out)


if __name__ == "__main__":
    main()
