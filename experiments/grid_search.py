import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when the script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import stats  # noqa: E402

# ---------------------------------------------------------
# 1. Configuration & Hyperparameter Grids
# ---------------------------------------------------------
GRIDS = {
    "rf": {
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["gini", "entropy"],
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
    "knn": {
        "k": [3, 5, 7],
        "distance": ["euclidean", "manhattan", "canberra", "clark", "heom"],
        "voting": ["majority", "inverse_distance"],
        "retention": ["never", "always", "different_class", "oblivion_by_goodness"],
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
    elif model == "knn":
        reduction = p.get("reduction", "none")
        if reduction == "none":
            p.pop("version", None)
            p.pop("threshold", None)
        elif reduction in ("fcnn", "drop"):
            p.pop("threshold", None)
        elif reduction == "eenth":
            p.pop("version", None)
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

            for _i, params in enumerate(combinations):
                pruned_params = prune_params(model, params)

                # Check if we already tested this exact mathematical model
                config_key = frozenset(pruned_params.items())
                if config_key in tested_configs:
                    skipped_count += 1
                    continue

                tested_configs.add(config_key)

                # Unique output file based on the parameters_used

                # 1. Joining parameter-value with "-", and each pair with "_"
                if pruned_params:
                    params_str = "_".join(f"{k}-{v}" for k, v in pruned_params.items())
                else:
                    params_str = "default"  # If empty dictionary

                # 2. Final path
                out_file = out_dir / dataset / model / f"{dataset}_{model}_{params_str}.csv"

                if out_file.exists():
                    continue
                print(f"  path: {out_file}")

                # Build the CLI command using the PRUNED params
                cmd = [
                    sys.executable,
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

                # Execute the command, capturing stderr to surface fail-fast exceptions
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
                )

                if result.returncode != 0:
                    print(f"\n[!] ERROR: Run failed for config: {pruned_params}")
                    print(f"Command executed: {' '.join(cmd)}")
                    print(f"Error Details:\n{result.stderr}")

                    # Halt the entire grid search on failure to prevent wasting time on a bad grid
                    raise RuntimeError(
                        "Experiment aborted due to subprocess failure. See stderr above."
                    )
                total_runs += 1

            print(f"  * Skipped {skipped_count} redundant combinations due to kernel pruning.\n")

    print(f"\n--- Grid Search Execution Completed ({total_runs} distinct combinations run) ---\n")


# ---------------------------------------------------------
# 4. Evaluation Phase (Statistical Analysis)
# ---------------------------------------------------------

# Keys of the model need to be the same as inside the model columns in the csv
TOP_COMBINATIONS = {
    "rf": {
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "None"],
    },
    "svm": {
        "kernel": ["linear", "rbf", "poly"],
        "class_weight": ["None", "balanced"],
    },
    "knn": {
        "retention": ["never", "always", "different_class", "oblivion_by_goodness"],
        "voting": ["majority", "inverse_distance"],
    },
}


def parse_params_str(p_str: str) -> dict:
    """Safely converts a parameter string like 'C=1.0;kernel=linear' back into a dictionary."""
    if pd.isna(p_str) or not p_str:
        return {}
    try:
        return dict(kv.split("=", 1) for kv in str(p_str).split(";"))
    except ValueError:
        return {}


# ---------------------------------------------------------
# 4. Evaluation Phase (Statistical Analysis)
# ---------------------------------------------------------
def evaluate_results(out_dir: Path):
    print("--- Evaluating Grid Search Results with Statistical Tests ---")

    csv_files = list(out_dir.rglob("*.csv"))

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

    report_file = out_dir / "optimal_results.txt"
    print("\nGENERATING OPTIMAL HYPERPARAMETERS AND CD DIAGRAMS\n")

    with open(report_file, "w") as f:
        f.write("OPTIMAL HYPERPARAMETERS & STATISTICAL ANALYSIS\n")
        f.write("==============================================\n\n")

        # Evaluate per dataset
        for dataset in full_df["dataset"].unique():
            # --- NEW: Collector for the Head-to-Head Cross-Model Wilcoxon ---
            best_mean_configs_data = {}
            best_rank_configs_data = {}

            # Evaluate per model
            for model in full_df["model"].unique():
                subset = full_df[
                    (full_df["dataset"] == dataset) & (full_df["model"] == model)
                ].copy()

                if subset.empty:
                    continue

                print(f"  -> Analyzing {model.upper()} on {dataset.upper()}...")

                # Map long model names ("Support Vector Machine") back to short names ("svm")
                short_model = (
                    "rf"
                    if "random forest" in model.lower()
                    else "svm"
                    if "support vector" in model.lower()
                    else model.lower()
                )

                subset["config"] = [
                    stats.config_name(m, p)
                    for m, p in zip(subset["model"], subset["params"], strict=False)
                ]

                try:
                    table = stats.score_table(subset, dataset=dataset, metric="accuracy")
                except ValueError as e:
                    msg = f"Skipping {model.upper()} on {dataset.upper()}: {e}\n"
                    print(f"    [!] {msg}")
                    f.write(msg + "-" * 80 + "\n\n")
                    continue

                # 1. Calculate Absolute Mean Accuracies for all configs
                mean_accs = table.mean().sort_values(ascending=False)
                mean_times = subset.groupby("config")["time"].mean()

                best_absolute_config = mean_accs.index[0]
                best_acc = mean_accs.iloc[0]
                best_time = mean_times[best_absolute_config]

                # Save the best configuration's fold data for the Cross-Model Wilcoxon later
                best_mean_configs_data[best_absolute_config] = table[best_absolute_config]

                # 2. Extract the Champions based on TOP_COMBINATIONS
                champion_configs = []

                if short_model in TOP_COMBINATIONS:
                    grid = TOP_COMBINATIONS[short_model]
                    keys = grid.keys()

                    for instance in itertools.product(*grid.values()):
                        target_comb = dict(zip(keys, instance, strict=False))

                        matching_configs = []
                        for conf in mean_accs.index:
                            p_str = subset.loc[subset["config"] == conf, "params"].iloc[0]
                            p_dict = parse_params_str(p_str)

                            if all(p_dict.get(k) == str(v) for k, v in target_comb.items()):
                                matching_configs.append(conf)

                        if matching_configs:
                            champion_configs.append(matching_configs[0])
                else:
                    champion_configs = mean_accs.index[:5].tolist()

                champion_configs = list(dict.fromkeys(champion_configs))
                filtered_table = table[champion_configs]
                n_blocks, k = filtered_table.shape

                # Write Summary Block
                result_block = (
                    f"Dataset: {dataset.upper()} | Model: {model.upper()}\n"
                    f"  Best Mean Accuracy Config: {best_absolute_config}\n"
                    f"  Mean Accuracy:        {best_acc:.4f}\n"
                    f"  Mean Time/Fold:       {best_time:.4f} seconds\n"
                    f"  Total Configs Tested: {table.shape[1]}\n"
                    f"  Champions Extracted:  {k} (Used for Statistical Testing)\n"
                )
                f.write(result_block)

                # 3. Statistical Testing on the Champions Only (Intra-Model)
                if k >= 3:
                    try:
                        friedman = stats.friedman_test(filtered_table)
                        f.write(f"  Friedman p-value:     {friedman.iman_davenport_p_value:.4e}\n")

                        if friedman.iman_davenport_p_value < 0.05:
                            f.write(
                                "  -> Significant differences found "
                                "among champions. Running Nemenyi...\n"
                            )
                            posthoc = stats.nemenyi_test(filtered_table)

                            # USE SHORT_MODEL FOR CONSISTENT FOLDERS AND FILENAMES
                            model_dir = out_dir / dataset / short_model
                            model_dir.mkdir(parents=True, exist_ok=True)

                            diagram_path = (
                                model_dir / f"cd_diagram_{dataset}_{short_model}_champions.png"
                            )
                            stats.plot_cd_diagram(
                                posthoc,
                                title=f"Nemenyi CD (Champions) - {model.upper()} "
                                f"on {dataset.upper()}",
                                path=diagram_path,
                            )
                            f.write(f"  -> CD Diagram saved to: {diagram_path.name}\n")

                            groups = stats.nemenyi_groups(posthoc)
                            best_rank_config = posthoc.mean_ranks.index[0]
                            top_group = next(g for g in groups if best_rank_config in g)
                            best_rank_config = posthoc.mean_ranks.index[0]
                            best_rank_configs_data[best_rank_config] = filtered_table[
                                best_rank_config
                            ]
                            f.write(
                                f"  Statistically Top Group (Equivalent to {best_rank_config}):\n"
                            )
                            for c in top_group:
                                f.write(f"    * {c} (Mean Rank: {posthoc.mean_ranks[c]:.2f})\n")
                    except ValueError as e:
                        f.write(f"  -> Stats Error: {e}\n")

                elif k == 2:
                    config1, config2 = filtered_table.columns
                    paired = stats.compare_two(filtered_table, config1, config2)
                    f.write(f"  Wilcoxon p-value:     {paired.wilcoxon_p_value:.4e}\n")
                    if paired.wilcoxon_p_value < 0.05:
                        winner = config2 if paired.r_plus > paired.r_minus else config1
                        f.write(f"  -> Significant difference. Winner: {winner}\n")
                        # NEW: Send the Wilcoxon winner to the tournament
                        best_rank_configs_data[winner] = filtered_table[winner]
                    else:
                        f.write("  -> No significant difference between the two champions.\n")
                        # NEW: If tied, send the absolute best to the tournament
                        best_rank_configs_data[config1] = filtered_table[config1]
                else:
                    f.write("  -> Only 1 champion extracted. No statistical comparison possible.\n")
                    # NEW: Send the only champion to the tournament
                    best_rank_configs_data[best_absolute_config] = filtered_table[
                        best_absolute_config
                    ]

                f.write("-" * 80 + "\n\n")

            # =====================================================================
            # 4. CROSS-MODEL HEAD-TO-HEAD (Best RF vs Best SVM)
            # =====================================================================
            if len(best_rank_configs_data) == 2:
                names = list(best_rank_configs_data.keys())
                config1, config2 = names[0], names[1]

                # Combine the two absolute bests into a 10x2 DataFrame
                head_to_head_table = pd.concat(
                    [best_rank_configs_data[config1], best_rank_configs_data[config2]], axis=1
                )

                print(f"  -> Running Wilcoxon Head-to-Head on {dataset.upper()}...")

                f.write(f"CROSS-MODEL HEAD-TO-HEAD: {dataset.upper()}\n")
                f.write(f"  Model 1: {config1}\n")
                f.write(f"  Model 2: {config2}\n")

                paired = stats.compare_two(head_to_head_table, config1, config2)
                f.write(f"  Wilcoxon p-value:     {paired.wilcoxon_p_value:.4e}\n")

                if paired.wilcoxon_p_value < 0.05:
                    winner = config2 if paired.r_plus > paired.r_minus else config1
                    f.write(f"  -> Significant difference. Winner: {winner}\n")
                else:
                    f.write("  -> No significant difference between the absolute best models.\n")

                f.write("=" * 80 + "\n\n")

    print(f"\nSummary report successfully saved to: {report_file}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manual Grid Search CV for hyperparameter tuning.")
    parser.add_argument("--dataset", choices=["vowel", "splice", "all"], default="all")
    parser.add_argument("--classifier", choices=["rf", "svm", "knn", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path("results/gridsearch"))

    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    datasets = ["vowel", "splice"] if args.dataset == "all" else [args.dataset]
    models = ["rf", "svm", "knn"] if args.classifier == "all" else [args.classifier]

    # 1. Run all combinations
    run_grid_search(datasets, models, args.out)

    # 2. Read all files, find the best, and save to optimal_results.txt
    evaluate_results(args.out)


if __name__ == "__main__":
    main()
