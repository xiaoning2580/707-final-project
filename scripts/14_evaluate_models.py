"""Evaluate the seven saved test-set predictions.

Run from the local final_project directory:
    python scripts/14_evaluate_models.py

Inputs are the COMMON_COVARIATES_V2 predictions in data/part2 and data/part3,
the shared assignments in data/part1, and the original Set A outcomes.
Use --outcomes to supply a different outcome-file location.

Bootstrap intervals condition on the fitted models and the original split.
They do not include uncertainty from model training or hyperparameter tuning.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUTCOME = "In-hospital_death"
MODEL_FILES = {
    "age_only": "part2/age_only.csv",
    "lasso_values": "part2/lasso_values.csv",
    "lasso_patterns": "part2/lasso_patterns.csv",
    "lasso_combined": "part2/lasso_combined.csv",
    "rf_values": "part3/rf_values.csv",
    "rf_patterns": "part3/rf_patterns.csv",
    "rf_combined": "part3/rf_combined.csv",
}
LABELS = {
    "age_only": "Age-only logistic",
    "lasso_values": "Values",
    "lasso_patterns": "Patterns",
    "lasso_combined": "Combined",
    "rf_values": "Values",
    "rf_patterns": "Patterns",
    "rf_combined": "Combined",
}


def read_table(path, columns):
    table = pd.read_csv(path, dtype={"RecordID": str})
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Missing required columns in {path}: {columns}")
    if table["RecordID"].isna().any() or table["RecordID"].duplicated().any():
        raise ValueError(f"Missing or duplicate RecordID in {path}")
    return table.set_index("RecordID")


def load_test_data(data_dir, outcome_path):
    splits = read_table(data_dir / "part1/split_assignments.csv", ["RecordID", "split"])
    outcomes = read_table(outcome_path, ["RecordID", OUTCOME])
    if len(splits) != 4000 or not splits["split"].isin(["train", "test"]).all():
        raise ValueError("Expected 4,000 shared train/test assignments.")
    if set(splits.index) != set(outcomes.index):
        raise ValueError("Outcome IDs do not match the shared assignments.")
    if not outcomes[OUTCOME].isin([0, 1]).all():
        raise ValueError("Outcomes must be coded as 0 or 1 without missing values.")

    test_ids = splits.index[splits["split"].eq("test")]
    if len(test_ids) != 800:
        raise ValueError("Expected exactly 800 test stays.")
    test = outcomes.loc[test_ids, [OUTCOME]].copy()
    if test[OUTCOME].nunique() != 2:
        raise ValueError("The test set must contain both outcome classes.")

    # Align each model by RecordID, never by CSV row position.
    for model, filename in MODEL_FILES.items():
        prediction = read_table(data_dir / filename, ["RecordID", "predicted_probability"])
        if set(prediction.index) != set(test_ids):
            raise ValueError(f"Test IDs do not match for {model}.")
        probability = prediction.loc[test_ids, "predicted_probability"].to_numpy(dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"Invalid predicted probabilities for {model}.")
        test[model] = probability
    return test


def calculate_metrics(test):
    y = test[OUTCOME].to_numpy()
    rows = []
    for model in MODEL_FILES:
        p = test[model].to_numpy()
        rows.append({
            "model": model,
            "n_test": len(y),
            "deaths": int(y.sum()),
            "auroc": roc_auc_score(y, p),
            "brier_score": brier_score_loss(y, p),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(test, metrics, n_bootstrap, seed):
    y = test[OUTCOME].to_numpy()
    rng = np.random.default_rng(seed)
    differences = {family: [] for family in ["lasso", "rf"]}
    skipped = 0
    completed = 0
    while completed < n_bootstrap:
        # Use the same sampled patients for both predictions and both families.
        sample = rng.integers(0, len(y), size=len(y))
        y_sample = y[sample]
        if np.unique(y_sample).size < 2:
            skipped += 1
            continue
        for family in differences:
            values = test[f"{family}_values"].to_numpy()[sample]
            combined = test[f"{family}_combined"].to_numpy()[sample]
            differences[family].append([
                roc_auc_score(y_sample, combined) - roc_auc_score(y_sample, values),
                brier_score_loss(y_sample, combined) - brier_score_loss(y_sample, values),
            ])
        completed += 1

    scores = metrics.set_index("model")
    rows = []
    for family, samples in differences.items():
        samples = np.asarray(samples)
        for column, metric in enumerate(["auroc", "brier_score"]):
            lower, upper = np.percentile(samples[:, column], [2.5, 97.5])
            rows.append({
                "model_family": family,
                "comparison": "combined_minus_values",
                "metric": metric,
                "difference": scores.loc[f"{family}_combined", metric]
                - scores.loc[f"{family}_values", metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            })
    print(f"Bootstrap: {completed} valid resamples; {skipped} single-class resamples skipped.")
    return pd.DataFrame(rows)


def calibration_bins(test):
    rows = []
    for model in MODEL_FILES:
        frame = test[[OUTCOME, model]].copy()
        # Five quantile bins avoid very small groups; tied edges are collapsed.
        frame["bin"] = pd.qcut(frame[model], q=5, duplicates="drop")
        if frame["bin"].isna().all():
            frame["bin"] = 0
        for number, (_, group) in enumerate(frame.groupby("bin", observed=True), start=1):
            rows.append({
                "model": model,
                "bin": number,
                "n": len(group),
                "deaths": int(group[OUTCOME].sum()),
                "mean_prediction": group[model].mean(),
                "observed_mortality": group[OUTCOME].mean(),
            })
    return pd.DataFrame(rows)


def save_figure(fig, output_dir, name):
    for extension in ["png", "pdf"]:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results(test, metrics, calibration, output_dir):
    scores = metrics.set_index("model")
    styles = {
        "age_only": ("0.4", "--"),
        "values": ("#0072B2", "-"),
        "patterns": ("#D55E00", "-"),
        "combined": ("#009E73", "-"),
    }
    for plot_name in ["roc_curves", "calibration"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, family, title in zip(axes, ["lasso", "rf"], ["LASSO logistic", "Random forest"]):
            ax.plot([0, 1], [0, 1], color="0.75", linestyle=":")
            for model in ["age_only"] + [f"{family}_{kind}" for kind in ["values", "patterns", "combined"]]:
                kind = "age_only" if model == "age_only" else model.split("_", 1)[1]
                color, linestyle = styles[kind]
                label = LABELS[model]
                if plot_name == "roc_curves":
                    x, y, _ = roc_curve(test[OUTCOME], test[model])
                    label += f" (AUC {scores.loc[model, 'auroc']:.3f})"
                    ax.plot(x, y, color=color, linestyle=linestyle, label=label)
                else:
                    bins = calibration.loc[calibration["model"].eq(model)]
                    ax.plot(bins["mean_prediction"], bins["observed_mortality"],
                            marker="o", color=color, linestyle=linestyle, label=label)
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9, loc="lower right" if plot_name == "roc_curves" else "upper left")
            ax.set_xlabel("False positive rate" if plot_name == "roc_curves" else "Mean predicted mortality")
        axes[0].set_ylabel("True positive rate" if plot_name == "roc_curves" else "Observed mortality")
        fig.tight_layout()
        save_figure(fig, output_dir, plot_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--outcomes", type=Path,
                        default=ROOT / "707-final-project-main/data/Outcomes-a.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=707)
    args = parser.parse_args()
    if args.n_bootstrap < 2:
        parser.error("--n-bootstrap must be at least 2.")

    test = load_test_data(args.data_dir, args.outcomes)
    print(f"Validated seven models: {len(test)} test stays, {int(test[OUTCOME].sum())} deaths.")
    metrics = calculate_metrics(test)
    differences = paired_bootstrap(test, metrics, args.n_bootstrap, args.seed)
    calibration = calibration_bins(test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_bins.csv", index=False)
    plot_results(test, metrics, calibration, args.output_dir)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(differences.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Differences are Combined minus Values: positive AUROC and negative Brier favor Combined.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
