"""Training-set exploratory analysis for the BIOSTAT 707 project.

Run:
    python scripts/07_training_eda.py

Uses existing split assignments.
Does not modify features, impute values, or fit models.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables" / "training_eda"
FIGURES = ROOT / "results" / "figures" / "training_eda"

COLORS = {
    "Survived": "#4477AA",
    "Died": "#CC6677",
}

# Fixed illustrative variables, not selected by outcome associations.
REPRESENTATIVE = [
    "HR", "MAP", "Temp", "RespRate",
    "Creatinine", "Lactate", "Glucose", "WBC",
]

UNITS = {
    "HR": "beats/min",
    "MAP": "mmHg",
    "Temp": "degrees C",
    "RespRate": "breaths/min",
    "Creatinine": "mg/dL",
    "Lactate": "mmol/L",
    "Glucose": "mg/dL",
    "WBC": "10^3 cells/uL",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_indexed(filename):
    path = PROCESSED / filename
    require(path.exists(), f"Missing file: {path}")

    df = pd.read_csv(path, dtype={"RecordID": str})
    require("RecordID" in df.columns, f"{filename}: RecordID is missing.")
    require(df["RecordID"].notna().all(), f"{filename}: missing IDs.")
    require(df["RecordID"].is_unique, f"{filename}: duplicate IDs.")

    return df.set_index("RecordID")


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def continuous_text(series):
    observed = series.dropna()
    if observed.empty:
        return "NA"
    q1, median, q3 = observed.quantile([0.25, 0.50, 0.75])
    return f"{median:.1f} [{q1:.1f}, {q3:.1f}]"


def proportion_text(n, denominator):
    if denominator == 0:
        return f"{n} (NA)"
    return f"{n} ({100 * n / denominator:.1f}%)"


def sample_description(static, outcomes):
    y = outcomes["In-hospital_death"]
    groups = {
        "Overall": static.index,
        "Survived": static.index[y.eq(0)],
        "Died": static.index[y.eq(1)],
    }

    rows = []

    def add_row(label, calculate):
        row = {"Characteristic": label}
        for name, ids in groups.items():
            row[name] = calculate(ids)
        rows.append(row)

    add_row("ICU stays, n", lambda ids: len(ids))

    for variable in ["Age", "Height"]:
        add_row(
            f"{variable}, median [Q1, Q3]",
            lambda ids, v=variable: continuous_text(static.loc[ids, v]),
        )
        add_row(
            f"{variable} missing, n (%)",
            lambda ids, v=variable: proportion_text(
                int(static.loc[ids, v].isna().sum()), len(ids)
            ),
        )

    # Percentages use all stays in the respective column,
    # including missing values in the denominator.
    for variable, labels in {
        "Gender": {0: "Female", 1: "Male"},
        "ICUType": {
            1: "Coronary care",
            2: "Cardiac surgery recovery",
            3: "Medical ICU",
            4: "Surgical ICU",
        },
    }.items():
        for code, label in labels.items():
            add_row(
                f"{variable}: {label}, n (%)",
                lambda ids, v=variable, c=code: proportion_text(
                    int(static.loc[ids, v].eq(c).sum()), len(ids)
                ),
            )

        add_row(
            f"{variable}: missing, n (%)",
            lambda ids, v=variable: proportion_text(
                int(static.loc[ids, v].isna().sum()), len(ids)
            ),
        )

        add_row(
            f"{variable}: other code, n (%)",
            lambda ids, v=variable, codes=list(labels): proportion_text(
                int(
                    (
                        static.loc[ids, v].notna()
                        & ~static.loc[ids, v].isin(codes)
                    ).sum()
                ),
                len(ids),
            ),
        )

    return pd.DataFrame(rows)


def summarize_patterns(patterns, outcomes, variables):
    y = outcomes["In-hospital_death"]
    groups = {
        "Overall": patterns.index,
        "Survived": patterns.index[y.eq(0)],
        "Died": patterns.index[y.eq(1)],
    }

    rows = []
    for variable in variables:
        for group_name, ids in groups.items():
            count = patterns.loc[ids, f"{variable}_count"]
            observed = count.loc[count.gt(0)]

            rows.append({
                "variable": variable,
                "group": group_name,
                "n_stays": len(ids),
                "n_missing": int(count.eq(0).sum()),
                "missing_pct": 100 * count.eq(0).mean(),
                "n_observed": int(count.gt(0).sum()),
                # Includes zero-count stays.
                "count_mean_all": count.mean(),
                "count_q1_all": count.quantile(0.25),
                "count_median_all": count.median(),
                "count_q3_all": count.quantile(0.75),
                "count_p95_all": count.quantile(0.95),
                "count_max_all": count.max(),
                # Clearly separated conditional summary.
                "count_median_observed": (
                    observed.median() if len(observed) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def summarize_values(values):
    rows = []
    for feature in values.columns:
        observed = values[feature].dropna()
        row = {
            "feature": feature,
            "n_observed": len(observed),
            "n_missing": int(values[feature].isna().sum()),
        }

        for name, q in [
            ("minimum", 0),
            ("p01", 0.01),
            ("q1", 0.25),
            ("median", 0.5),
            ("q3", 0.75),
            ("p99", 0.99),
            ("maximum", 1),
        ]:
            row[name] = observed.quantile(q) if len(observed) else np.nan

        row["mean"] = observed.mean()
        row["sd"] = observed.std()
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    splits = read_indexed("split_assignments.csv")
    require(
        splits["split"].isin(["train", "test"]).all(),
        "Unexpected split labels.",
    )
    train_ids = splits.index[splits["split"].eq("train")]
    require(len(train_ids) == 3200, "Expected 3200 training stays.")

    # Read saved tables, then immediately restrict all analysis to training.
    tables = {}
    for name, filename in {
        "values": "features_values.csv",
        "patterns": "features_patterns.csv",
        "static": "static_descriptors.csv",
        "outcomes": "outcomes.csv",
    }.items():
        full = read_indexed(filename)
        require(
            set(full.index) == set(splits.index),
            f"{filename}: IDs do not match split assignments.",
        )
        tables[name] = full.loc[train_ids].copy()

    values = tables["values"]
    patterns = tables["patterns"]
    static = tables["static"]
    outcomes = tables["outcomes"]
    y = outcomes["In-hospital_death"]

    require(y.isin([0, 1]).all(), "Invalid outcomes.")
    require(y.nunique() == 2, "Both outcome classes are required.")

    variables = [
        col.removesuffix("_count")
        for col in patterns.columns if col.endswith("_count")
    ]
    require(len(variables) == 37, "Expected 37 variables.")

    for variable in variables:
        count = patterns[f"{variable}_count"]
        missing = patterns[f"{variable}_missing"]
        require(
            count.notna().all()
            and count.ge(0).all()
            and count.eq(np.floor(count)).all(),
            f"{variable}: invalid count.",
        )
        require(
            missing.eq(count.eq(0).astype(int)).all(),
            f"{variable}: missingness/count mismatch.",
        )
        for statistic in ["mean", "min", "max"]:
            feature = f"{variable}_{statistic}"
            require(feature in values, f"Missing feature: {feature}")
            require(
                values[feature].isna().eq(count.eq(0)).all(),
                f"{feature}: missingness does not match count.",
            )

    description = sample_description(static, outcomes)
    pattern_summary = summarize_patterns(patterns, outcomes, variables)
    value_summary = summarize_values(values)

    description.to_csv(TABLES / "01_sample_description.csv", index=False)
    pattern_summary.to_csv(
        TABLES / "02_measurement_patterns.csv", index=False
    )
    value_summary.to_csv(
        TABLES / "03_value_feature_distributions.csv", index=False
    )

    overall = (
        pattern_summary.loc[pattern_summary["group"].eq("Overall")]
        .set_index("variable")
    )

    # Figure 1: patient-level missingness.
    order = overall.sort_values("missing_pct", ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(order, overall.loc[order, "missing_pct"], color="#4477AA")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Stays with no valid record (%)")
    ax.set_title("Variable missingness in the training set (n = 3,200)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "01_missingness")

    # Figure 2: counts, including zero-count stays.
    count_order = overall.sort_values(
        "count_median_all", ascending=False
    ).index
    position = np.arange(len(count_order))
    ordered = overall.loc[count_order]

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.hlines(
        position,
        ordered["count_q1_all"],
        ordered["count_q3_all"],
        color="#4477AA",
        linewidth=3,
    )
    ax.scatter(
        ordered["count_median_all"], position,
        color="#222222", s=24, zorder=3,
    )
    ax.set_yticks(position)
    ax.set_yticklabels(count_order)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Valid record count: median and IQR (symlog scale)")
    ax.set_title("Recording frequency, including stays with zero records")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "02_record_counts")

    # Figure 3: missingness by outcome; denominators are group-specific.
    grouped_missing = pattern_summary.pivot(
        index="variable", columns="group", values="missing_pct"
    ).loc[order]

    fig, ax = plt.subplots(figsize=(10, 11))
    position = np.arange(len(order))
    ax.hlines(
        position,
        grouped_missing[["Survived", "Died"]].min(axis=1),
        grouped_missing[["Survived", "Died"]].max(axis=1),
        color="#BBBBBB",
    )
    for group in ["Survived", "Died"]:
        ax.scatter(
            grouped_missing[group], position,
            color=COLORS[group], label=group, s=30,
        )
    ax.set_yticks(position)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Stays with no valid record within outcome group (%)")
    ax.set_title("Missingness by in-hospital outcome — training set")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "03_missingness_by_outcome")

    comparison = grouped_missing[["Survived", "Died"]].copy()
    comparison["difference_percentage_points_died_minus_survived"] = (
        comparison["Died"] - comparison["Survived"]
    )
    comparison.to_csv(TABLES / "04_missingness_comparison.csv")

    # Figure 4: ECDF of recording counts for fixed illustrative variables.
    # ECDF includes zeros and does not hide extreme counts.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, variable in zip(axes.flat, REPRESENTATIVE):
        for code, group in [(0, "Survived"), (1, "Died")]:
            x = np.sort(
                patterns.loc[y.eq(code), f"{variable}_count"].to_numpy()
            )
            cumulative = np.arange(1, len(x) + 1) / len(x)
            ax.step(
                x, cumulative, where="post",
                color=COLORS[group], label=group,
            )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_ylim(0, 1.02)
        ax.set_title(variable)
        ax.set_xlabel("Valid count (symlog)")
        ax.set_ylabel("Cumulative proportion")
        ax.grid(alpha=0.2)

    axes[0, 0].legend()
    fig.suptitle(
        "Recording-count distributions by outcome — training set",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "04_count_ecdf_by_outcome")

    # Figure 5: patient-level means, conditional on having a valid record.
    # log1p scales preserve all nonnegative observations, including zero.
    log_variables = {"Creatinine", "Lactate", "Glucose", "WBC"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    representative_rows = []

    for ax, variable in zip(axes.flat, REPRESENTATIVE):
        feature = f"{variable}_mean"

        for code, group in [(0, "Survived"), (1, "Died")]:
            observed = values.loc[y.eq(code), feature].dropna()
            x = np.sort(observed.to_numpy())

            representative_rows.append({
                "variable": variable,
                "group": group,
                "n_observed": len(observed),
                "n_missing": int(
                    values.loc[y.eq(code), feature].isna().sum()
                ),
                "median": observed.median(),
                "q1": observed.quantile(0.25),
                "q3": observed.quantile(0.75),
            })

            if not len(x):
                continue

            if variable in log_variables:
                require((x >= 0).all(), f"{variable}: negative mean.")
                x = np.log1p(x)

            cumulative = np.arange(1, len(x) + 1) / len(x)
            ax.step(
                x, cumulative, where="post",
                color=COLORS[group],
                label=f"{group} (n={len(x)})",
            )

        ax.set_title(variable)
        label = f"Stay mean ({UNITS[variable]})"
        if variable in log_variables:
            label = f"log1p of stay mean ({UNITS[variable]})"
        ax.set_xlabel(label)
        ax.set_ylabel("Cumulative proportion")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Patient-level measurement means among observed stays — training set",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "05_value_ecdf_by_outcome")

    pd.DataFrame(representative_rows).to_csv(
        TABLES / "05_representative_value_summary.csv", index=False
    )

    # Flag constant/all-missing features for model-team awareness.
    # Do not drop them here: fold-specific preprocessing belongs in CV.
    combined = pd.concat([values, patterns], axis=1)
    diagnostics = pd.DataFrame({
        "feature": combined.columns,
        "n_missing": combined.isna().sum().to_numpy(),
        "n_unique_observed": combined.nunique(dropna=True).to_numpy(),
    })
    diagnostics["all_missing"] = (
        diagnostics["n_missing"].eq(len(combined))
    )
    diagnostics["constant_among_observed"] = (
        diagnostics["n_unique_observed"].eq(1)
    )
    diagnostics.to_csv(
        TABLES / "06_feature_diagnostics.csv", index=False
    )

    # A stay with no valid time-series data is retained, but reported.
    count_columns = [f"{v}_count" for v in variables]
    n_no_valid_time_series = int(
        patterns[count_columns].sum(axis=1).eq(0).sum()
    )

    summary = {
        "analysis_population": "training only",
        "n_training_stays": len(train_ids),
        "n_survivors": int(y.eq(0).sum()),
        "n_deaths": int(y.eq(1).sum()),
        "mortality_rate": float(y.mean()),
        "n_time_series_variables": len(variables),
        "n_stays_without_any_valid_time_series": n_no_valid_time_series,
        "n_all_missing_features": int(diagnostics["all_missing"].sum()),
        "n_constant_among_observed_features": int(
            diagnostics["constant_among_observed"].sum()
        ),
        "imputation_performed": False,
        "features_modified": False,
        "split_modified": False,
        "n_figures": 5,
        "figure_formats": ["png", "pdf"],
    }
    (TABLES / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nEDA summary:")
    print(json.dumps(summary, indent=2))

    print("\nTraining sample description:")
    print(description.to_string(index=False))

    print("\nTen variables with the highest missingness:")
    print(
        overall.sort_values("missing_pct", ascending=False)[
            [
                "n_missing", "missing_pct",
                "count_median_all", "count_q3_all",
            ]
        ].head(10).to_string()
    )

    print("\nTen variables with the largest median record counts:")
    print(
        overall.sort_values("count_median_all", ascending=False)[
            ["count_q1_all", "count_median_all", "count_q3_all"]
        ].head(10).to_string()
    )

    flagged = diagnostics.loc[
        diagnostics["all_missing"]
        | diagnostics["constant_among_observed"]
    ]
    print("\nAll-missing or constant-among-observed features:")
    print(flagged.to_string(index=False) if len(flagged) else "None")

    print(f"\nTables saved to: {TABLES}")
    print(f"Figures saved to: {FIGURES}")
    print("EDA completed. Features and split assignments were not changed.")


if __name__ == "__main__":
    main()