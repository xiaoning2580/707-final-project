"""Build ICU-level features and shared train/test/CV assignments.

Run from the repository root:
    python scripts/06_build_features.py

Raw data are never modified.
No imputation, scaling, model fitting, or outcome-based feature selection.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDS = DATA / "set-a"
OUTPUT = DATA / "processed"
AUDIT = DATA / "audit"

SEED = 707
TEST_SIZE = 0.20
N_FOLDS = 5

# Fixed feature schema: do not select variables using observed outcomes.
VARIABLES = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS",
    "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
    "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets",
    "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC", "Weight",
]

STATIC = ["Age", "Gender", "Height", "ICUType"]

# Accommodate documented abbreviations while keeping output names consistent.
ALIASES = {
    "TropI": "TroponinI",
    "TropT": "TroponinT",
}

VALUE_STATS = ["mean", "min", "max"]
PATTERN_STATS = ["count", "missing"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_record(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()

    require(
        {"Time", "Parameter", "Value"}.issubset(df.columns),
        f"{path.name}: missing required columns.",
    )

    for col in ["Time", "Parameter", "Value"]:
        df[col] = df[col].str.strip()

    df["source_line"] = range(2, len(df) + 2)
    df["raw_parameter"] = df["Parameter"]
    df["Parameter"] = df["Parameter"].replace(ALIASES)
    df["numeric_value"] = pd.to_numeric(df["Value"], errors="raise")

    require(
        np.isfinite(df["numeric_value"].to_numpy(dtype=float)).all(),
        f"{path.name}: nonfinite measurement value.",
    )

    parts = df["Time"].str.extract(r"^(\d+):([0-5]\d)$")
    require(
        parts.notna().all().all(),
        f"{path.name}: invalid timestamp.",
    )
    df["time_minutes"] = (
        parts[0].astype(int) * 60 + parts[1].astype(int)
    )

    unknown = set(df["Parameter"]) - set(VARIABLES + STATIC + ["RecordID"])
    require(not unknown, f"{path.name}: unknown parameters: {unknown}")

    ids = df.loc[df["Parameter"].eq("RecordID"), "numeric_value"]
    require(
        len(ids) == 1 and ids.iloc[0] == int(path.stem),
        f"{path.name}: internal RecordID mismatch.",
    )

    # Prevent silently combining alternative names within one stay.
    for canonical in set(ALIASES.values()):
        names = df.loc[
            df["Parameter"].eq(canonical), "raw_parameter"
        ].unique()
        require(
            len(names) <= 1,
            f"{path.name}: multiple aliases for {canonical}: {names}",
        )

    return df


def validate_features(values, patterns, combined):
    require(values.index.is_unique, "Duplicate feature IDs.")
    require(values.index.equals(patterns.index), "Feature IDs differ.")
    require(values.index.equals(combined.index), "Combined IDs differ.")

    for variable in VARIABLES:
        count = patterns[f"{variable}_count"]
        missing = patterns[f"{variable}_missing"]
        block = values[[f"{variable}_{s}" for s in VALUE_STATS]]

        require(
            (count.ge(0) & count.eq(count.astype(int))).all(),
            f"{variable}: invalid counts.",
        )
        require(
            missing.eq(count.eq(0).astype(int)).all(),
            f"{variable}: missing indicator mismatch.",
        )
        require(
            block.loc[count.eq(0)].isna().all().all(),
            f"{variable}: nonmissing summaries when count is zero.",
        )
        require(
            block.loc[count.gt(0)].notna().all().all(),
            f"{variable}: missing summaries despite positive count.",
        )

        observed = count.gt(0)
        low = values.loc[observed, f"{variable}_min"]
        mean = values.loc[observed, f"{variable}_mean"]
        high = values.loc[observed, f"{variable}_max"]
        tolerance = 1e-10 * (1 + low.abs() + high.abs())

        require(
            ((low <= mean + tolerance) & (mean <= high + tolerance)).all(),
            f"{variable}: min/mean/max ordering failed.",
        )

        single = count.eq(1)
        require(
            np.allclose(
                values.loc[single, f"{variable}_mean"],
                values.loc[single, f"{variable}_min"],
            )
            and np.allclose(
                values.loc[single, f"{variable}_mean"],
                values.loc[single, f"{variable}_max"],
            ),
            f"{variable}: single-observation summaries disagree.",
        )

    require(
        combined.equals(pd.concat([values, patterns], axis=1)),
        "Combined is not exactly values plus patterns.",
    )


def make_or_load_splits(outcomes):
    """Reuse existing assignments instead of silently changing a shared split."""
    path = OUTPUT / "split_assignments.csv"
    ids = outcomes.index

    if path.exists():
        split = pd.read_csv(path, dtype={"RecordID": str})
        require(split["RecordID"].is_unique, "Duplicate IDs in existing split.")
        require(set(split["RecordID"]) == set(ids), "Existing split IDs differ.")
        split = split.set_index("RecordID").loc[ids]
        print("Reusing existing split_assignments.csv")
    else:
        train_ids, test_ids = train_test_split(
            ids.to_numpy(),
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=outcomes["In-hospital_death"].to_numpy(),
        )

        # Stable ordering before generating folds.
        train_ids = np.sort(train_ids)
        test_ids = np.sort(test_ids)

        split = pd.DataFrame(index=ids)
        split["split"] = "train"
        split.loc[test_ids, "split"] = "test"
        split["cv_fold"] = pd.Series(pd.NA, index=ids, dtype="Int64")

        cv = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        )
        train_y = outcomes.loc[train_ids, "In-hospital_death"].to_numpy()

        for fold, (_, validation_index) in enumerate(
            cv.split(np.zeros(len(train_ids)), train_y), start=1
        ):
            split.loc[train_ids[validation_index], "cv_fold"] = fold

    require(
        split["split"].isin(["train", "test"]).all(),
        "Invalid split labels.",
    )
    require(
        split["split"].eq("train").sum() == 3200
        and split["split"].eq("test").sum() == 800,
        "Expected 3200 training and 800 test stays.",
    )

    train_mask = split["split"].eq("train")
    fold_values = pd.to_numeric(split["cv_fold"], errors="raise")
    require(
        fold_values.loc[train_mask].isin(range(1, N_FOLDS + 1)).all(),
        "Invalid training fold assignments.",
    )
    require(
        fold_values.loc[~train_mask].isna().all(),
        "Test stays must not have CV fold assignments.",
    )
    require(
        set(fold_values.loc[train_mask]) == set(range(1, N_FOLDS + 1)),
        "Some CV folds are absent.",
    )
    split["cv_fold"] = fold_values.astype("Int64")
    split.index.name = "RecordID"

    check = split.join(outcomes)
    for label, group in check.groupby("split"):
        require(
            group["In-hospital_death"].nunique() == 2,
            f"{label}: only one outcome class.",
        )
    for fold, group in check.loc[train_mask].groupby("cv_fold"):
        require(
            group["In-hospital_death"].nunique() == 2,
            f"CV fold {fold}: only one outcome class.",
        )

    return split


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    files = sorted(RECORDS.glob("*.txt"))
    require(len(files) == 4000, f"Expected 4000 files; found {len(files)}.")

    outcomes = pd.read_csv(DATA / "Outcomes-a.txt", dtype=str)
    outcomes.columns = outcomes.columns.str.strip()
    outcomes = outcomes[["RecordID", "In-hospital_death"]].copy()
    outcomes["RecordID"] = outcomes["RecordID"].str.strip()
    require(outcomes["RecordID"].is_unique, "Duplicate outcome IDs.")
    require(
        set(outcomes["RecordID"]) == {p.stem for p in files},
        "Measurement and outcome IDs do not match.",
    )

    y = pd.to_numeric(outcomes["In-hospital_death"], errors="raise")
    require(y.isin([0, 1]).all(), "Outcome must be 0 or 1.")
    outcomes["In-hospital_death"] = y.astype(int)
    outcomes = outcomes.set_index("RecordID").sort_index()

    value_rows = []
    pattern_rows = []
    static_rows = []
    timestamp_rows = []
    excluded_rows = []
    cleaning_rows = []
    seen_variables = set()

    for index, path in enumerate(files, start=1):
        record_id = path.stem
        df = read_record(path)

        # Window filtering is explicit even though the audit found no violations.
        in_window = df["time_minutes"].between(0, 2880)
        window = df.loc[in_window].copy()

        # Stop on any newly discovered negative values needing a new rule.
        unexpected_negative = (
            window["numeric_value"].lt(0)
            & window["numeric_value"].ne(-1)
            & window["Parameter"].ne("Temp")
        )
        require(
            not unexpected_negative.any(),
            f"{path.name}: unexpected negative non-temperature value.",
        )

        reason = pd.Series("", index=window.index, dtype=object)
        reason.loc[window["numeric_value"].eq(-1)] = "missing_minus1"
        reason.loc[
            window["Parameter"].eq("Temp")
            & window["numeric_value"].lt(10)
            & window["numeric_value"].ne(-1)
        ] = "temperature_below_10"

        # BP_RULE_B_V1
        # Evaluate context BEFORE masking zero values.
        bp_variables = [
            "MAP", "SysABP", "DiasABP",
            "NIMAP", "NISysABP", "NIDiasABP",
        ]
        families = [
            ("MAP", "SysABP", "DiasABP"),
            ("NIMAP", "NISysABP", "NIDiasABP"),
        ]

        baseline_bp = window.loc[
            window["Parameter"].isin(bp_variables)
            & window["numeric_value"].ge(0)
        ]

        context_mask = pd.Series(False, index=window.index)

        for mean_name, sys_name, dia_name in families:
            supporting = baseline_bp.loc[
                baseline_bp["Parameter"].isin([sys_name, dia_name])
            ]
            supporting = (
                supporting.groupby(["time_minutes", "Parameter"])
                ["numeric_value"]
                .agg(n="size", value="first")
            )

            zero_times = {}
            for name in [sys_name, dia_name]:
                if name in supporting.index.get_level_values("Parameter"):
                    table = supporting.xs(name, level="Parameter")
                    zero_times[name] = set(
                        table.index[
                            table["n"].eq(1) & table["value"].eq(0)
                        ]
                    )
                else:
                    zero_times[name] = set()

            matched_times = zero_times[sys_name] & zero_times[dia_name]
            context_mask |= (
                window["Parameter"].eq(mean_name)
                & window["time_minutes"].isin(matched_times)
                & window["numeric_value"].ge(0)
            )

        bp_zero = (
            window["Parameter"].isin(bp_variables)
            & window["numeric_value"].eq(0)
        )

        # Exclusive reason labels: zero takes priority when both apply.
        reason.loc[bp_zero] = "bp_zero"
        reason.loc[context_mask & ~bp_zero] = "bp_double_zero_context"

        window["clean_value"] = window["numeric_value"].mask(reason.ne(""))

        if reason.ne("").any():
            removed = window.loc[
                reason.ne(""),
                ["source_line", "Time", "Parameter", "Value"],
            ].copy()
            removed.insert(0, "RecordID", record_id)
            removed["reason"] = reason.loc[reason.ne("")]
            excluded_rows.append(removed)

        static = {"RecordID": record_id}
        for variable in STATIC:
            observations = window.loc[
                window["Parameter"].eq(variable), "clean_value"
            ].dropna().unique()
            require(
                len(observations) <= 1,
                f"{path.name}: conflicting static values for {variable}.",
            )
            static[variable] = (
                float(observations[0]) if len(observations) else np.nan
            )
        static_rows.append(static)

        time_series = window.loc[window["Parameter"].isin(VARIABLES)]
        seen_variables.update(time_series["Parameter"])

        value_row = {"RecordID": record_id}
        pattern_row = {"RecordID": record_id}
        timestamp_row = {"RecordID": record_id}

        for variable in VARIABLES:
            observations = time_series.loc[
                time_series["Parameter"].eq(variable)
            ]
            valid_rows = observations.loc[observations["clean_value"].notna()]
            valid_values = valid_rows["clean_value"]
            n = len(valid_values)

            value_row[f"{variable}_mean"] = valid_values.mean() if n else np.nan
            value_row[f"{variable}_min"] = valid_values.min() if n else np.nan
            value_row[f"{variable}_max"] = valid_values.max() if n else np.nan
            pattern_row[f"{variable}_count"] = n
            pattern_row[f"{variable}_missing"] = int(n == 0)

            # Diagnostic only: not included in any of the three model matrices.
            timestamp_row[f"{variable}_unique_times"] = (
                valid_rows["time_minutes"].nunique()
            )

        value_rows.append(value_row)
        pattern_rows.append(pattern_row)
        timestamp_rows.append(timestamp_row)

        cleaning_rows.append({
            "RecordID": record_id,
            "n_rows_outside_window": int((~in_window).sum()),
            "n_minus1": int(reason.eq("missing_minus1").sum()),
            "n_temp_below10_excluding_minus1": int(
                reason.eq("temperature_below_10").sum()
            ),
        })

        if index % 500 == 0:
            print(f"Processed {index}/{len(files)} stays")

    require(
        seen_variables == set(VARIABLES),
        f"Expected variables absent from all records: "
        f"{set(VARIABLES) - seen_variables}",
    )

    def indexed(rows):
        return (
            pd.DataFrame(rows)
            .set_index("RecordID")
            .loc[outcomes.index]
        )

    values = indexed(value_rows)
    patterns = indexed(pattern_rows).astype(int)
    combined = pd.concat([values, patterns], axis=1)
    static = indexed(static_rows)
    timestamps = indexed(timestamp_rows).astype(int)
    cleaning = indexed(cleaning_rows)

    validate_features(values, patterns, combined)
    splits = make_or_load_splits(outcomes)

    excluded = (
        pd.concat(excluded_rows, ignore_index=True)
        if excluded_rows else pd.DataFrame(
            columns=["RecordID", "source_line", "Time", "Parameter",
                     "Value", "reason"]
        )
    )

    # Dataset-specific checks against the reviewed audit.
    require(
        int(patterns["Temp_count"].sum()) == 86276,
        "Temperature count differs from expected 86276; review inputs/rules.",
    )
    require(
        int(patterns["Temp_missing"].sum()) == 64,
        "Temperature missing-stay count differs from expected 64.",
    )
    require(
        int(cleaning["n_temp_below10_excluding_minus1"].sum()) == 129,
        "Expected 129 excluded temperature records.",
    )

    feature_sets = {
        "values": list(values.columns),
        "patterns": list(patterns.columns),
        "combined": list(combined.columns),
        "baseline": ["Age"],
    }

    dictionary = []
    definitions = {
        "mean": "Arithmetic mean of retained records; NA if none",
        "min": "Minimum of retained records; NA if none",
        "max": "Maximum of retained records; NA if none",
        "count": "Number of retained records, including same-time records",
        "missing": "1 if count is zero; otherwise 0",
    }
    for variable in VARIABLES:
        for statistic in VALUE_STATS + PATTERN_STATS:
            dictionary.append({
                "feature": f"{variable}_{statistic}",
                "source_variable": variable,
                "statistic": statistic,
                "feature_family": (
                    "values" if statistic in VALUE_STATS else "patterns"
                ),
                "definition": definitions[statistic],
            })

    split_check = splits.join(outcomes)
    split_summary = (
        split_check.groupby("split")["In-hospital_death"]
        .agg(n="size", deaths="sum", mortality_rate="mean")
        .reset_index()
    )
    fold_summary = (
        split_check.loc[split_check["split"].eq("train")]
        .groupby("cv_fold")["In-hospital_death"]
        .agg(n="size", deaths="sum", mortality_rate="mean")
        .reset_index()
    )

    # Save only after the main validation checks have passed.
    values.to_csv(OUTPUT / "features_values.csv", na_rep="NA")
    patterns.to_csv(OUTPUT / "features_patterns.csv")
    combined.to_csv(OUTPUT / "features_combined.csv", na_rep="NA")
    static[["Age"]].to_csv(OUTPUT / "baseline.csv", na_rep="NA")
    static.to_csv(OUTPUT / "static_descriptors.csv", na_rep="NA")
    outcomes.to_csv(OUTPUT / "outcomes.csv")
    splits.to_csv(OUTPUT / "split_assignments.csv", na_rep="NA")

    timestamps.to_csv(AUDIT / "valid_unique_timestamps.csv")
    cleaning.to_csv(AUDIT / "cleaning_by_stay.csv")
    excluded.to_csv(AUDIT / "excluded_measurement_records.csv", index=False)
    split_summary.to_csv(AUDIT / "split_summary.csv", index=False)
    fold_summary.to_csv(AUDIT / "cv_fold_summary.csv", index=False)
    pd.DataFrame(dictionary).to_csv(
        OUTPUT / "feature_dictionary.csv", index=False
    )
    (OUTPUT / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2) + "\n"
    )

    manifest = {
        "n_stays": len(values),
        "n_time_series_variables": len(VARIABLES),
        "n_value_features": values.shape[1],
        "n_pattern_features": patterns.shape[1],
        "n_combined_features": combined.shape[1],
        "n_deaths": int(outcomes["In-hospital_death"].sum()),
        "n_excluded_minus1": int(cleaning["n_minus1"].sum()),
        "n_excluded_temperature_below10": int(
            cleaning["n_temp_below10_excluding_minus1"].sum()
        ),
        "n_retained_temperature_records": int(patterns["Temp_count"].sum()),
        "n_stays_without_valid_temperature": int(patterns["Temp_missing"].sum()),
        "blood_pressure_rule": "B_zero_and_double_zero_context",
        "blood_pressure_rule_version": "BP_RULE_B_V1",
        "n_excluded_bp_zero": int(excluded["reason"].eq("bp_zero").sum()),
        "n_excluded_bp_double_zero_context": int(
            excluded["reason"].eq("bp_double_zero_context").sum()
        ),
        "n_missing_age": int(static["Age"].isna().sum()),
        "observation_window_minutes_inclusive": [0, 2880],
        "same_time_record_policy": "Retain all otherwise valid records",
        "temperature_rule": "Exclude Temp < 10; retain 24.2 and high values",
        "count_definition": "Number of valid records, not independent tests",
        "static_covariates_in_main_feature_sets": False,
        "imputation_performed": False,
        "standardization_performed": False,
        "new_split_seed": SEED,
        "split_policy": "Reuse existing assignments if present and valid",
    }

    (OUTPUT / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\nFeature construction summary:")
    print(json.dumps(manifest, indent=2))
    print("\nTraining/test summary:")
    print(split_summary.to_string(index=False))
    print("\nTraining CV validation-fold summary:")
    print(fold_summary.to_string(index=False))
    print(f"\nFeatures saved to: {OUTPUT}")
    print(f"Audit outputs saved to: {AUDIT}")
    print("All checks passed. No imputation or standardization performed.")


if __name__ == "__main__":
    main()