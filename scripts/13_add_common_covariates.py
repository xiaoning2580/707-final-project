"""Build revised feature sets with common admission covariates.

Inputs:
    data/processed/

Outputs:
    data/processed/common_covariates_v2/

No imputation, standardization, or split generation is performed.
"""

from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed"
DESTINATION = SOURCE / "common_covariates_v2"
VERSION = "COMMON_COVARIATES_V2"
COMMON = ["Age", "Gender", "ICUType"]

FEATURE_FILES = {
    "values": "features_values.csv",
    "patterns": "features_patterns.csv",
    "combined": "features_combined.csv",
}
OLD_COUNTS = {"values": 111, "patterns": 74, "combined": 185}
NEW_COUNTS = {"values": 114, "patterns": 77, "combined": 188}

COPY_FILES = [
    "baseline.csv",
    "static_descriptors.csv",
    "outcomes.csv",
    "split_assignments.csv",
]

SOURCE_FILES = list(FEATURE_FILES.values()) + COPY_FILES + [
    "feature_dictionary.csv",
    "feature_sets.json",
    "preparation_manifest.json",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path):
    frame = pd.read_csv(path, dtype={"RecordID": "string"})
    require("RecordID" in frame, f"{path.name}: missing RecordID")
    require(frame["RecordID"].notna().all(), f"{path.name}: missing IDs")
    require(frame["RecordID"].is_unique, f"{path.name}: duplicate IDs")
    return frame.set_index("RecordID").sort_index()


def equal_frames(left, right):
    pd.testing.assert_frame_equal(
        left,
        right,
        check_dtype=False,
        check_exact=True,
    )


def write_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    require(
        not DESTINATION.exists(),
        f"Output directory already exists: {DESTINATION}\n"
        "Existing outputs were not overwritten. Archive that directory "
        "before intentionally rebuilding this version.",
    )

    for name in SOURCE_FILES:
        require((SOURCE / name).is_file(), f"Missing source file: {name}")

    source_hashes = {name: sha256(SOURCE / name) for name in SOURCE_FILES}

    static = read_table(SOURCE / "static_descriptors.csv")
    require(len(static) == 4000, "Expected 4,000 stays.")
    require(set(COMMON).issubset(static.columns), "Missing common variables.")

    common = static[COMMON].apply(pd.to_numeric, errors="raise")
    require(
        np.isfinite(common.to_numpy(dtype=float, na_value=np.nan)
                    [common.notna().to_numpy()]).all(),
        "Common covariates contain infinite values.",
    )

    for column, allowed in [
        ("Gender", {0, 1}),
        ("ICUType", {1, 2, 3, 4}),
    ]:
        require(
            set(common[column].dropna().unique()).issubset(allowed),
            f"Unexpected codes in {column}.",
        )

    originals = {}
    revised = {}

    for group, filename in FEATURE_FILES.items():
        frame = read_table(SOURCE / filename)
        require(frame.index.equals(common.index), f"{group}: ID mismatch")
        require(
            frame.shape[1] == OLD_COUNTS[group],
            f"{group}: unexpected original feature count",
        )
        require(
            not set(COMMON).intersection(frame.columns),
            f"{group}: common covariates are already present",
        )

        originals[group] = frame
        revised[group] = pd.concat([common, frame], axis=1)

        require(
            revised[group].shape == (4000, NEW_COUNTS[group]),
            f"{group}: incorrect revised dimensions",
        )
        require(
            revised[group].columns.is_unique,
            f"{group}: duplicate feature columns",
        )
        equal_frames(revised[group][frame.columns], frame)
        equal_frames(revised[group][COMMON], common)

    value_columns = list(originals["values"].columns)
    pattern_columns = list(originals["patterns"].columns)

    require(
        not set(value_columns).intersection(pattern_columns),
        "Value and pattern feature names overlap.",
    )
    require(
        set(originals["combined"].columns)
        == set(value_columns) | set(pattern_columns),
        "Original Combined has unexpected columns.",
    )
    equal_frames(originals["combined"][value_columns], originals["values"])
    equal_frames(originals["combined"][pattern_columns], originals["patterns"])

    require(
        originals["patterns"].notna().all().all(),
        "Original measurement-pattern features contain missing values.",
    )

    copied_tables = {}
    for name in COPY_FILES:
        frame = read_table(SOURCE / name)
        require(frame.index.equals(common.index), f"{name}: ID mismatch")
        copied_tables[name] = frame

    baseline = copied_tables["baseline.csv"]
    equal_frames(
        baseline[["Age"]].apply(pd.to_numeric, errors="raise"),
        common[["Age"]],
    )

    split = copied_tables["split_assignments.csv"]
    require(
        {"split", "cv_fold"}.issubset(split.columns),
        "Unexpected split schema.",
    )
    require(
        split["split"].value_counts().to_dict() == {"train": 3200, "test": 800},
        "Unexpected training/test assignments.",
    )
    training_ids = split.index[split["split"].eq("train")]

    folds = pd.to_numeric(split.loc[training_ids, "cv_fold"], errors="raise")
    require(
        folds.notna().all()
        and set(folds) == {1, 2, 3, 4, 5}
        and folds.value_counts().eq(640).all(),
        "Unexpected training CV folds.",
    )

    dictionary = pd.read_csv(SOURCE / "feature_dictionary.csv")
    required_dictionary_columns = {
        "feature", "source_variable", "statistic",
        "feature_family", "definition",
    }
    require(
        required_dictionary_columns.issubset(dictionary.columns),
        "Unexpected feature dictionary schema.",
    )
    require(
        not dictionary["feature"].isin(COMMON).any(),
        "Common covariates already occur in the source dictionary.",
    )

    additions = pd.DataFrame([
        {
            "feature": "Age",
            "source_variable": "Age",
            "statistic": "static",
            "feature_family": "common",
            "definition": (
                "Age in years; numeric; shared by all three feature sets; "
                "missing values preserved"
            ),
        },
        {
            "feature": "Gender",
            "source_variable": "Gender",
            "statistic": "static",
            "feature_family": "common",
            "definition": (
                "Categorical: 0=female, 1=male; missing values preserved; "
                "shared by all three feature sets"
            ),
        },
        {
            "feature": "ICUType",
            "source_variable": "ICUType",
            "statistic": "static",
            "feature_family": "common",
            "definition": (
                "Categorical: 1=coronary care, 2=cardiac surgery recovery, "
                "3=medical ICU, 4=surgical ICU; missing values preserved; "
                "shared by all three feature sets"
            ),
        },
    ])
    new_dictionary = pd.concat([additions, dictionary], ignore_index=True)
    require(new_dictionary["feature"].is_unique, "Duplicate dictionary entries.")

    # Explicit schema for the revised handoff; do not assume the old JSON layout.
    feature_sets = {
        "schema_version": VERSION,
        "id_column": "RecordID",
        "common_covariates": COMMON,
        "categorical_features": ["Gender", "ICUType"],
        "feature_sets": {
            group: {
                "file": FEATURE_FILES[group],
                "n_predictors_before_encoding": NEW_COUNTS[group],
                "columns": list(revised[group].columns),
                "numeric_columns": [
                    column for column in revised[group].columns
                    if column not in ["Gender", "ICUType"]
                ],
                "categorical_columns": ["Gender", "ICUType"],
            }
            for group in FEATURE_FILES
        },
    }

    manifest = json.loads(
        (SOURCE / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update({
        "feature_schema_version": VERSION,
        "common_covariates": COMMON,
        "static_covariates_in_main_feature_sets": True,
        "n_value_features": 114,
        "n_pattern_features": 77,
        "n_combined_features": 188,
        "feature_count_definition": (
            "Predictor columns before categorical encoding; excludes RecordID"
        ),
        "n_measurement_value_features": 111,
        "n_measurement_pattern_features": 74,
        "n_measurement_combined_features": 185,
        "imputation_performed": False,
        "standardization_performed": False,
        "categorical_encoding_performed": False,
        "split_action": "Copied existing assignments without regeneration",
        "split_stratification_variable": "In-hospital_death",
        "cv_stratification_variable": "In-hospital_death",
        "source_sha256": source_hashes,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })

    # Stage outputs; publish the directory only after all checks pass.
    stage = Path(tempfile.mkdtemp(prefix=".common_v2_", dir=SOURCE))
    try:
        for group, filename in FEATURE_FILES.items():
            revised[group].to_csv(stage / filename, index_label="RecordID")

            # Verify the written CSV, allowing only numeric serialization rounding.
            reread = read_table(stage / filename)
            pd.testing.assert_frame_equal(
                reread,
                revised[group],
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

        for name in COPY_FILES:
            shutil.copy2(SOURCE / name, stage / name)
            require(
                sha256(stage / name) == source_hashes[name],
                f"Copied file differs from source: {name}",
            )

        new_dictionary.to_csv(stage / "feature_dictionary.csv", index=False)
        write_json(stage / "feature_sets.json", feature_sets)

        # Preserve the old manifest separately for provenance.
        shutil.copy2(
            SOURCE / "preparation_manifest.json",
            stage / "source_preparation_manifest.json",
        )

        # Training-only description of common covariates.
        training_common = common.loc[training_ids]
        training_common.describe().to_csv(
            stage / "training_common_covariate_summary.csv"
        )
        training_common.isna().sum().rename("n_missing").to_csv(
            stage / "training_common_covariate_missingness.csv",
            index_label="feature",
        )

        category_rows = []
        for fold in range(1, 6):
            subsets = {
                "fitting": folds.index[folds.ne(fold)],
                "validation": folds.index[folds.eq(fold)],
            }
            for role, ids in subsets.items():
                for column, categories in [
                    ("Gender", [0, 1]),
                    ("ICUType", [1, 2, 3, 4]),
                ]:
                    series = common.loc[ids, column]
                    for category in categories:
                        category_rows.append({
                            "cv_fold": fold,
                            "subset": role,
                            "variable": column,
                            "category": category,
                            "n": int(series.eq(category).sum()),
                        })
                    category_rows.append({
                        "cv_fold": fold,
                        "subset": role,
                        "variable": column,
                        "category": "missing",
                        "n": int(series.isna().sum()),
                    })

        pd.DataFrame(category_rows).to_csv(
            stage / "training_cv_category_counts.csv", index=False
        )

        checks = {
            "all_tables_have_4000_matching_unique_ids": True,
            "original_measurement_features_preserved_in_memory": True,
            "csv_roundtrip_verified_with_tolerance_1e_minus12": True,
            "common_covariates_identical_across_feature_sets": True,
            "combined_contains_common_covariates_once": True,
            "measurement_pattern_columns_have_no_missing_values": True,
            "baseline_age_matches_static_age": True,
            "baseline_file_byte_identical": True,
            "outcomes_file_byte_identical": True,
            "split_assignments_file_byte_identical": True,
            "no_imputation_scaling_or_categorical_encoding": True,
        }

        for name, expected_hash in source_hashes.items():
            require(
                sha256(SOURCE / name) == expected_hash,
                f"Source file changed during execution: {name}",
            )
        checks["source_files_unchanged"] = True

        manifest["revision_validation"] = checks
        write_json(stage / "preparation_manifest.json", manifest)
        write_json(stage / "validation_summary.json", checks)

        output_hashes = {
            path.name: sha256(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        write_json(stage / "file_checksums.json", output_hashes)

        (stage / "README.md").write_text(
            """# Revised common-covariate inputs

Version: COMMON_COVARIATES_V2

All three feature sets include Age, Gender, and ICUType.
Predictor counts before encoding are 114, 77, and 188.
RecordID is an alignment key, not a predictor.

Age is numeric. Gender and ICUType must be treated as categorical.
Missing values are preserved. Fit any learned imputation, encoding,
scaling, and feature filtering inside the cross-validation pipeline.
Use the same categorical missing-data convention across model families.

The measurement-pattern columns remain complete. Common covariates
may contain missing values, so do not require the entire revised
Patterns table to be complete.

Combined contains common covariates only once. Do not concatenate
the revised Values and Patterns tables without removing their
shared common-covariate columns.

The original split_assignments.csv, outcomes.csv, baseline.csv, and
static_descriptors.csv are copied byte-for-byte. The original
training/test split and training folds were stratified by
In-hospital_death; this revision does not regenerate assignments.

feature_sets.json uses an explicit versioned schema:
feature_sets -> values/patterns/combined -> columns.
Model loaders must be updated for this schema and the revised counts.

Reproduction from the repository root:
    python scripts/06_build_features.py
    python scripts/07_training_eda.py
    python scripts/13_add_common_covariates.py

Script 07 continues to use the original measurement-only tables.
Script 13 creates this revised modeling directory and refuses to
overwrite it. Archive an existing revised directory before rebuilding.

If source preparation is rerun, rebuild this directory before modeling.
Do not use an older revised directory with newly generated source files.

The file_checksums.json inventory covers generated data and metadata
files, excluding this README and the checksum inventory itself.
""",
            encoding="utf-8",
        )

        stage.rename(DESTINATION)

    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"Version: {VERSION}")
    for group in FEATURE_FILES:
        print(
            f"{group}: {len(revised[group])} stays, "
            f"{revised[group].shape[1]} predictors"
        )

    print("\nTraining common-covariate missingness:")
    print(training_common.isna().sum().to_string())
    print("\nAll validation checks passed.")
    print("Baseline, outcomes, and split files are byte-identical to source.")
    print("No imputation, standardization, or categorical encoding performed.")
    print(f"\nSaved revised inputs to: {DESTINATION}")


if __name__ == "__main__":
    main()