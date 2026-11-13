"""Back up and inspect inputs before adding common covariates."""

from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
COMMON = ["Age", "Gender", "ICUType"]

REQUIRED = [
    "features_values.csv",
    "features_patterns.csv",
    "features_combined.csv",
    "static_descriptors.csv",
    "baseline.csv",
    "outcomes.csv",
    "split_assignments.csv",
    "feature_dictionary.csv",
    "feature_sets.json",
    "preparation_manifest.json",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_indexed(filename):
    frame = pd.read_csv(
        PROCESSED / filename,
        dtype={"RecordID": "string"},
    )
    require("RecordID" in frame.columns, f"{filename}: missing RecordID")
    require(frame["RecordID"].notna().all(), f"{filename}: missing IDs")
    require(frame["RecordID"].is_unique, f"{filename}: duplicate IDs")
    return frame.set_index("RecordID").sort_index()


def main():
    missing_files = [
        name for name in REQUIRED if not (PROCESSED / name).is_file()
    ]
    require(not missing_files, f"Missing files: {missing_files}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = ROOT / "data" / "backups" / f"before_common_covariates_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    # Preserve all current processed inputs, including metadata.
    shutil.copytree(PROCESSED, backup / "processed")

    # Preserve the scripts and documentation needed to interpret this version.
    for folder in ["scripts", "docs"]:
        source = ROOT / folder
        if source.is_dir():
            shutil.copytree(
                source,
                backup / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    for name in ["README.md", ".gitignore"]:
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, backup / name)

    original_hashes = {
        name: sha256(PROCESSED / name) for name in REQUIRED
    }
    for name in REQUIRED:
        require(
            original_hashes[name] == sha256(backup / "processed" / name),
            f"Backup verification failed: {name}",
        )

    print(f"Verified backup saved to: {backup}", flush=True)

    tables = {
        name: read_indexed(name)
        for name in REQUIRED
        if name.endswith(".csv") and name != "feature_dictionary.csv"
    }

    static = tables["static_descriptors.csv"]
    require(len(static) == 4000, "Expected 4,000 stays.")
    require(
        set(COMMON).issubset(static.columns),
        f"Missing common covariates. Available columns: {list(static.columns)}",
    )

    for name, frame in tables.items():
        require(
            frame.index.equals(static.index),
            f"{name}: RecordID set differs from static_descriptors.csv",
        )

    # Validate codes without changing the source data.
    numeric_common = static[COMMON].apply(pd.to_numeric, errors="raise")
    for column, allowed in [
        ("Gender", {0, 1}),
        ("ICUType", {1, 2, 3, 4}),
    ]:
        observed = set(numeric_common[column].dropna().unique())
        require(
            observed.issubset(allowed),
            f"{column}: unexpected codes {observed - allowed}. "
            "Review these before applying the update.",
        )

    values = tables["features_values.csv"]
    patterns = tables["features_patterns.csv"]
    combined = tables["features_combined.csv"]

    expected = [
        ("Values", values, 111),
        ("Patterns", patterns, 74),
        ("Combined", combined, 185),
    ]

    for label, frame, count in expected:
        require(
            not set(COMMON).intersection(frame.columns),
            f"{label} already contains common covariates. "
            "Stop to avoid applying the update twice.",
        )
        require(
            frame.shape[1] == count,
            f"{label}: expected {count} predictors, found {frame.shape[1]}",
        )

    require(
        not set(values.columns).intersection(patterns.columns),
        "Existing Values and Patterns have overlapping feature names.",
    )
    require(
        set(combined.columns) == set(values.columns) | set(patterns.columns),
        "Combined feature names do not match Values plus Patterns.",
    )

    pd.testing.assert_frame_equal(
        combined.loc[:, values.columns],
        values,
        check_dtype=False,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        combined.loc[:, patterns.columns],
        patterns,
        check_dtype=False,
        check_exact=True,
    )

    baseline = tables["baseline.csv"]
    require("Age" in baseline.columns, "baseline.csv does not contain Age")
    pd.testing.assert_series_equal(
        pd.to_numeric(baseline["Age"], errors="raise"),
        numeric_common["Age"],
        check_dtype=False,
        check_exact=True,
    )

    split = tables["split_assignments.csv"]
    require(
        {"split", "cv_fold"}.issubset(split.columns),
        f"Unexpected split columns: {list(split.columns)}",
    )
    require(
        set(split["split"]) == {"train", "test"},
        "Unexpected split labels.",
    )
    require(
        int(split["split"].eq("train").sum()) == 3200
        and int(split["split"].eq("test").sum()) == 800,
        "Unexpected training/test sizes.",
    )

    train_folds = pd.to_numeric(
        split.loc[split["split"].eq("train"), "cv_fold"],
        errors="raise",
    )
    require(train_folds.notna().all(), "Missing training fold assignments")
    require(set(train_folds) == {1, 2, 3, 4, 5}, "Unexpected CV fold labels")
    require(
        train_folds.value_counts().eq(640).all(),
        "Expected 640 validation stays in each training fold.",
    )

    # Report only training-set covariate distributions.
    training_ids = split.index[split["split"].eq("train")]
    training_common = numeric_common.loc[training_ids]

    lines = [
        "COMMON COVARIATE UPDATE: PREPARATION CHECK",
        "",
        f"Backup: {backup}",
        "All source tables have the same 4,000 unique RecordIDs.",
        "Original Combined equals Values plus Patterns.",
        "Baseline Age matches static Age.",
        "Training/test sizes: 3,200 / 800.",
        "Training folds: five folds of 640 stays.",
        "",
        "CURRENT TABLE SCHEMAS",
    ]

    for name, frame in tables.items():
        columns = list(frame.columns)
        preview = columns if len(columns) <= 12 else columns[:8] + ["..."]
        lines.append(
            f"{name}: {len(frame)} rows, {len(columns)} non-ID columns; "
            f"columns={preview}"
        )

    lines.extend([
        "",
        "TRAINING COMMON-COVARIATE MISSINGNESS",
        training_common.isna().sum().to_string(),
        "",
        "TRAINING AGE SUMMARY",
        training_common["Age"].describe().to_string(),
    ])

    for column in ["Gender", "ICUType"]:
        lines.extend([
            "",
            f"TRAINING {column} COUNTS",
            training_common[column].value_counts(
                dropna=False
            ).sort_index().to_string(),
        ])

    for name in ["feature_sets.json", "preparation_manifest.json"]:
        payload = json.loads((PROCESSED / name).read_text(encoding="utf-8"))
        lines.extend([
            "",
            f"METADATA: {name}",
            json.dumps(payload, indent=2, ensure_ascii=False),
        ])

    dictionary = pd.read_csv(PROCESSED / "feature_dictionary.csv")
    lines.extend([
        "",
        "FEATURE DICTIONARY SCHEMA AND FIRST FIVE ROWS",
        dictionary.head().to_string(index=False),
    ])

    # Confirm that this inspection has not changed the original inputs.
    for name, expected_hash in original_hashes.items():
        require(
            sha256(PROCESSED / name) == expected_hash,
            f"Source file changed during inspection: {name}",
        )

    audit_dir = ROOT / "data" / "audit" / "common_covariate_update"
    audit_dir.mkdir(parents=True, exist_ok=True)

    report_path = audit_dir / f"pre_update_check_{stamp}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    snapshot = {
        "backup_directory": str(backup),
        "source_sha256": original_hashes,
        "checks_passed": True,
        "source_files_modified": False,
    }
    snapshot_path = audit_dir / f"pre_update_snapshot_{stamp}.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nInspection report: {report_path}")
    print(f"Hash snapshot: {snapshot_path}")
    print("\nAll preparation checks passed.")
    print("Original feature tables and split assignments were not modified.")


if __name__ == "__main__":
    main()