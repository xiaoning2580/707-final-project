"""Apply blood-pressure rule B, rebuild features, and regenerate EDA.

Run from the repository root:
    python scripts/10_apply_bp_rule_b.py

Backs up existing outputs before making changes.
Keeps existing train/test and CV assignments.
Does not apply the candidate C range thresholds.
"""

from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "06_build_features.py"
EDA = ROOT / "scripts" / "07_training_eda.py"
PROCESSED = ROOT / "data" / "processed"
AUDIT = ROOT / "data" / "audit"
SPLIT = PROCESSED / "split_assignments.csv"

MARKER = "# BP_RULE_B_V1"

OLD_CLEAN = '''        window["clean_value"] = window["numeric_value"].mask(reason.ne(""))'''

NEW_CLEAN = '''        # BP_RULE_B_V1
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

        window["clean_value"] = window["numeric_value"].mask(reason.ne(""))'''

MANIFEST_ANCHOR = '''        "n_missing_age": int(static["Age"].isna().sum()),'''

MANIFEST_REPLACEMENT = '''        "blood_pressure_rule": "B_zero_and_double_zero_context",
        "blood_pressure_rule_version": "BP_RULE_B_V1",
        "n_excluded_bp_zero": int(excluded["reason"].eq("bp_zero").sum()),
        "n_excluded_bp_double_zero_context": int(
            excluded["reason"].eq("bp_double_zero_context").sum()
        ),
        "n_missing_age": int(static["Age"].isna().sum()),'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_split(path):
    df = pd.read_csv(path, dtype={"RecordID": str})
    require(df["RecordID"].is_unique, "Duplicate split IDs.")
    require(df["RecordID"].notna().all(), "Missing split IDs.")
    df["cv_fold"] = pd.to_numeric(
        df["cv_fold"], errors="raise"
    ).astype("Int64")
    return df.set_index("RecordID").sort_index()


def main():
    for path in [BUILD, EDA, SPLIT]:
        require(path.exists(), f"Required file missing: {path}")

    split_before = read_split(SPLIT)
    require(len(split_before) == 4000, "Expected 4000 split assignments.")

    source = BUILD.read_text()

    if MARKER not in source:
        require(
            source.count(OLD_CLEAN) == 1,
            "Could not uniquely locate the cleaning line. "
            "No files have been changed; check 06_build_features.py.",
        )
        require(
            source.count(MANIFEST_ANCHOR) == 1,
            "Could not uniquely locate the manifest line. "
            "No files have been changed.",
        )
        updated = source.replace(OLD_CLEAN, NEW_CLEAN, 1)
        updated = updated.replace(
            MANIFEST_ANCHOR, MANIFEST_REPLACEMENT, 1
        )
    else:
        require(
            NEW_CLEAN in source and MANIFEST_REPLACEMENT in source,
            "Rule B marker exists but implementation differs. "
            "Review the file before proceeding.",
        )
        updated = source
        print("Rule B is already installed; rebuilding consistently.")

    # Validate syntax before writing.
    compile(updated, str(BUILD), "exec")

    # Backup is outside data/audit to avoid recursive copying.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = ROOT / "data" / "backups" / f"before_bp_B_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    items = [
        Path("scripts/06_build_features.py"),
        Path("scripts/07_training_eda.py"),
        Path("data/processed"),
        Path("data/audit"),
        Path("results/tables/training_eda"),
        Path("results/figures/training_eda"),
    ]

    for relative in items:
        original = ROOT / relative
        if not original.exists():
            continue
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(original, destination)
        else:
            shutil.copy2(original, destination)

    print(f"Backup saved to: {backup}", flush=True)

    if updated != source:
        BUILD.write_text(updated)
        print("Installed rule B in 06_build_features.py.", flush=True)

    subprocess.run(
        [sys.executable, str(BUILD)],
        cwd=ROOT,
        check=True,
    )

    split_after = read_split(SPLIT)
    pd.testing.assert_frame_equal(split_before, split_after)
    print("\nVerified: train/test and CV assignments are unchanged.")

    excluded = pd.read_csv(
        AUDIT / "excluded_measurement_records.csv",
        dtype={"RecordID": str},
    )
    require(
        excluded["RecordID"].isin(split_after.index).all(),
        "Excluded-record audit contains unknown IDs.",
    )
    excluded["split"] = excluded["RecordID"].map(split_after["split"])

    bp_reasons = ["bp_zero", "bp_double_zero_context"]
    bp = excluded.loc[excluded["reason"].isin(bp_reasons)].copy()

    # Match the training-only audit already reviewed.
    expected = {
        "MAP": 325,
        "SysABP": 623,
        "DiasABP": 619,
        "NIMAP": 89,
        "NISysABP": 143,
        "NIDiasABP": 90,
    }
    training_bp = bp.loc[bp["split"].eq("train")]
    actual = training_bp.groupby("Parameter").size().to_dict()

    require(
        actual == expected,
        f"Training exclusions differ from audit. "
        f"Expected {expected}; found {actual}. "
        f"Backup is available at {backup}.",
    )
    require(
        training_bp["RecordID"].nunique() == 359,
        "Expected 359 affected training stays.",
    )

    report_rows = []
    for population in ["all", "train", "test"]:
        subset = bp if population == "all" else bp.loc[
            bp["split"].eq(population)
        ]
        report_rows.append({
            "population": population,
            "n_bp_zero": int(subset["reason"].eq("bp_zero").sum()),
            "n_bp_context": int(
                subset["reason"].eq("bp_double_zero_context").sum()
            ),
            "n_bp_excluded_total": len(subset),
            "n_affected_stays": subset["RecordID"].nunique(),
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(AUDIT / "bp_rule_B_applied_summary.csv", index=False)

    by_variable = (
        bp.groupby(["split", "Parameter", "reason"])
        .size()
        .reset_index(name="n_excluded")
    )
    by_variable.to_csv(
        AUDIT / "bp_rule_B_applied_by_variable.csv", index=False
    )

    # Record per-stay BP exclusions alongside the existing cleaning counts.
    cleaning_path = AUDIT / "cleaning_by_stay.csv"
    cleaning = pd.read_csv(
        cleaning_path, dtype={"RecordID": str}
    ).set_index("RecordID")

    for reason, column in [
        ("bp_zero", "n_bp_zero"),
        ("bp_double_zero_context", "n_bp_double_zero_context"),
    ]:
        counts = bp.loc[bp["reason"].eq(reason)].groupby("RecordID").size()
        cleaning[column] = counts.reindex(
            cleaning.index, fill_value=0
        ).astype(int)

    cleaning.to_csv(cleaning_path)

    manifest_path = PROCESSED / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_assignments_verified_unchanged"] = True
    manifest["bp_rule_B_training_audit_matched"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nBlood-pressure rule B applied:")
    print(report.to_string(index=False))
    print("\nRebuilding training EDA...", flush=True)

    subprocess.run(
        [sys.executable, str(EDA)],
        cwd=ROOT,
        check=True,
    )

    # Recheck after both scripts finish.
    pd.testing.assert_frame_equal(split_before, read_split(SPLIT))

    print("\nCompleted:")
    print("- Rule B is installed in the feature-building script.")
    print("- Features were rebuilt for all 4000 stays.")
    print("- Training exclusions match the previous audit: 1889 rows.")
    print("- Train/test and CV assignments are unchanged.")
    print("- Training EDA was regenerated.")
    print(f"- Previous files are backed up at: {backup}")


if __name__ == "__main__":
    main()