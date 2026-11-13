"""Audit original PhysioNet Challenge 2012 Set A files.

Run from the repository root:
    python scripts/01_check_raw_data.py

This script reads the raw files without modifying them.
Patient-level audit outputs remain under data/, which is ignored by Git.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDS = DATA / "set-a"
OUTCOMES = DATA / "Outcomes-a.txt"
AUDIT = DATA / "audit"


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)

    if not RECORDS.is_dir():
        raise FileNotFoundError(f"Missing directory: {RECORDS}")

    if not OUTCOMES.is_file():
        raise FileNotFoundError(f"Missing outcome file: {OUTCOMES}")

    files = sorted(RECORDS.glob("*.txt"))
    fatal = []
    record_rows = []
    variable_rows = []

    if len(files) != 4000:
        fatal.append(f"Expected 4000 measurement files; found {len(files)}.")

    # Read outcomes while retaining IDs as strings.
    outcomes = pd.read_csv(OUTCOMES, dtype=str)
    outcomes.columns = outcomes.columns.str.strip()

    required = {"RecordID", "In-hospital_death"}
    missing_columns = required - set(outcomes.columns)

    if missing_columns:
        raise ValueError(
            f"Outcome file is missing columns: {sorted(missing_columns)}"
        )

    outcomes["RecordID"] = outcomes["RecordID"].str.strip()

    id_valid = outcomes["RecordID"].str.fullmatch(r"\d+").fillna(False)
    if not id_valid.all():
        fatal.append("Outcome file contains missing or invalid RecordID values.")

    if outcomes["RecordID"].duplicated().any():
        fatal.append("Outcome file contains duplicate RecordID values.")

    if len(outcomes) != 4000:
        fatal.append(f"Expected 4000 outcome rows; found {len(outcomes)}.")

    y = pd.to_numeric(outcomes["In-hospital_death"], errors="coerce")

    if not y.isin([0, 1]).all():
        fatal.append("Outcome contains missing values or values other than 0/1.")

    file_ids = {p.stem for p in files}
    outcome_ids = set(outcomes.loc[id_valid, "RecordID"])

    missing_outcomes = sorted(file_ids - outcome_ids)
    missing_records = sorted(outcome_ids - file_ids)

    if missing_outcomes:
        fatal.append(
            f"{len(missing_outcomes)} measurement IDs have no outcome."
        )

    if missing_records:
        fatal.append(
            f"{len(missing_records)} outcome IDs have no measurement file."
        )

    pd.DataFrame(
        [{"RecordID": x, "issue": "missing_outcome"} for x in missing_outcomes]
        + [{"RecordID": x, "issue": "missing_measurement_file"}
           for x in missing_records],
        columns=["RecordID", "issue"],
    ).to_csv(AUDIT / "unmatched_ids.csv", index=False)

    for index, path in enumerate(files, start=1):
        if not path.stem.isdigit():
            fatal.append(f"Invalid measurement filename: {path.name}")

        try:
            df = pd.read_csv(path, dtype=str)
            df.columns = df.columns.str.strip()

            required_record = {"Time", "Parameter", "Value"}
            if not required_record.issubset(df.columns):
                raise ValueError(
                    f"Expected Time, Parameter, Value; got {list(df.columns)}"
                )

            if df.empty:
                raise ValueError("File contains no records.")

            for column in ["Time", "Parameter", "Value"]:
                df[column] = df[column].str.strip()

            value = pd.to_numeric(df["Value"], errors="coerce")
            finite = pd.Series(
                np.isfinite(value.to_numpy(dtype=float)),
                index=df.index,
            )

            # Elapsed hours may exceed 23; do not parse as clock time.
            parts = df["Time"].str.extract(r"^(\d+):([0-5]\d)$")
            minutes = (
                pd.to_numeric(parts[0], errors="coerce") * 60
                + pd.to_numeric(parts[1], errors="coerce")
            )

            bad_time = minutes.isna()
            outside_window = minutes.notna() & ~minutes.between(0, 2880)

            blank_parameter = (
                df["Parameter"].isna() | df["Parameter"].eq("")
            )

            internal_ids = value.loc[df["Parameter"].eq("RecordID")]
            id_matches = (
                len(internal_ids) == 1
                and path.stem.isdigit()
                and internal_ids.iloc[0] == int(path.stem)
            )

            if not id_matches:
                fatal.append(f"{path.name}: internal RecordID mismatch.")

            if bad_time.any():
                fatal.append(f"{path.name}: unparseable timestamp.")

            if blank_parameter.any():
                fatal.append(f"{path.name}: missing parameter name.")

            if (~finite).any():
                fatal.append(f"{path.name}: nonnumeric or nonfinite value.")

            record_rows.append({
                "RecordID": path.stem,
                "n_rows": len(df),
                "n_parameters": df["Parameter"].nunique(),
                "min_time_minutes": minutes.min(),
                "max_time_minutes": minutes.max(),
                "n_invalid_times": int(bad_time.sum()),
                "n_rows_outside_48h": int(outside_window.sum()),
                "n_minus1_values": int(value.eq(-1).sum()),
                "n_other_negative_values": int(
                    (value.lt(0) & value.ne(-1)).sum()
                ),
                "n_nonnumeric_or_nonfinite": int((~finite).sum()),
                "n_exact_duplicate_rows": int(df.duplicated().sum()),
                "n_repeated_parameter_time_rows": int(
                    df.duplicated(["Time", "Parameter"]).sum()
                ),
                "internal_id_matches_filename": bool(id_matches),
            })

            df = df.assign(numeric_value=value, finite_value=finite)

            for parameter, group in df.groupby("Parameter"):
                values = group["numeric_value"]
                valid = (
                    group["finite_value"]
                    & values.ge(0)
                )

                variable_rows.append({
                    "Parameter": parameter,
                    "n_records_with_parameter": 1,
                    "n_rows": len(group),
                    "n_minus1": int(values.eq(-1).sum()),
                    "n_other_negative": int(
                        (values.lt(0) & values.ne(-1)).sum()
                    ),
                    "n_nonnegative_finite": int(valid.sum()),
                    "min_nonnegative_value": (
                        values.loc[valid].min() if valid.any() else np.nan
                    ),
                    "max_nonnegative_value": (
                        values.loc[valid].max() if valid.any() else np.nan
                    ),
                })

        except Exception as exc:
            fatal.append(f"{path.name}: {type(exc).__name__}: {exc}")

        if index % 500 == 0:
            print(f"Checked {index}/{len(files)} files")

    record_audit = pd.DataFrame(record_rows)
    record_audit.to_csv(AUDIT / "record_audit.csv", index=False)

    if variable_rows:
        variable_audit = (
            pd.DataFrame(variable_rows)
            .groupby("Parameter", as_index=False)
            .agg(
                n_records_with_parameter=("n_records_with_parameter", "sum"),
                n_rows=("n_rows", "sum"),
                n_minus1=("n_minus1", "sum"),
                n_other_negative=("n_other_negative", "sum"),
                n_nonnegative_finite=("n_nonnegative_finite", "sum"),
                min_nonnegative_value=("min_nonnegative_value", "min"),
                max_nonnegative_value=("max_nonnegative_value", "max"),
            )
        )
        variable_audit.to_csv(AUDIT / "variable_audit.csv", index=False)

    def total(column):
        return int(record_audit[column].sum()) if not record_audit.empty else 0

    summary = {
        "n_measurement_files": len(files),
        "n_successfully_parsed_files": len(record_rows),
        "n_outcome_rows": len(outcomes),
        "n_unique_outcome_ids": int(outcomes["RecordID"].nunique()),
        "n_survivors": int(y.eq(0).sum()),
        "n_deaths": int(y.eq(1).sum()),
        "mortality_rate": float(y.mean()) if y.isin([0, 1]).all() else None,
        "n_measurement_ids_without_outcome": len(missing_outcomes),
        "n_outcome_ids_without_measurements": len(missing_records),
        "n_rows_outside_48h": total("n_rows_outside_48h"),
        "n_minus1_values": total("n_minus1_values"),
        "n_other_negative_values": total("n_other_negative_values"),
        "n_exact_duplicate_rows": total("n_exact_duplicate_rows"),
        "n_repeated_parameter_time_rows": total(
            "n_repeated_parameter_time_rows"
        ),
        "n_fatal_issues": len(fatal),
        "structural_checks_passed": len(fatal) == 0,
    }

    (AUDIT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    pd.DataFrame({"issue": fatal}).to_csv(
        AUDIT / "fatal_issues.csv", index=False
    )

    print("\nAudit summary:")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"\nSaved audit outputs to: {AUDIT}")

    if fatal:
        print("\nFirst issues:")
        for issue in fatal[:10]:
            print(f"- {issue}")
        raise SystemExit(1)

    print(
        "\nStructural checks passed. Review duplicates, negative values, "
        "time-window findings, and variable ranges before feature construction."
    )


if __name__ == "__main__":
    main()