"""Inspect repeated Parameter-Time groups within each ICU stay.

Run:
    python scripts/03_inspect_repeated_times.py

Raw files are never modified.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "set-a"
OUTPUT = ROOT / "data" / "audit"

STATIC_PARAMETERS = {
    "RecordID", "Age", "Gender", "Height", "ICUType"
}

CATEGORIES = [
    "no_valid_value",
    "one_valid_row",
    "multiple_valid_same_value",
    "multiple_valid_different_values",
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    files = sorted(RECORDS.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No measurement files found in {RECORDS}")

    group_rows = []
    detail_frames = []
    raw_extra_total = 0

    for index, path in enumerate(files, start=1):
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()

        required = {"Time", "Parameter", "Value"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path.name}: missing required columns.")

        for column in required:
            df[column] = df[column].str.strip()

        df["source_line"] = range(2, len(df) + 2)
        df["numeric_value"] = pd.to_numeric(df["Value"], errors="raise")

        values = df["numeric_value"]
        finite = pd.Series(
            np.isfinite(values.to_numpy(dtype=float)),
            index=df.index,
        )

        df["value_status"] = "valid_nonnegative"
        df.loc[values.eq(-1), "value_status"] = "missing_minus1"
        df.loc[
            values.lt(0) & values.ne(-1),
            "value_status",
        ] = "other_negative"
        df.loc[~finite, "value_status"] = "nonfinite"

        # Match the definition used in the original audit:
        # extra rows beyond the first per Parameter-Time group.
        raw_extra_total += int(
            df.duplicated(["Time", "Parameter"]).sum()
        )

        repeated = df.loc[
            df.duplicated(["Time", "Parameter"], keep=False)
        ].copy()

        for (time, parameter), group in repeated.groupby(
            ["Time", "Parameter"], sort=True
        ):
            hours, minutes = map(int, time.split(":"))
            time_minutes = hours * 60 + minutes

            valid = group.loc[
                group["value_status"].eq("valid_nonnegative"),
                "numeric_value",
            ]

            n_rows = len(group)
            n_valid = len(valid)
            n_unique_valid = valid.nunique()

            if n_valid == 0:
                category = "no_valid_value"
            elif n_valid == 1:
                category = "one_valid_row"
            elif n_unique_valid == 1:
                category = "multiple_valid_same_value"
            else:
                category = "multiple_valid_different_values"

            group_id = f"{path.stem}|{parameter}|{time}"

            group_rows.append({
                "group_id": group_id,
                "RecordID": path.stem,
                "Parameter": parameter,
                "Time": time,
                "time_minutes": time_minutes,
                "is_static_parameter": parameter in STATIC_PARAMETERS,
                "n_rows": n_rows,
                "n_extra_rows": n_rows - 1,
                "n_unique_raw_values": group["numeric_value"].nunique(),
                "n_valid_rows": n_valid,
                "n_valid_extra_rows": max(n_valid - 1, 0),
                "n_unique_valid_values": n_unique_valid,
                "n_minus1": int(
                    group["value_status"].eq("missing_minus1").sum()
                ),
                "n_other_negative": int(
                    group["value_status"].eq("other_negative").sum()
                ),
                "n_nonfinite": int(
                    group["value_status"].eq("nonfinite").sum()
                ),
                "valid_min": valid.min() if n_valid else np.nan,
                "valid_max": valid.max() if n_valid else np.nan,
                "valid_range": (
                    valid.max() - valid.min() if n_valid else np.nan
                ),
                "category": category,
                # Preserve original order and raw text for inspection.
                "raw_values": " | ".join(group["Value"].tolist()),
            })

            detail = group[
                [
                    "source_line",
                    "Time",
                    "Parameter",
                    "Value",
                    "numeric_value",
                    "value_status",
                ]
            ].copy()
            detail.insert(0, "RecordID", path.stem)
            detail.insert(0, "group_id", group_id)
            detail["category"] = category
            detail_frames.append(detail)

        if index % 500 == 0:
            print(f"Checked {index}/{len(files)} files")

    if not group_rows:
        print("No repeated Parameter-Time groups were found.")
        return

    groups = pd.DataFrame(group_rows).sort_values(
        ["Parameter", "RecordID", "time_minutes"]
    )
    details = pd.concat(detail_frames, ignore_index=True)

    reconstructed_total = int(groups["n_extra_rows"].sum())
    if reconstructed_total != raw_extra_total:
        raise RuntimeError("Repeated-row totals do not reconcile.")

    by_variable = (
        groups.groupby("Parameter")
        .agg(
            n_groups=("group_id", "size"),
            n_affected_stays=("RecordID", "nunique"),
            n_rows_in_groups=("n_rows", "sum"),
            n_extra_rows=("n_extra_rows", "sum"),
            n_valid_extra_rows=("n_valid_extra_rows", "sum"),
            n_minus1=("n_minus1", "sum"),
            n_other_negative=("n_other_negative", "sum"),
            max_rows_per_group=("n_rows", "max"),
        )
    )

    category_counts = pd.crosstab(
        groups["Parameter"], groups["category"]
    ).reindex(columns=CATEGORIES, fill_value=0)
    category_counts.columns = [
        f"groups_{name}" for name in category_counts.columns
    ]

    by_variable = (
        by_variable.join(category_counts)
        .reset_index()
        .sort_values("n_extra_rows", ascending=False)
    )

    by_category = (
        groups.groupby("category")
        .agg(
            n_groups=("group_id", "size"),
            n_affected_stays=("RecordID", "nunique"),
            n_extra_rows=("n_extra_rows", "sum"),
        )
        .reindex(CATEGORIES, fill_value=0)
        .reset_index()
    )

    # Deterministic examples: up to 3 groups per variable and category.
    examples = (
        groups.sort_values(
            ["Parameter", "category", "RecordID", "time_minutes"]
        )
        .groupby(["Parameter", "category"], sort=False)
        .head(3)
    )

    groups.to_csv(
        OUTPUT / "repeated_time_groups.csv", index=False
    )
    details.to_csv(
        OUTPUT / "repeated_time_details.csv", index=False
    )
    by_variable.to_csv(
        OUTPUT / "repeated_time_by_variable.csv", index=False
    )
    by_category.to_csv(
        OUTPUT / "repeated_time_by_category.csv", index=False
    )
    examples.to_csv(
        OUTPUT / "repeated_time_examples.csv", index=False
    )

    summary = {
        "n_files_checked": len(files),
        "n_affected_stays": int(groups["RecordID"].nunique()),
        "n_repeated_groups": len(groups),
        "n_rows_in_repeated_groups": int(groups["n_rows"].sum()),
        "n_extra_rows": reconstructed_total,
        "matches_previous_3744": reconstructed_total == 3744,
        "n_valid_extra_rows_after_negative_removal": int(
            groups["n_valid_extra_rows"].sum()
        ),
        "n_groups_with_multiple_different_valid_values": int(
            groups["category"].eq(
                "multiple_valid_different_values"
            ).sum()
        ),
        "n_repeated_static_groups": int(
            groups["is_static_parameter"].sum()
        ),
    }

    (OUTPUT / "repeated_time_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nOverall summary:")
    print(json.dumps(summary, indent=2))

    print("\nSummary by category:")
    print(by_category.to_string(index=False))

    print("\nSummary by variable:")
    print(
        by_variable[
            [
                "Parameter",
                "n_groups",
                "n_affected_stays",
                "n_extra_rows",
                "n_valid_extra_rows",
                "n_minus1",
                "n_other_negative",
                "max_rows_per_group",
            ]
        ].to_string(index=False)
    )

    print("\nExamples: first repeated group for each variable")
    first_examples = (
        groups.groupby("Parameter", sort=True).head(1)
    )
    print(
        first_examples[
            ["RecordID", "Parameter", "Time", "raw_values", "category"]
        ].to_string(index=False)
    )

    if reconstructed_total != 3744:
        print(
            "\nNOTE: Total differs from the previous audit (3744). "
            "Check whether the raw files or parsing rules changed."
        )

    print(f"\nSaved outputs to: {OUTPUT}")
    print("Original data have not been modified.")


if __name__ == "__main__":
    main()