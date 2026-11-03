"""Training-only blood-pressure audit and candidate-rule impact analysis.

Run:
    python scripts/09_audit_blood_pressure.py

No outcome file is read.
Raw data, processed features, and split assignments are not modified.
All candidate-rule results are hypothetical audit outputs.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "audit" / "blood_pressure_review"

BP = [
    "MAP", "SysABP", "DiasABP",
    "NIMAP", "NISysABP", "NIDiasABP",
]

FAMILIES = {
    "invasive": ("MAP", "SysABP", "DiasABP"),
    "noninvasive": ("NIMAP", "NISysABP", "NIDiasABP"),
}

RULES = [
    "A_zero_only",
    "B_zero_and_double_zero_context",
    "C_B_and_mean_pressure_screen",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_indexed(path):
    df = pd.read_csv(path, dtype={"RecordID": str})
    require(
        df["RecordID"].notna().all() and df["RecordID"].is_unique,
        f"Missing or duplicate IDs: {path}",
    )
    return df.set_index("RecordID")


def summarize_records(records, ids):
    """Return six-variable summaries, including stays with no records."""
    valid = records.loc[records["baseline_valid"]]
    summary = (
        valid.groupby(["RecordID", "Parameter"])["numeric_value"]
        .agg(count="size", mean="mean", min="min", max="max")
    )
    full_index = pd.MultiIndex.from_product(
        [ids, BP], names=["RecordID", "Parameter"]
    )
    summary = summary.reindex(full_index)
    summary["count"] = summary["count"].fillna(0).astype(int)
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    splits = read_indexed(DATA / "processed" / "split_assignments.csv")
    require(
        splits["split"].isin(["train", "test"]).all(),
        "Invalid split labels.",
    )
    train_ids = splits.index[splits["split"].eq("train")].sort_values()
    require(len(train_ids) == 3200, "Expected 3200 training stays.")

    frames = []

    for number, record_id in enumerate(train_ids, start=1):
        path = DATA / "set-a" / f"{record_id}.txt"
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()

        for column in ["Time", "Parameter", "Value"]:
            df[column] = df[column].str.strip()

        df["source_line"] = range(2, len(df) + 2)
        df = df.loc[df["Parameter"].isin(BP)].copy()

        parts = df["Time"].str.extract(r"^(\d+):([0-5]\d)$")
        require(
            parts.notna().all().all(),
            f"{record_id}: invalid timestamp.",
        )

        df["time_minutes"] = (
            parts[0].astype(int) * 60 + parts[1].astype(int)
        )
        df = df.loc[df["time_minutes"].between(0, 2880)].copy()
        df["numeric_value"] = pd.to_numeric(df["Value"], errors="raise")

        require(
            np.isfinite(df["numeric_value"]).all(),
            f"{record_id}: nonfinite blood-pressure value.",
        )
        require(
            not (
                df["numeric_value"].lt(0)
                & df["numeric_value"].ne(-1)
            ).any(),
            f"{record_id}: unexpected negative blood pressure.",
        )

        df.insert(0, "RecordID", record_id)
        frames.append(df)

        if number % 500 == 0:
            print(f"Read {number}/{len(train_ids)} training stays")

    records = pd.concat(frames, ignore_index=True)
    records["baseline_valid"] = records["numeric_value"].ge(0)

    baseline = summarize_records(records, train_ids)

    # Verify against the previously generated feature files.
    saved_values = read_indexed(
        DATA / "processed" / "features_values.csv"
    ).loc[train_ids]
    saved_patterns = read_indexed(
        DATA / "processed" / "features_patterns.csv"
    ).loc[train_ids]

    for variable in BP:
        check = baseline.xs(variable, level="Parameter").loc[train_ids]
        require(
            np.array_equal(
                check["count"].to_numpy(),
                saved_patterns[f"{variable}_count"].to_numpy(),
            ),
            f"{variable}: raw count does not match saved features.",
        )
        for statistic in ["mean", "min", "max"]:
            require(
                np.allclose(
                    check[statistic].to_numpy(),
                    saved_values[f"{variable}_{statistic}"].to_numpy(),
                    equal_nan=True,
                ),
                f"{variable}: raw {statistic} does not match saved features.",
            )

    # Zero-value and distribution summaries.
    variable_rows = []
    for variable in BP:
        subset = records.loc[records["Parameter"].eq(variable)]
        valid = subset.loc[subset["baseline_valid"]]
        zero = valid.loc[valid["numeric_value"].eq(0)]
        positive = valid.loc[valid["numeric_value"].gt(0)]
        counts = baseline.xs(variable, level="Parameter")["count"]

        variable_rows.append({
            "variable": variable,
            "n_raw_rows": len(subset),
            "n_minus1": int(subset["numeric_value"].eq(-1).sum()),
            "n_baseline_valid_rows": len(valid),
            "n_zero_rows": len(zero),
            "n_stays_with_zero": zero["RecordID"].nunique(),
            "n_stays_without_baseline_valid_record": int(counts.eq(0).sum()),
            "positive_min": positive["numeric_value"].min(),
            "positive_median": positive["numeric_value"].median(),
            "positive_max": positive["numeric_value"].max(),
        })

    variable_summary = pd.DataFrame(variable_rows)
    variable_summary.to_csv(
        OUTPUT / "01_variable_summary.csv", index=False
    )

    # Same-time context. Count original rows, not unique numeric values:
    # even identical repeated values are not assumed to be uniquely paired.
    context_rows = []
    double_zero_keys = set()

    valid_records = records.loc[records["baseline_valid"]]

    for (record_id, minute), group in valid_records.groupby(
        ["RecordID", "time_minutes"], sort=True
    ):
        for family, (mean_name, sys_name, dia_name) in FAMILIES.items():
            arrays = {
                name: group.loc[
                    group["Parameter"].eq(name), "numeric_value"
                ].tolist()
                for name in [mean_name, sys_name, dia_name]
            }

            if not any(arrays.values()):
                continue

            m = arrays[mean_name]
            s = arrays[sys_name]
            d = arrays[dia_name]

            ambiguous = any(len(x) > 1 for x in [m, s, d])
            complete_unique = all(len(x) == 1 for x in [m, s, d])

            # MAP can have multiple rows, but supporting S/D must each
            # have exactly one valid row and both must be zero.
            double_zero = (
                len(s) == 1 and len(d) == 1
                and s[0] == 0 and d[0] == 0
            )
            if double_zero:
                double_zero_keys.add((record_id, minute, mean_name))

            # Review flags only; no automatic exclusion from these flags.
            reversed_sd = (
                len(s) == 1 and len(d) == 1
                and s[0] > 0 and d[0] > 0
                and s[0] < d[0]
            )
            mean_outside_sd = (
                complete_unique
                and all(x[0] > 0 for x in [m, s, d])
                and not (d[0] <= m[0] <= s[0])
            )

            context_rows.append({
                "RecordID": record_id,
                "time_minutes": minute,
                "Time": f"{minute // 60:02d}:{minute % 60:02d}",
                "family": family,
                "mean_variable": mean_name,
                "mean_values": " | ".join(f"{x:g}" for x in m),
                "systolic_values": " | ".join(f"{x:g}" for x in s),
                "diastolic_values": " | ".join(f"{x:g}" for x in d),
                "n_mean_rows": len(m),
                "n_systolic_rows": len(s),
                "n_diastolic_rows": len(d),
                "ambiguous_multiple_rows": ambiguous,
                "complete_unique_triplet": complete_unique,
                "double_zero_sd_with_mean": double_zero and len(m) > 0,
                "positive_systolic_below_diastolic": reversed_sd,
                "positive_mean_outside_sd": mean_outside_sd,
            })

    context = pd.DataFrame(context_rows)
    context.to_csv(OUTPUT / "02_same_time_context.csv", index=False)

    flag_columns = [
        "ambiguous_multiple_rows",
        "complete_unique_triplet",
        "double_zero_sd_with_mean",
        "positive_systolic_below_diastolic",
        "positive_mean_outside_sd",
    ]
    context_summary = (
        context.groupby("family")[flag_columns].sum().astype(int)
    )
    context_summary.to_csv(OUTPUT / "03_same_time_summary.csv")

    context.loc[
        context[
            [
                "ambiguous_multiple_rows",
                "double_zero_sd_with_mean",
                "positive_systolic_below_diastolic",
                "positive_mean_outside_sd",
            ]
        ].any(axis=1)
    ].to_csv(OUTPUT / "04_flagged_time_groups.csv", index=False)

    # Three hypothetical exclusion masks, applied to baseline-valid rows only.
    zero_mask = (
        records["baseline_valid"] & records["numeric_value"].eq(0)
    )
    context_mask = pd.Series(
        [
            (record_id, minute, variable) in double_zero_keys
            for record_id, minute, variable in zip(
                records["RecordID"],
                records["time_minutes"],
                records["Parameter"],
            )
        ],
        index=records.index,
    ) & records["baseline_valid"]

    range_mask = (
        records["baseline_valid"]
        & records["Parameter"].isin(["MAP", "NIMAP"])
        & (
            records["numeric_value"].lt(10)
            | records["numeric_value"].gt(200)
        )
    )

    masks = {
        RULES[0]: zero_mask,
        RULES[1]: zero_mask | context_mask,
        RULES[2]: zero_mask | context_mask | range_mask,
    }

    impact_rows = []
    overall_rows = []
    comparison_frames = []
    flagged_records = records.copy()

    for name, mask in masks.items():
        flagged_records[name] = mask

        retained = records.copy()
        retained["baseline_valid"] = records["baseline_valid"] & ~mask
        after = summarize_records(retained, train_ids)

        overall_rows.append({
            "candidate": name,
            "n_removed_rows": int(mask.sum()),
            "n_affected_stays": records.loc[mask, "RecordID"].nunique(),
            "n_stays_losing_all_bp_records": int(
                (
                    baseline["count"].groupby(level="RecordID").sum().gt(0)
                    & after["count"].groupby(level="RecordID").sum().eq(0)
                ).sum()
            ),
        })

        for variable in BP:
            before_v = baseline.xs(variable, level="Parameter")
            after_v = after.xs(variable, level="Parameter")
            removed_v = mask & records["Parameter"].eq(variable)

            impact_rows.append({
                "candidate": name,
                "variable": variable,
                "n_removed_rows": int(removed_v.sum()),
                "n_affected_stays": records.loc[
                    removed_v, "RecordID"
                ].nunique(),
                "n_missing_before": int(before_v["count"].eq(0).sum()),
                "n_missing_after": int(after_v["count"].eq(0).sum()),
                "n_newly_missing": int(
                    (
                        before_v["count"].gt(0)
                        & after_v["count"].eq(0)
                    ).sum()
                ),
            })

        map_comparison = baseline.xs("MAP", level="Parameter").add_prefix(
            "before_"
        ).join(
            after.xs("MAP", level="Parameter").add_prefix("after_")
        )
        map_comparison["candidate"] = name
        map_comparison["mean_change"] = (
            map_comparison["after_mean"] - map_comparison["before_mean"]
        )
        comparison_frames.append(map_comparison.reset_index())

    overall_impact = pd.DataFrame(overall_rows)
    impact = pd.DataFrame(impact_rows)
    comparisons = pd.concat(comparison_frames, ignore_index=True)

    overall_impact.to_csv(
        OUTPUT / "05_candidate_overall_impact.csv", index=False
    )
    impact.to_csv(OUTPUT / "06_candidate_variable_impact.csv", index=False)
    comparisons.to_csv(
        OUTPUT / "07_candidate_map_comparison.csv", index=False
    )
    flagged_records.loc[
        flagged_records[RULES].any(axis=1)
    ].to_csv(OUTPUT / "08_candidate_flagged_records.csv", index=False)

    # Low and high MAP mean review samples.
    map_baseline = baseline.xs("MAP", level="Parameter")
    observed_map = map_baseline.dropna(subset=["mean"])
    lowest = observed_map.nsmallest(10, "mean")
    highest = observed_map.nlargest(10, "mean")
    lowest.to_csv(OUTPUT / "09_lowest10_map_mean.csv")

    records.loc[records["RecordID"].isin(lowest.index)].to_csv(
        OUTPUT / "10_lowest10_full_bp_records.csv", index=False
    )
    context.loc[context["RecordID"].isin(lowest.index)].to_csv(
        OUTPUT / "11_lowest10_same_time_context.csv", index=False
    )

    definitions = {
        "population": "3200 training stays only",
        "baseline": "Finite nonnegative BP records; -1 excluded",
        "A_zero_only": "Exclude zero values in all six BP variables",
        "B_zero_and_double_zero_context": (
            "A plus exclude MAP/NIMAP when corresponding same-time "
            "systolic and diastolic each have exactly one valid row, both zero"
        ),
        "C_B_and_mean_pressure_screen": (
            "B plus exclude MAP/NIMAP <10 or >200; "
            "screening scenario only, not a validated clinical range"
        ),
        "same_time_flags": (
            "Diagnostic only; timestamps do not establish sensor identity. "
            "No automatic deletion based on pressure ordering."
        ),
        "files_modified": "Audit outputs only",
    }
    (OUTPUT / "candidate_definitions.json").write_text(
        json.dumps(definitions, indent=2) + "\n"
    )

    print("\nBlood-pressure variable summary:")
    print(variable_summary.to_string(index=False))

    print("\nSame-time group summary (counts of time-family groups):")
    print(context_summary.to_string())

    print("\nLowest 10 training MAP means:")
    print(lowest.to_string())

    print("\nCandidate overall impact:")
    print(overall_impact.to_string(index=False))

    print("\nCandidate impact by variable:")
    print(impact.to_string(index=False))

    review_ids = set(lowest.index) | set(highest.index)
    changed = comparisons.loc[
        comparisons["RecordID"].isin(review_ids)
        & comparisons["before_count"].ne(comparisons["after_count"])
    ]
    print("\nMAP changes among the low/high review stays:")
    print(
        changed[
            [
                "RecordID", "candidate",
                "before_count", "after_count",
                "before_mean", "after_mean",
            ]
        ].to_string(index=False)
    )

    print(f"\nAudit files saved to: {OUTPUT}")
    print("No outcomes were read.")
    print("No raw data, features, or split assignments were modified.")


if __name__ == "__main__":
    main()