"""Audit temperature records without modifying the original data."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "set-a"
OUTPUT = ROOT / "data" / "audit"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    files = sorted(RECORDS.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No measurement files found in {RECORDS}")

    frames = []
    all_ids = [path.stem for path in files]

    for path in files:
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()

        for column in ["Time", "Parameter", "Value"]:
            df[column] = df[column].str.strip()

        df["source_line"] = range(2, len(df) + 2)

        temp = df.loc[df["Parameter"].eq("Temp")].copy()
        if temp.empty:
            continue

        temp.insert(0, "RecordID", path.stem)
        temp["temperature"] = pd.to_numeric(temp["Value"], errors="raise")
        frames.append(temp)

    if not frames:
        raise ValueError("No temperature records found.")

    temp = pd.concat(frames, ignore_index=True)

    # These categories describe the data; positive values still need review.
    temp["status"] = "positive"
    temp.loc[temp["temperature"].lt(0), "status"] = "negative"
    temp.loc[temp["temperature"].eq(-1), "status"] = "missing_minus1"
    temp.loc[temp["temperature"].eq(0), "status"] = "zero"

    positive = temp.loc[temp["temperature"].gt(0)].copy()
    nonpositive = temp.loc[temp["temperature"].le(0)].copy()

    per_stay = (
        temp.groupby("RecordID")
        .agg(
            n_temp_rows=("temperature", "size"),
            n_positive=("temperature", lambda x: int(x.gt(0).sum())),
            n_zero=("temperature", lambda x: int(x.eq(0).sum())),
            n_negative=("temperature", lambda x: int(x.lt(0).sum())),
        )
        .reindex(all_ids, fill_value=0)
        .rename_axis("RecordID")
        .reset_index()
    )

    per_stay["no_original_temp"] = per_stay["n_temp_rows"].eq(0)
    per_stay["no_positive_temp"] = per_stay["n_positive"].eq(0)

    newly_missing = per_stay.loc[
        per_stay["n_temp_rows"].gt(0)
        & per_stay["n_positive"].eq(0)
    ]

    by_status = (
        temp.groupby("status", as_index=False)
        .agg(
            n_rows=("temperature", "size"),
            n_stays=("RecordID", "nunique"),
        )
    )

    quantiles = (
        positive["temperature"]
        .quantile([0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1])
        .rename_axis("quantile")
        .reset_index(name="temperature")
    )

    # Display extremes for review, without defining exclusion thresholds.
    lowest = positive.nsmallest(20, "temperature")
    highest = positive.nlargest(20, "temperature")

    summary = {
        "n_stays": len(files),
        "n_temperature_rows": len(temp),
        "n_stays_with_temperature": int(temp["RecordID"].nunique()),
        "n_negative_rows": int(temp["temperature"].lt(0).sum()),
        "n_zero_rows": int(temp["temperature"].eq(0).sum()),
        "n_stays_with_zero": int(
            temp.loc[temp["temperature"].eq(0), "RecordID"].nunique()
        ),
        "n_positive_rows": len(positive),
        "n_stays_without_original_temperature": int(
            per_stay["no_original_temp"].sum()
        ),
        "n_stays_without_positive_temperature": int(
            per_stay["no_positive_temp"].sum()
        ),
        "n_stays_becoming_missing_after_nonpositive_removal": len(
            newly_missing
        ),
    }

    nonpositive.to_csv(
        OUTPUT / "temperature_nonpositive_records.csv", index=False
    )
    per_stay.to_csv(
        OUTPUT / "temperature_per_stay.csv", index=False
    )
    newly_missing.to_csv(
        OUTPUT / "temperature_newly_missing_stays.csv", index=False
    )
    quantiles.to_csv(
        OUTPUT / "temperature_positive_quantiles.csv", index=False
    )

    extremes = pd.concat(
        [
            lowest.assign(extreme_type="lowest"),
            highest.assign(extreme_type="highest"),
        ],
        ignore_index=True,
    )
    extremes.to_csv(
        OUTPUT / "temperature_positive_extremes.csv", index=False
    )

    (OUTPUT / "temperature_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("\nTemperature summary:")
    print(json.dumps(summary, indent=2))

    print("\nCounts by status:")
    print(by_status.to_string(index=False))

    print("\nPositive-temperature quantiles (record-level):")
    print(quantiles.to_string(index=False))

    columns = ["RecordID", "Time", "source_line", "temperature"]

    print("\nLowest 20 positive temperature records:")
    print(lowest[columns].to_string(index=False))

    print("\nHighest 20 positive temperature records:")
    print(highest[columns].to_string(index=False))

    print(f"\nResults saved to: {OUTPUT}")
    print("Original files have not been modified.")


if __name__ == "__main__":
    main()