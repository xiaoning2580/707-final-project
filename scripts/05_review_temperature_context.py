"""Review unusual temperature records; do not modify raw data."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "set-a"
OUTPUT = ROOT / "data" / "audit"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = []

    files = sorted(RECORDS.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No measurement files found in {RECORDS}")

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

        parts = temp["Time"].str.split(":", expand=True)
        temp["time_minutes"] = (
            pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])
        )
        frames.append(temp)

    if not frames:
        raise ValueError("No temperature records found.")

    temp = pd.concat(frames, ignore_index=True)

    # Review threshold only; this does not define a cleaning rule.
    flagged = temp.loc[
        temp["temperature"].gt(0) & temp["temperature"].lt(25)
    ].copy()

    flagged_ids = set(flagged["RecordID"])

    context_ids = flagged_ids | {"136753"}
    context = temp.loc[temp["RecordID"].isin(context_ids)].copy()
    context = context.sort_values(
        ["RecordID", "time_minutes", "source_line"]
    )

    summary = (
        temp.loc[temp["RecordID"].isin(flagged_ids)]
        .groupby("RecordID", as_index=False)
        .agg(
            n_temp_rows=("temperature", "size"),
            n_nonpositive=("temperature", lambda x: int(x.le(0).sum())),
            n_positive_below_25=(
                "temperature",
                lambda x: int((x.gt(0) & x.lt(25)).sum()),
            ),
            n_at_least_25=("temperature", lambda x: int(x.ge(25).sum())),
            minimum=("temperature", "min"),
            median=("temperature", "median"),
            maximum=("temperature", "max"),
        )
    )

    flagged.to_csv(
        OUTPUT / "temperature_positive_below25.csv", index=False
    )
    summary.to_csv(
        OUTPUT / "temperature_below25_stay_summary.csv", index=False
    )
    context.to_csv(
        OUTPUT / "temperature_review_full_sequences.csv", index=False
    )

    columns = ["RecordID", "Time", "source_line", "temperature"]

    print(f"\nPositive temperature records below 25: {len(flagged)}")
    print(f"Affected stays: {len(flagged_ids)}")

    print("\nAffected-stay summary:")
    print(summary.to_string(index=False))

    print("\nAll positive temperature records below 25:")
    print(
        flagged.sort_values(
            ["RecordID", "time_minutes", "source_line"]
        )[columns].to_string(index=False)
    )

    # Print the complete sequences for the three specific examples.
    for record_id in ["139761", "141449", "136753"]:
        print(f"\nFull temperature sequence: {record_id}")
        print(
            context.loc[
                context["RecordID"].eq(record_id), columns
            ].to_string(index=False)
        )

    print(f"\nSaved results to: {OUTPUT}")
    print("Review only: no records were changed or removed.")


if __name__ == "__main__":
    main()