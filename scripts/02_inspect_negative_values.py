"""Inspect negative measurement values other than the -1 missing code."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "set-a"
OUTPUT = ROOT / "data" / "audit"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    files = sorted(RECORDS.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No measurement files found in {RECORDS}")

    negative_rows = []
    context_rows = []

    for path in files:
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()

        for column in ["Time", "Parameter", "Value"]:
            df[column] = df[column].str.strip()

        values = pd.to_numeric(df["Value"], errors="raise")
        negative_mask = (
        df["Parameter"].eq("Temp")
        & values.le(0)
        & values.ne(-1)
        )

        if not negative_mask.any():
            continue

        # Header is line 1, so the first data row is line 2.
        df.insert(0, "source_line", range(2, len(df) + 2))
        df.insert(0, "RecordID", path.stem)
        df["numeric_value"] = values

        negative_rows.append(df.loc[negative_mask].copy())

        # Retain all observations of each affected variable in this stay.
        # This provides context without relying on unrelated adjacent rows.
        affected_variables = df.loc[negative_mask, "Parameter"].unique()
        context = df.loc[
            df["Parameter"].isin(affected_variables)
        ].copy()

        context["is_other_negative"] = (
            context["numeric_value"].lt(0)
            & context["numeric_value"].ne(-1)
        )

        time_parts = context["Time"].str.split(":", expand=True)
        context["time_minutes"] = (
            pd.to_numeric(time_parts[0]) * 60
            + pd.to_numeric(time_parts[1])
        )
        context_rows.append(context)

    if not negative_rows:
        print("No negative values other than -1 were found.")
        return

    negatives = pd.concat(negative_rows, ignore_index=True)
    negatives = negatives.sort_values(
        ["Parameter", "RecordID", "source_line"]
    )

    summary = (
        negatives.groupby("Parameter", as_index=False)
        .agg(
            n_negative_rows=("numeric_value", "size"),
            n_affected_stays=("RecordID", "nunique"),
            minimum_negative_value=("numeric_value", "min"),
            maximum_negative_value=("numeric_value", "max"),
        )
    )

    contexts = pd.concat(context_rows, ignore_index=True)
    contexts = contexts.sort_values(
        ["RecordID", "Parameter", "time_minutes", "source_line"]
    )

    negatives.to_csv(
        OUTPUT / "other_negative_values.csv", index=False
    )
    summary.to_csv(
        OUTPUT / "other_negative_summary.csv", index=False
    )
    contexts.to_csv(
        OUTPUT / "other_negative_context.csv", index=False
    )

    print(f"\nTotal non-minus-one negative rows: {len(negatives)}")
    print(f"Affected ICU stays: {negatives['RecordID'].nunique()}")

    print("\nSummary by variable:")
    print(summary.to_string(index=False))

    print("\nAll non-minus-one negative records:")
    print(
        negatives[
            ["RecordID", "source_line", "Time", "Parameter", "Value"]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT}")
    print("Original data have not been modified.")


if __name__ == "__main__":
    main()