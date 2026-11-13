"""Inspect the upper tail of training-set MAP means.

Run:
    python scripts/08_inspect_map_tail.py

Does not read outcomes or modify raw/processed data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "audit" / "map_review"

BP_VARIABLES = [
    "MAP", "SysABP", "DiasABP",
    "NIMAP", "NISysABP", "NIDiasABP",
]


def read_indexed(path):
    df = pd.read_csv(path, dtype={"RecordID": str})
    if df["RecordID"].isna().any() or df["RecordID"].duplicated().any():
        raise ValueError(f"Missing or duplicate IDs: {path}")
    return df.set_index("RecordID")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    splits = read_indexed(DATA / "processed" / "split_assignments.csv")
    features = read_indexed(DATA / "processed" / "features_values.csv")
    patterns = read_indexed(DATA / "processed" / "features_patterns.csv")

    if not (
        set(splits.index) == set(features.index) == set(patterns.index)
    ):
        raise ValueError("IDs differ between feature and split files.")

    train_ids = splits.index[splits["split"].eq("train")]
    if len(train_ids) != 3200:
        raise ValueError("Expected 3200 training stays.")

    training = features.loc[
        train_ids, ["MAP_mean", "MAP_min", "MAP_max"]
    ].copy()
    training["MAP_count"] = patterns.loc[train_ids, "MAP_count"]

    quantiles = (
        training["MAP_mean"]
        .quantile([0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        .rename_axis("quantile")
        .reset_index(name="MAP_mean")
    )
    quantiles.to_csv(OUTPUT / "map_mean_quantiles.csv", index=False)

    # Fixed-size review sample, not a clinical exclusion threshold.
    top = (
        training.dropna(subset=["MAP_mean"])
        .sort_values("MAP_mean", ascending=False)
        .head(10)
    )
    top.to_csv(OUTPUT / "top10_map_mean_stays.csv")

    raw_frames = []
    summaries = []
    same_time_frames = []

    for record_id in top.index:
        path = DATA / "set-a" / f"{record_id}.txt"
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()

        for column in ["Time", "Parameter", "Value"]:
            df[column] = df[column].str.strip()

        df["source_line"] = range(2, len(df) + 2)
        df["numeric_value"] = pd.to_numeric(df["Value"], errors="raise")

        parts = df["Time"].str.extract(r"^(\d+):([0-5]\d)$")
        if parts.isna().any().any():
            raise ValueError(f"{record_id}: invalid time.")

        df["time_minutes"] = (
            parts[0].astype(int) * 60 + parts[1].astype(int)
        )

        bp = df.loc[
            df["Parameter"].isin(BP_VARIABLES)
            & df["time_minutes"].between(0, 2880)
        ].copy()

        bp.insert(0, "RecordID", record_id)

        if not np.isfinite(bp["numeric_value"]).all():
            raise ValueError(f"{record_id}: nonfinite blood-pressure value.")

        unexpected_negative = (
            bp["numeric_value"].lt(0)
            & bp["numeric_value"].ne(-1)
        )
        if unexpected_negative.any():
            raise ValueError(f"{record_id}: unexpected negative BP value.")

        bp["clean_value"] = bp["numeric_value"].mask(
            bp["numeric_value"].eq(-1)
        )
        bp = bp.sort_values(["time_minutes", "Parameter", "source_line"])
        raw_frames.append(bp)

        map_rows = bp.loc[
            bp["Parameter"].eq("MAP") & bp["clean_value"].notna()
        ]
        map_values = map_rows["clean_value"]

        # Verify the feature reflects the source records.
        calculated = np.array([
            map_values.mean(), map_values.min(), map_values.max()
        ])
        saved = top.loc[
            record_id, ["MAP_mean", "MAP_min", "MAP_max"]
        ].to_numpy(dtype=float)

        if not np.allclose(calculated, saved):
            raise ValueError(f"{record_id}: MAP feature/raw mismatch.")

        if len(map_values) != int(top.loc[record_id, "MAP_count"]):
            raise ValueError(f"{record_id}: MAP count/raw mismatch.")

        for variable in BP_VARIABLES:
            observed = bp.loc[
                bp["Parameter"].eq(variable), "clean_value"
            ].dropna()

            summaries.append({
                "RecordID": record_id,
                "variable": variable,
                "count": len(observed),
                "minimum": observed.min(),
                "median": observed.median(),
                "mean": observed.mean(),
                "maximum": observed.max(),
            })

        # Preserve all values at each timestamp as text.
        # Do not average duplicates or assume different sensors are paired.
        valid_bp = bp.loc[bp["clean_value"].notna()]
        same_time = (
            valid_bp.groupby(["time_minutes", "Time", "Parameter"])["clean_value"]
            .agg(lambda x: " | ".join(f"{v:g}" for v in x))
            .unstack("Parameter")
            .reindex(columns=BP_VARIABLES)
            .reset_index()
        )
        same_time.insert(0, "RecordID", record_id)
        same_time_frames.append(same_time)

    summaries = pd.DataFrame(summaries)
    raw = pd.concat(raw_frames, ignore_index=True)
    same_time = pd.concat(same_time_frames, ignore_index=True)

    summaries.to_csv(OUTPUT / "blood_pressure_summary.csv", index=False)
    raw.to_csv(OUTPUT / "blood_pressure_raw_sequences.csv", index=False)
    same_time.to_csv(
        OUTPUT / "blood_pressure_same_time_context.csv", index=False
    )

    print("\nTraining-set MAP mean quantiles:")
    print(quantiles.to_string(index=False))

    print("\nTen training stays with the highest MAP mean:")
    print(top.to_string())

    print("\nBlood-pressure summaries for the top three stays:")
    print(
        summaries.loc[
            summaries["RecordID"].isin(top.index[:3])
        ].to_string(index=False)
    )

    print("\nHighest 20 individual MAP records among the reviewed stays:")
    largest = (
        raw.loc[
            raw["Parameter"].eq("MAP") & raw["clean_value"].notna()
        ]
        .nlargest(20, "clean_value")
    )
    print(
        largest[
            ["RecordID", "Time", "source_line", "clean_value"]
        ].to_string(index=False)
    )

    print(f"\nFiles saved to: {OUTPUT}")
    print("Raw-to-feature checks passed for all reviewed stays.")
    print("No outcomes were read. No records or features were modified.")


if __name__ == "__main__":
    main()