"""Tune three random forests on shared training folds and export test probabilities."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "results/tables/random_forests"
FEATURE_SETS = {"values": 111, "patterns": 74, "combined": 185}
GRID = {
    "model__max_depth": [8, None],
    "model__min_samples_leaf": [10, 2],
    "model__max_features": ["sqrt", 0.5],
}


def environment_versions():
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
                "joblib", "threadpoolctl"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in packages}}


def read_table(name):
    return pd.read_csv(PROCESSED / name, dtype={"RecordID": str}).set_index("RecordID")


def input_hashes():
    names = ["split_assignments.csv", "outcomes.csv", "feature_sets.json",
             "preparation_manifest.json"]
    names += [f"features_{name}.csv" for name in FEATURE_SETS]
    return {name: hashlib.sha256((PROCESSED / name).read_bytes()).hexdigest()
            for name in names}


def load_inputs():
    splits = read_table("split_assignments.csv").sort_index()
    outcomes = read_table("outcomes.csv")
    assert splits.index.is_unique and len(splits) == 4000
    assert outcomes.index.is_unique and set(outcomes.index) == set(splits.index)
    assert outcomes["In-hospital_death"].isin([0, 1]).all()
    assert set(splits["split"]) == {"train", "test"}
    train_ids = splits.index[splits["split"].eq("train")]
    test_ids = splits.index[splits["split"].eq("test")]
    assert len(train_ids) == 3200 and len(test_ids) == 800
    assert splits.loc[test_ids, "cv_fold"].isna().all()
    folds = splits.loc[train_ids, "cv_fold"].astype(int)
    assert folds.value_counts().sort_index().to_dict() == dict.fromkeys(range(1, 6), 640)
    y_train = outcomes.loc[train_ids, "In-hospital_death"]
    for fold in range(1, 6):
        assert y_train.loc[folds.eq(fold)].nunique() == 2

    manifest = json.loads((PROCESSED / "preparation_manifest.json").read_text())
    assert manifest["blood_pressure_rule_version"] == "BP_RULE_B_V1"
    schema = json.loads((PROCESSED / "feature_sets.json").read_text())
    features = {}
    for name, width in FEATURE_SETS.items():
        frame = read_table(f"features_{name}.csv")
        assert frame.index.is_unique and set(frame.index) == set(splits.index)
        assert list(frame.columns) == schema[name] and len(frame.columns) == width
        assert not set(frame.columns) & {"RecordID", "split", "cv_fold", "In-hospital_death"}
        assert not np.isinf(frame.to_numpy(dtype=float)).any()
        features[name] = frame
    assert features["patterns"].notna().all().all()
    assert (features["patterns"].filter(like="_count") >= 0).all().all()
    assert features["patterns"].filter(like="_missing").isin([0, 1]).all().all()
    pd.testing.assert_frame_equal(
        features["combined"], pd.concat([features["values"], features["patterns"]], axis=1)
    )
    cv = PredefinedSplit(folds.to_numpy() - 1)  # (3200,): project folds 1-5 become 0-4.
    for fold, (fit_rows, validation_rows) in enumerate(cv.split(), start=1):
        assert set(train_ids[validation_rows]) == set(train_ids[folds.eq(fold)])
        assert set(train_ids[fit_rows]).isdisjoint(train_ids[validation_rows])
    print("Inputs checked: 3200 training stays, 800 test stays, five shared folds.", flush=True)
    return features, train_ids, test_ids, y_train, cv


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False,
                                  keep_empty_features=True)),
        ("constant_filter", VarianceThreshold()),
        ("model", RandomForestClassifier(n_estimators=300, random_state=707,
                                          class_weight=None, n_jobs=1)),
    ])


def tune(features, train_ids, y_train, cv, jobs):
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = {}
    tables = []
    for name in FEATURE_SETS:
        print(f"Tuning {name}: 8 parameter combinations x 5 folds.", flush=True)
        search = GridSearchCV(make_pipeline(), GRID, scoring="roc_auc", cv=cv,
                              n_jobs=jobs, refit=False, error_score="raise")
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            search.fit(features[name].loc[train_ids], y_train)
        result = search.cv_results_
        best = int(np.argmax(result["mean_test_score"]))
        selected[name] = result["params"][best]
        rows = []
        for candidate, params in enumerate(result["params"]):
            row = {"feature_set": name, "candidate": candidate + 1,
                   **{key.removeprefix("model__"): value for key, value in params.items()}}
            row.update({f"fold_{fold + 1}_auroc": result[f"split{fold}_test_score"][candidate]
                        for fold in range(5)})
            row["mean_auroc"] = result["mean_test_score"][candidate]
            row["std_auroc"] = result["std_test_score"][candidate]
            row["selected"] = int(candidate == best)
            rows.append(row)
        table = pd.DataFrame(rows)
        assert np.isfinite(table.filter(like="auroc").to_numpy()).all()
        table.to_csv(RESULTS / f"rf_{name}_cv_results.csv", index=False)
        tables.append(table)
        print(f"Selected {name}: {selected[name]}, mean CV AUROC={row_score(result, best):.6f}", flush=True)
    pd.concat(tables, ignore_index=True).to_csv(RESULTS / "rf_cv_results.csv", index=False)
    metadata = {
        "selected_parameters": selected,
        "fixed_parameters": {"n_estimators": 300, "random_state": 707, "class_weight": None},
        "selection": "Highest mean shared-five-fold AUROC; first grid candidate in an exact tie.",
        "grid": GRID,
        "input_sha256": input_hashes(),
        "environment": environment_versions(),
    }
    (RESULTS / "rf_selection.json").write_text(json.dumps(metadata, indent=2) + "\n")


def row_score(results, index):
    return float(results["mean_test_score"][index])


def predict_test(features, train_ids, test_ids, y_train, jobs):
    metadata = json.loads((RESULTS / "rf_selection.json").read_text())
    assert metadata["input_sha256"] == input_hashes(), "Inputs changed after tuning."
    current = environment_versions()
    assert all(current[name] == value for name, value in metadata["environment"].items()
               if name != "platform"), "Modeling environment changed after tuning."
    output = ROOT / "data/predictions"
    model_dir = ROOT / "data/models"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for name in FEATURE_SETS:
        pipeline = make_pipeline().set_params(**metadata["selected_parameters"][name])
        pipeline.set_params(model__n_jobs=jobs)
        pipeline.fit(features[name].loc[train_ids], y_train)
        positive = list(pipeline.classes_).index(1)
        probability = pipeline.predict_proba(features[name].loc[test_ids])[:, positive]  # (800, 2) -> (800,).
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        path = output / f"rf_{name}.csv"
        pd.DataFrame({"RecordID": test_ids, "predicted_probability": probability}).to_csv(path, index=False)
        saved = pd.read_csv(path, dtype={"RecordID": str})
        assert list(saved.columns) == ["RecordID", "predicted_probability"]
        assert len(saved) == 800 and saved["RecordID"].is_unique
        assert saved["RecordID"].tolist() == test_ids.tolist()
        np.testing.assert_allclose(saved["predicted_probability"], probability, rtol=1e-12, atol=1e-12)
        joblib.dump(pipeline, model_dir / f"rf_{name}.joblib")
        checks.append({"model": f"rf_{name}", "n_train": len(train_ids), "n_test": len(saved),
                       "n_input_features": features[name].shape[1],
                       "n_model_features": pipeline.named_steps["model"].n_features_in_,
                       "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"Saved {path.name}: 800 aligned test probabilities.", flush=True)
    pd.DataFrame(checks).to_csv(RESULTS / "rf_prediction_checks.csv", index=False)
    print("No test performance was calculated; predictions are ready for group evaluation.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "tune", "predict", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(environment_versions(), indent=2), flush=True)
    features, train_ids, test_ids, y_train, cv = load_inputs()
    if args.mode in ("tune", "all"):
        tune(features, train_ids, y_train, cv, args.jobs)
    if args.mode in ("predict", "all"):
        predict_test(features, train_ids, test_ids, y_train, args.jobs)


if __name__ == "__main__":
    main()
