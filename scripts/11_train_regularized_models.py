"""Train regularized in-hospital mortality models.

Inputs are COMMON_COVARIATES_V2 tables in data/processed/common_covariates_v2/.
The feature-set widths below are column counts before categorical encoding.

Run from the repository root:

    python scripts/11_train_regularized_models.py --tune-lasso
    python scripts/11_train_regularized_models.py --predict-test

--tune-lasso selects one C for each LASSO feature set on the shared
training folds and writes the tuning table. It does not refit on all
training rows. --predict-test reads those selected penalties, refits the
age baseline and three LASSO models on all training stays, and writes
test-set probabilities. With no arguments, the script prints this usage.
"""

from pathlib import Path
import argparse
import json
import subprocess
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "COMMON_COVARIATES_V2"
PROCESSED = ROOT / "data" / "processed" / "common_covariates_v2"
LASSO_TUNING_TABLE = (
    ROOT
    / "results"
    / "tables"
    / "regularized_models"
    / "common_covariates_v2"
    / "lasso_cv_results.csv"
)
PREDICTION_DIR = ROOT / "data" / "predictions" / "common_covariates_v2"
PREDICTION_FILES = {
    "age_only": PREDICTION_DIR / "age_only.csv",
    "values": PREDICTION_DIR / "lasso_values.csv",
    "patterns": PREDICTION_DIR / "lasso_patterns.csv",
    "combined": PREDICTION_DIR / "lasso_combined.csv",
}

AGE_PREDICTORS = ["Age"]
OUTCOME = "In-hospital_death"
POSITIVE_CLASS = 1
LASSO_SOLVER = "saga"
LASSO_RANDOM_STATE = 707
LASSO_TUNE_C = [0.001, 0.01, 0.1, 1.0, 10.0]
# C=10 did not converge at max_iter=5000, so tuning and final fits use 20000.
LASSO_TUNE_MAX_ITER = 20000
LASSO_FEATURE_SETS = ("values", "patterns", "combined")
FEATURE_SET_FILES = {
    "values": "features_values.csv",
    "patterns": "features_patterns.csv",
    "combined": "features_combined.csv",
}
COMMON_COVARIATES = ["Age", "Gender", "ICUType"]
# Predictor counts before one-hot encoding. RecordID is not included.
FEATURE_SET_WIDTHS = {"values": 114, "patterns": 77, "combined": 188}
N_MEASUREMENT_VALUES = 111
N_MEASUREMENT_PATTERNS = 74
EXCLUDED_COLUMNS = {
    "RecordID",
    OUTCOME,
    "split",
    "cv_fold",
    "Height",
    "SAPS-I",
    "SOFA",
    "Length_of_stay",
    "Survival",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_indexed_table(path, required_columns):
    frame = pd.read_csv(path, dtype={"RecordID": str})
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    require(
        "RecordID" in frame.columns and not missing_columns,
        f"{path.name}: missing columns {['RecordID', *missing_columns]}.",
    )

    record_ids = frame["RecordID"]
    require(record_ids.notna().all(), f"{path.name}: missing RecordID.")
    record_ids = record_ids.str.strip()
    require(record_ids.ne("").all(), f"{path.name}: empty RecordID.")
    require(record_ids.is_unique, f"{path.name}: duplicate RecordID.")

    indexed = frame.copy()
    indexed["RecordID"] = record_ids
    return indexed.set_index("RecordID")


def require_same_record_ids(tables):
    reference_name, reference = next(iter(tables.items()))
    reference_ids = set(reference.index)
    for name, frame in tables.items():
        identifiers = set(frame.index)
        missing = len(reference_ids - identifiers)
        extra = len(identifiers - reference_ids)
        require(
            missing == 0 and extra == 0,
            f"{name} RecordID set differs from {reference_name}: "
            f"{missing} missing, {extra} extra.",
        )


def require_input_version():
    """Reject any feature directory other than COMMON_COVARIATES_V2."""
    feature_path = PROCESSED / "feature_sets.json"
    manifest_path = PROCESSED / "preparation_manifest.json"
    require(feature_path.is_file(), f"Missing feature metadata: {feature_path}.")
    require(manifest_path.is_file(), f"Missing preparation manifest: {manifest_path}.")
    feature_sets = json.loads(feature_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    found_schema = feature_sets.get("schema_version") if isinstance(feature_sets, dict) else None
    require(
        found_schema == FEATURE_VERSION,
        "feature_sets.json schema_version must be "
        f"{FEATURE_VERSION}, found {found_schema!r}.",
    )
    found_manifest = manifest.get("feature_schema_version")
    require(
        found_manifest == FEATURE_VERSION,
        "preparation_manifest.json feature_schema_version must be "
        f"{FEATURE_VERSION}, found {found_manifest!r}.",
    )
    return feature_sets


def load_aligned_inputs():
    """Align baseline, outcome, and split rows by RecordID without a join."""
    require_input_version()
    tables = {
        "outcomes": read_indexed_table(
            PROCESSED / "outcomes.csv", [OUTCOME]
        ),
        "baseline": read_indexed_table(
            PROCESSED / "baseline.csv", AGE_PREDICTORS
        ),
        "splits": read_indexed_table(
            PROCESSED / "split_assignments.csv", ["split", "cv_fold"]
        ),
    }
    require_same_record_ids(tables)

    ordered_ids = tables["outcomes"].index
    aligned = pd.DataFrame(
        {
            "Age": tables["baseline"].loc[ordered_ids, "Age"],
            OUTCOME: pd.to_numeric(
                tables["outcomes"][OUTCOME], errors="raise"
            ),
            "split": tables["splits"].loc[ordered_ids, "split"],
            "cv_fold": tables["splits"].loc[ordered_ids, "cv_fold"],
        },
        index=ordered_ids,
    )
    aligned.index.name = "RecordID"
    require(aligned.index.is_unique, "Aligned RecordID values are not unique.")
    require(
        len(aligned) == len(ordered_ids),
        "Alignment changed the number of stays.",
    )
    require(
        aligned[OUTCOME].isin([0, 1]).all(),
        "In-hospital_death must be 0 or 1.",
    )
    aligned[OUTCOME] = aligned[OUTCOME].astype(int)
    return aligned


def make_age_baseline():
    """Return an unpenalized age-only logistic pipeline.

    scikit-learn 1.9 deprecates penalty=None. Leaving penalty unset and
    setting C=np.inf removes the penalty term. l1_ratio stays at 0.
    Age stays in years because this pipeline does not standardize it.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    C=np.inf,
                    solver="lbfgs",
                    fit_intercept=True,
                ),
            ),
        ]
    )


def death_probabilities(pipeline, features):
    classes = list(pipeline.classes_)
    require(
        POSITIVE_CLASS in classes,
        f"Class {POSITIVE_CLASS} is absent from classes_: {classes}.",
    )
    column = classes.index(POSITIVE_CLASS)
    predicted = pipeline.predict_proba(features)
    require(
        predicted.shape == (len(features), len(classes)),
        "predict_proba shape does not match the fitted classes.",
    )
    return pd.Series(
        predicted[:, column],
        index=features.index,
        name="predicted_probability",
    )


def training_rows(frame):
    require(
        frame["split"].isin(["train", "test"]).all(),
        "Split labels must be train or test.",
    )
    training = frame.loc[frame["split"].eq("train")].copy()
    folds = pd.to_numeric(training["cv_fold"], errors="raise")
    require(folds.notna().all(), "Training cv_fold is missing.")
    integer_folds = folds.astype(int)
    require(
        folds.eq(integer_folds).all()
        and integer_folds.isin([1, 2, 3, 4, 5]).all(),
        "Training cv_fold must be an integer from 1 to 5.",
    )
    training["cv_fold"] = integer_folds
    return training


def category_codes(values):
    """Map category codes to stable strings and missing values to Missing."""

    def label(value):
        if pd.isna(value):
            return "Missing"
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            if text == "" or text.lower() in {"nan", "none", "<na>"}:
                return "Missing"
            return text
        if np.isfinite(number) and float(number).is_integer():
            return str(int(number))
        text = str(value).strip()
        return "Missing" if text == "" else text

    return pd.DataFrame(values).map(label).to_numpy(dtype=object)


def lasso_preprocessor(numeric_columns, categorical_columns):
    """Median-impute numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
                list(numeric_columns),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "codes",
                            FunctionTransformer(
                                category_codes,
                                feature_names_out="one-to-one",
                                validate=False,
                                check_inverse=False,
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                drop=None,
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_lasso_pipeline(feature_set, max_iter=LASSO_TUNE_MAX_ITER):
    """Return the L1 logistic pipeline for one feature set.

    Numeric columns, including Age and the measurement features, are
    median-imputed from the rows used to fit the pipeline. Gender and ICUType
    are string-coded, with missing values mapped to Missing, then one-hot
    encoded. After the branches are joined, constant columns are removed and
    every remaining column is standardized, including one-hot columns.

    scikit-learn 1.9 deprecates penalty='l1'. l1_ratio=1 is the pure L1
    penalty, with the penalty argument left unset.
    """
    columns = load_feature_spec()["sets"][feature_set]
    numeric_columns = columns["numeric_columns"]
    categorical_columns = columns["categorical_columns"]
    require(
        categorical_columns == ["Gender", "ICUType"],
        f"{feature_set} categorical columns are {categorical_columns}.",
    )
    require("Age" in numeric_columns, f"{feature_set} numeric columns do not include Age.")
    require(
        set(numeric_columns).isdisjoint(categorical_columns),
        f"{feature_set} sends a column through both preprocessing branches.",
    )
    return Pipeline(
        steps=[
            ("preprocess", lasso_preprocessor(numeric_columns, categorical_columns)),
            ("constant_filter", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver=LASSO_SOLVER,
                    fit_intercept=True,
                    random_state=LASSO_RANDOM_STATE,
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def predefined_training_split(training):
    """Map project folds 1-5 to PredefinedSplit test folds 0-4."""
    project_folds = training["cv_fold"].to_numpy(dtype=int)
    splitter = PredefinedSplit(project_folds - 1)
    require(
        list(splitter.unique_folds) == [0, 1, 2, 3, 4],
        f"Unexpected PredefinedSplit folds: {list(splitter.unique_folds)}.",
    )
    return splitter


def require_predefined_folds_match(features, training, splitter):
    """Compare validation IDs, not only counts, with split_assignments.csv."""
    record_ids = features.index.to_numpy()
    seen = []
    for project_fold, (fit_index, validation_index) in enumerate(splitter.split(), start=1):
        validation_ids = set(record_ids[validation_index])
        fit_ids = set(record_ids[fit_index])
        expected_ids = set(training.index[training["cv_fold"].eq(project_fold)])
        require(
            validation_ids == expected_ids,
            f"Project fold {project_fold} validation IDs differ from split_assignments.csv.",
        )
        require(fit_ids.isdisjoint(validation_ids), f"Fold {project_fold} reuses a validation ID for fitting.")
        require(
            fit_ids | validation_ids == set(training.index),
            f"Fold {project_fold} does not cover the training IDs.",
        )
        seen.extend(record_ids[validation_index])
    require(
        len(seen) == len(set(seen)) == len(training),
        "Training records do not each appear in exactly one validation fold.",
    )
    print("Training fold IDs match split_assignments.csv.")


def load_lasso_training_matrices():
    """Align all three feature sets to the same sorted training rows."""
    training, _test, train_features, _test_features = load_train_and_test()
    matrices = {
        feature_set: train_features[feature_set] for feature_set in LASSO_FEATURE_SETS
    }
    for feature_set, features in matrices.items():
        require(
            features.index.equals(matrices["values"].index),
            f"{feature_set} training row order differs from values.",
        )
    return matrices, training


def column_mismatches(left, right):
    """Count missingness and value disagreements, treating paired NA values as equal."""
    missing_disagreement = left.isna() ^ right.isna()
    value_disagreement = left.notna() & right.notna() & left.ne(right)
    return int(missing_disagreement.sum()), int(value_disagreement.sum())


def require_same_series(left, right, label):
    missing_disagreement, value_disagreement = column_mismatches(left, right)
    require(
        missing_disagreement == 0 and value_disagreement == 0,
        f"{label}: {missing_disagreement} missingness differences, "
        f"{value_disagreement} value differences.",
    )


def pattern_pairs(columns):
    pairs = []
    for column in columns:
        if column.endswith("_count"):
            variable = column[: -len("_count")]
            missing_name = f"{variable}_missing"
            require(
                missing_name in columns,
                f"features_patterns.csv column {column} has no {missing_name}.",
            )
            pairs.append((column, missing_name))
        else:
            require(
                column.endswith("_missing"),
                f"features_patterns.csv column {column} is not a count or missing indicator.",
            )
            count_name = f"{column[: -len('_missing')]}_count"
            require(
                count_name in columns,
                f"features_patterns.csv column {column} has no {count_name}.",
            )
    require(
        len(pairs) * 2 == len(columns),
        "features_patterns.csv count and missing columns do not form complete pairs.",
    )
    return pairs


def require_pattern_coding(patterns):
    """Check the 74 measurement-pattern columns.

    Common covariates are omitted. They may contain the agreed missing values.
    """
    failures = []
    zero_counts = 0
    for count_name, missing_name in pattern_pairs(list(patterns.columns)):
        count = pd.to_numeric(patterns[count_name], errors="coerce")
        missing = pd.to_numeric(patterns[missing_name], errors="coerce")
        if count.isna().any():
            failures.append(
                f"features_patterns.csv {count_name}: "
                f"{int(count.isna().sum())} missing or non-numeric counts."
            )
            continue
        if missing.isna().any() or not missing.isin([0, 1]).all():
            invalid = int((~missing.isin([0, 1])).sum())
            failures.append(
                f"features_patterns.csv {missing_name}: "
                f"{invalid} values are not 0 or 1."
            )
            continue
        count_values = count.to_numpy(dtype=float)
        non_integer = int((count_values != np.floor(count_values)).sum())
        negative = int((count_values < 0).sum())
        if non_integer or negative:
            failures.append(
                f"features_patterns.csv {count_name}: "
                f"{negative} negative and {non_integer} non-integer counts."
            )
            continue
        expected_missing = count.eq(0).astype(int)
        mismatches = int(missing.ne(expected_missing).sum())
        if mismatches:
            failures.append(
                f"features_patterns.csv {missing_name}: "
                f"{mismatches} indicators disagree with {count_name} == 0."
            )
            continue
        zero_counts += int(count.eq(0).sum())
    require(
        not failures,
        "Pattern checks failed:\n" + "\n".join(failures),
    )
    require(
        zero_counts > 0,
        "features_patterns.csv has no zero counts in the training matrix.",
    )
    print(f"Retained zero pattern counts: {zero_counts}")
    print("Pattern missing indicators match count == 0.")


def score_lasso_candidate(features, outcome, splitter, feature_set, c_value, max_iter):
    search = GridSearchCV(
        estimator=make_lasso_pipeline(feature_set, max_iter),
        param_grid={"model__C": [c_value]},
        scoring="roc_auc",
        cv=splitter,
        refit=False,
        error_score="raise",
        n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(features, outcome)
        messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    results = search.cv_results_
    require(
        len(results["params"]) == 1,
        f"{feature_set} C={c_value}: unexpected candidate count.",
    )
    scores = [float(results[f"split{fold}_test_score"][0]) for fold in range(5)]
    mean_score = float(results["mean_test_score"][0])
    require(
        np.isfinite(scores).all(),
        f"{feature_set} C={c_value}: non-finite fold AUROC.",
    )
    require(
        np.isfinite(mean_score),
        f"{feature_set} C={c_value}: non-finite mean AUROC.",
    )
    require(
        np.isclose(mean_score, float(np.mean(scores))),
        f"{feature_set} C={c_value}: mean AUROC does not match the five folds.",
    )
    print(
        f"{feature_set} C={c_value}: mean AUROC {mean_score:.10f}; "
        f"warnings {len(messages)}",
        flush=True,
    )
    return {
        "feature_set": feature_set,
        "C": c_value,
        "scores": scores,
        "mean_auroc": mean_score,
        "warnings": messages,
    }


def choose_c(rows):
    """Select the highest unrounded mean AUROC, breaking ties toward smaller C."""
    best_mean = max(row["mean_auroc"] for row in rows)
    tied = [row for row in rows if row["mean_auroc"] == best_mean]
    return min(tied, key=lambda row: row["C"])


def tuning_table(rows, choices):
    records = []
    for row in rows:
        record = {
            "feature_set": row["feature_set"],
            "C": format(row["C"], "g"),
            "mean_auroc": format(row["mean_auroc"], ".17g"),
            "selected": int(row["C"] == choices[row["feature_set"]]["C"]),
        }
        for fold, score in enumerate(row["scores"], start=1):
            record[f"fold_{fold}_auroc"] = format(score, ".17g")
        records.append(record)
    columns = [
        "feature_set",
        "C",
        "fold_1_auroc",
        "fold_2_auroc",
        "fold_3_auroc",
        "fold_4_auroc",
        "fold_5_auroc",
        "mean_auroc",
        "selected",
    ]
    return pd.DataFrame(records)[columns]


def verify_saved_tuning_table(path, choices):
    saved = pd.read_csv(path)
    expected_columns = [
        "feature_set",
        "C",
        "fold_1_auroc",
        "fold_2_auroc",
        "fold_3_auroc",
        "fold_4_auroc",
        "fold_5_auroc",
        "mean_auroc",
        "selected",
    ]
    require(list(saved.columns) == expected_columns, "Saved tuning table columns differ.")
    require(len(saved) == 15, f"Saved tuning table has {len(saved)} rows, expected 15.")
    require(
        not saved.duplicated(["feature_set", "C"]).any(),
        "Saved tuning table repeats a feature set and C.",
    )
    score_columns = [column for column in expected_columns if column.endswith("auroc")]
    require(
        np.isfinite(saved[score_columns].to_numpy(dtype=float)).all(),
        "Saved tuning table contains a non-finite score.",
    )
    require(
        set(saved["feature_set"]) == set(LASSO_FEATURE_SETS),
        "Saved tuning table is missing a feature set.",
    )
    saved_pairs = set(zip(saved["feature_set"], saved["C"].map(float)))
    expected_pairs = {
        (feature_set, c_value)
        for feature_set in LASSO_FEATURE_SETS
        for c_value in LASSO_TUNE_C
    }
    require(saved_pairs == expected_pairs, "Saved tuning grid does not match the fixed C values.")
    for feature_set, chosen in choices.items():
        group = saved.loc[saved["feature_set"].eq(feature_set)]
        selected = group.loc[group["selected"].eq(1), "C"]
        require(
            len(selected) == 1,
            f"Saved table does not mark exactly one C for {feature_set}.",
        )
        require(
            float(selected.iloc[0]) == chosen["C"],
            f"Saved selection for {feature_set} differs from the unrounded choice.",
        )
        means = group["mean_auroc"].to_numpy(dtype=float)
        best_mean = float(np.max(means))
        tied = group.loc[np.isclose(means, best_mean, rtol=0, atol=0), "C"].map(float)
        # Exact equality can fail after formatting, so also require the stored
        # choice to be one of the re-read maxima and the smallest such C.
        require(
            float(selected.iloc[0]) == float(tied.min()),
            f"Re-read means do not select the saved C for {feature_set}.",
        )


def run_lasso_tuning():
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Inputs: {PROCESSED}")
    print(f"Tuning output: {LASSO_TUNING_TABLE}")
    matrices, training = load_lasso_training_matrices()
    for feature_set in LASSO_FEATURE_SETS:
        pipeline = make_lasso_pipeline(feature_set)
        print(
            f"{feature_set} training matrix: "
            f"{matrices[feature_set].shape[0]} x {matrices[feature_set].shape[1]}; "
            "CV estimator steps: "
            + ", ".join(name for name, _step in pipeline.steps)
        )
    splitter = predefined_training_split(training)
    require_predefined_folds_match(matrices["values"], training, splitter)
    model = make_lasso_pipeline("values").named_steps["model"]
    require(model.l1_ratio == 1, "LASSO tuning l1_ratio is not 1.")
    print(
        "L1 setting: "
        f"l1_ratio={model.l1_ratio}, solver={model.solver!r}, "
        f"fit_intercept={bool(model.fit_intercept)}, "
        f"random_state={model.random_state}, max_iter={model.max_iter}"
    )
    outcome = training[OUTCOME].to_numpy(dtype=int)
    rows = []
    for feature_set in LASSO_FEATURE_SETS:
        for c_value in LASSO_TUNE_C:
            rows.append(
                score_lasso_candidate(
                    matrices[feature_set],
                    outcome,
                    splitter,
                    feature_set,
                    c_value,
                    LASSO_TUNE_MAX_ITER,
                )
            )

    affected = [row for row in rows if row["warnings"]]
    print(f"Fit warnings: {sum(len(row['warnings']) for row in rows)}")
    for row in affected:
        print(
            f"Affected: {row['feature_set']} C={row['C']} "
            f"({len(row['warnings'])} warnings)"
        )
        for message in sorted(set(row["warnings"])):
            print(message)
    require(
        not affected,
        "Unsolved fit warnings remain. C was not selected and no tuning table was written.",
    )

    choices = {}
    boundary = {LASSO_TUNE_C[0], LASSO_TUNE_C[-1]}
    for feature_set in LASSO_FEATURE_SETS:
        group = [row for row in rows if row["feature_set"] == feature_set]
        chosen = choose_c(group)
        choices[feature_set] = chosen
        location = "grid boundary" if chosen["C"] in boundary else "interior grid point"
        print(
            f"Selected {feature_set}: C={chosen['C']} "
            f"mean AUROC {chosen['mean_auroc']:.10f} ({location})"
        )

    LASSO_TUNING_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table = tuning_table(rows, choices)
    table.to_csv(LASSO_TUNING_TABLE, index=False, lineterminator="\n")
    verify_saved_tuning_table(LASSO_TUNING_TABLE, choices)
    print(f"Tuning table: {LASSO_TUNING_TABLE}")
    print("Re-read check: 15 unique rows, finite scores, one selected C per feature set.")


def read_selected_penalties(path=LASSO_TUNING_TABLE):
    """Read one selected C per feature set from the saved tuning table."""
    require(path.is_file(), f"Tuning table is missing: {path}")
    table = pd.read_csv(path)
    expected_columns = [
        "feature_set",
        "C",
        "fold_1_auroc",
        "fold_2_auroc",
        "fold_3_auroc",
        "fold_4_auroc",
        "fold_5_auroc",
        "mean_auroc",
        "selected",
    ]
    require(
        list(table.columns) == expected_columns,
        f"{path.name} columns are {list(table.columns)}.",
    )
    require(
        set(table["feature_set"]) == set(LASSO_FEATURE_SETS),
        f"{path.name} feature sets are {sorted(set(table['feature_set']))}.",
    )
    require(
        not table.duplicated(["feature_set", "C"]).any(),
        f"{path.name} repeats a feature set and C.",
    )

    penalties = {}
    for feature_set in LASSO_FEATURE_SETS:
        group = table.loc[table["feature_set"].eq(feature_set)].copy()
        require(not group.empty, f"{path.name} has no rows for {feature_set}.")
        selected_flag = pd.to_numeric(group["selected"], errors="raise")
        require(
            selected_flag.isin([0, 1]).all(),
            f"{path.name} {feature_set} has a selected flag outside 0 or 1.",
        )
        selected_count = int(selected_flag.sum())
        require(
            selected_count == 1,
            f"{path.name} {feature_set} has {selected_count} selected rows.",
        )
        group["C"] = pd.to_numeric(group["C"], errors="raise")
        group["mean_auroc"] = pd.to_numeric(group["mean_auroc"], errors="raise")
        require(
            np.isfinite(group["C"]).all() and np.isfinite(group["mean_auroc"]).all(),
            f"{path.name} {feature_set} has a non-finite C or mean AUROC.",
        )
        rows = [
            {"C": float(c_value), "mean_auroc": float(mean_score)}
            for c_value, mean_score in zip(group["C"], group["mean_auroc"])
        ]
        chosen = choose_c(rows)
        selected_c = float(group.loc[selected_flag.eq(1), "C"].iloc[0])
        require(
            selected_c == chosen["C"],
            f"{path.name} {feature_set} selects C={selected_c}, "
            f"but the mean-AUROC rule selects C={chosen['C']}.",
        )
        penalties[feature_set] = selected_c
        print(
            f"Read {feature_set} C={selected_c} "
            f"(mean AUROC {chosen['mean_auroc']:.10f})"
        )
    return penalties


def load_feature_spec():
    """Read predictor, numeric, and categorical columns for each feature set."""
    payload = require_input_version()
    require(payload.get("id_column") == "RecordID", "feature_sets.json id_column must be RecordID.")
    require(
        payload.get("common_covariates") == COMMON_COVARIATES,
        "feature_sets.json common covariates must be Age, Gender, and ICUType.",
    )
    require(
        payload.get("categorical_features") == ["Gender", "ICUType"],
        "feature_sets.json categorical features must be Gender and ICUType.",
    )
    groups = payload.get("feature_sets")
    require(
        isinstance(groups, dict) and set(groups) == set(LASSO_FEATURE_SETS),
        "feature_sets.json must contain values, patterns, and combined.",
    )

    spec = {}
    for feature_set in LASSO_FEATURE_SETS:
        block = groups[feature_set]
        require(isinstance(block, dict), f"feature_sets.json {feature_set} must be an object.")
        columns = block.get("columns")
        numeric = block.get("numeric_columns")
        categorical = block.get("categorical_columns")
        expected = FEATURE_SET_WIDTHS[feature_set]
        require(
            isinstance(columns, list) and len(columns) == expected,
            f"{feature_set} must list {expected} columns before encoding, "
            f"found {len(columns) if isinstance(columns, list) else columns!r}.",
        )
        require(len(columns) == len(set(columns)), f"feature_sets.json repeats a {feature_set} predictor.")
        require(
            isinstance(numeric, list) and isinstance(categorical, list),
            f"{feature_set} numeric_columns and categorical_columns must be lists.",
        )
        require(set(numeric).isdisjoint(categorical), f"{feature_set} numeric and categorical columns overlap.")
        require(
            set(numeric) | set(categorical) == set(columns),
            f"{feature_set} numeric and categorical columns must cover every predictor.",
        )
        require(categorical == ["Gender", "ICUType"], f"{feature_set} categorical columns must be Gender and ICUType.")
        require("Age" in numeric and "Age" not in categorical, f"{feature_set} must keep Age in numeric_columns.")
        excluded = EXCLUDED_COLUMNS.intersection(columns)
        require(not excluded, f"{feature_set} includes excluded columns: {sorted(excluded)}.")
        require(
            columns[:3] == COMMON_COVARIATES,
            f"{feature_set} columns must start with Age, Gender, and ICUType.",
        )
        spec[feature_set] = {
            "columns": columns,
            "numeric_columns": numeric,
            "categorical_columns": categorical,
        }

    value_measurements = [column for column in spec["values"]["numeric_columns"] if column != "Age"]
    pattern_measurements = [column for column in spec["patterns"]["numeric_columns"] if column != "Age"]
    require(
        len(value_measurements) == N_MEASUREMENT_VALUES,
        f"Expected {N_MEASUREMENT_VALUES} measurement value columns, found {len(value_measurements)}.",
    )
    require(
        len(pattern_measurements) == N_MEASUREMENT_PATTERNS,
        f"Expected {N_MEASUREMENT_PATTERNS} measurement pattern columns, found {len(pattern_measurements)}.",
    )
    require(
        set(value_measurements).isdisjoint(pattern_measurements),
        "Measurement value and pattern column names overlap.",
    )
    require(
        set(value_measurements).isdisjoint(COMMON_COVARIATES),
        "Measurement value columns include a common covariate.",
    )
    require(
        set(pattern_measurements).isdisjoint(COMMON_COVARIATES),
        "Measurement pattern columns include a common covariate.",
    )
    require(
        spec["values"]["columns"] == COMMON_COVARIATES + value_measurements,
        "Values columns are not the common covariates followed by measurement values.",
    )
    require(
        spec["patterns"]["columns"] == COMMON_COVARIATES + pattern_measurements,
        "Patterns columns are not the common covariates followed by measurement patterns.",
    )
    require(
        spec["combined"]["columns"] == COMMON_COVARIATES + value_measurements + pattern_measurements,
        "Combined columns repeat a common covariate or omit a measurement column.",
    )
    require(
        spec["combined"]["numeric_columns"] == ["Age", *value_measurements, *pattern_measurements],
        "Combined numeric columns do not match Age and the measurement columns.",
    )
    return {
        "sets": spec,
        "measurement_value_columns": value_measurements,
        "measurement_pattern_columns": pattern_measurements,
    }


def require_v2_feature_consistency(matrices, cohort, spec):
    """Check common covariates, measurement patterns, and retained empty rows once."""
    value_columns = spec["measurement_value_columns"]
    pattern_columns = spec["measurement_pattern_columns"]
    for feature_set in ("patterns", "combined"):
        for column in COMMON_COVARIATES:
            require_same_series(
                matrices["values"][column],
                matrices[feature_set][column],
                f"{feature_set} {column} differs from values",
            )
    for column in value_columns:
        require_same_series(
            matrices["combined"][column],
            matrices["values"][column],
            f"Combined measurement column {column} differs from values",
        )
    for column in pattern_columns:
        require_same_series(
            matrices["combined"][column],
            matrices["patterns"][column],
            f"Combined measurement column {column} differs from patterns",
        )
    combined_columns = list(matrices["combined"].columns)
    require(
        combined_columns == COMMON_COVARIATES + value_columns + pattern_columns,
        "Combined columns are not common covariates plus measurement columns.",
    )
    require(
        all(combined_columns.count(column) == 1 for column in COMMON_COVARIATES),
        "Combined repeats a common covariate.",
    )
    require_same_series(
        cohort.loc[matrices["values"].index, "Age"],
        matrices["values"]["Age"],
        "baseline.csv Age differs from the values Age column",
    )
    require_pattern_coding(matrices["patterns"].loc[:, pattern_columns])

    training = training_rows(cohort)
    test = cohort.loc[cohort["split"].eq("test")]
    require(len(training) == 3200, f"Expected 3200 training stays, found {len(training)}.")
    require(len(test) == 800, f"Expected 800 test stays, found {len(test)}.")
    require(set(training.index).isdisjoint(test.index), "Training and test RecordID sets overlap.")
    require(
        len(training) + len(test) == len(cohort),
        "Training and test rows do not cover the aligned cohort.",
    )
    require(test["cv_fold"].isna().all(), "Test stays have a training fold number.")
    fold_counts = training["cv_fold"].value_counts()
    require(
        set(fold_counts.index) == {1, 2, 3, 4, 5} and bool(fold_counts.eq(640).all()),
        f"Training folds are not five groups of 640: {fold_counts.to_dict()}.",
    )
    training_common = matrices["values"].loc[training.index, COMMON_COVARIATES]
    gender = pd.to_numeric(training_common["Gender"], errors="raise")
    icu_type = pd.to_numeric(training_common["ICUType"], errors="raise")
    require(int(training_common["Age"].isna().sum()) == 0, "Training Age contains missing values.")
    require(int(gender.isna().sum()) == 3, f"Training Gender has {int(gender.isna().sum())} missing values, expected 3.")
    require(int(icu_type.isna().sum()) == 0, "Training ICUType contains missing values.")
    require(
        set(gender.dropna().unique()).issubset({0, 1}),
        f"Training Gender codes are {sorted(gender.dropna().unique().tolist())}.",
    )
    require(
        set(icu_type.dropna().unique()).issubset({1, 2, 3, 4}),
        f"Training ICUType codes are {sorted(icu_type.dropna().unique().tolist())}.",
    )
    measurement = matrices["values"].loc[training.index, value_columns]
    empty_ids = measurement.index[measurement.isna().all(axis=1).to_numpy()]
    require(
        len(empty_ids) == 3,
        "Expected 3 training stays with all 111 measurement values missing, "
        f"found {len(empty_ids)}.",
    )
    for feature_set in LASSO_FEATURE_SETS:
        require(
            empty_ids.isin(matrices[feature_set].index).all(),
            f"All-missing measurement stays are absent from {feature_set}.",
        )
    print(f"Training rows with all 111 measurement values missing: {len(empty_ids)} (retained)")


def load_feature_matrices():
    """Align the three V2 feature tables to one RecordID order and check them once."""
    cohort = load_aligned_inputs().sort_index()
    spec = load_feature_spec()
    tables = {"cohort": cohort}
    for feature_set in LASSO_FEATURE_SETS:
        columns = spec["sets"][feature_set]["columns"]
        path = PROCESSED / FEATURE_SET_FILES[feature_set]
        frame = read_indexed_table(path, columns)
        require(
            list(frame.columns) == columns,
            f"{path.name} columns differ from feature_sets.json {feature_set}.",
        )
        tables[feature_set] = frame
    require_same_record_ids(tables)

    matrices = {}
    for feature_set in LASSO_FEATURE_SETS:
        columns = spec["sets"][feature_set]["columns"]
        path = PROCESSED / FEATURE_SET_FILES[feature_set]
        features = tables[feature_set].loc[cohort.index, columns]
        require(
            features.index.equals(cohort.index),
            f"{path.name} rows are not aligned with the cohort RecordID order.",
        )
        require(len(features) == len(cohort), f"{path.name} alignment changed the number of stays.")
        matrices[feature_set] = features
    require_v2_feature_consistency(matrices, cohort, spec)
    return cohort, matrices, spec


def load_train_and_test():
    """Align age and LASSO inputs, then split them without dropping stays."""
    cohort, matrices, _spec = load_feature_matrices()
    training = training_rows(cohort)
    test = cohort.loc[cohort["split"].eq("test")].copy()
    require(training.index.is_monotonic_increasing, "Training RecordID order is not sorted.")
    require(test.index.is_monotonic_increasing, "Test RecordID order is not sorted.")
    require(len(training) == 3200, f"Expected 3200 training stays, found {len(training)}.")
    require(len(test) == 800, f"Expected 800 test stays, found {len(test)}.")

    train_features = {"age_only": training.loc[:, AGE_PREDICTORS]}
    test_features = {"age_only": test.loc[:, AGE_PREDICTORS]}
    require(list(train_features["age_only"].columns) == ["Age"], "Age baseline predictors are not ['Age'].")
    require(list(test_features["age_only"].columns) == ["Age"], "Age baseline predictors are not ['Age'].")
    for feature_set in LASSO_FEATURE_SETS:
        train_matrix = matrices[feature_set].loc[training.index]
        test_matrix = matrices[feature_set].loc[test.index]
        width = FEATURE_SET_WIDTHS[feature_set]
        require(
            train_matrix.index.equals(training.index),
            f"{feature_set} training rows are not aligned by RecordID.",
        )
        require(
            test_matrix.index.equals(test.index),
            f"{feature_set} test rows are not aligned by RecordID.",
        )
        require(
            train_matrix.shape == (3200, width),
            f"{feature_set} training matrix is {train_matrix.shape}, expected (3200, {width}) before encoding.",
        )
        require(
            test_matrix.shape == (800, width),
            f"{feature_set} test matrix is {test_matrix.shape}, expected (800, {width}) before encoding.",
        )
        train_features[feature_set] = train_matrix
        test_features[feature_set] = test_matrix
    return training, test, train_features, test_features


def fit_model(pipeline, features, outcome, name):
    """Fit one pipeline on the supplied training rows and keep warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(features, outcome)
        messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    model = pipeline.named_steps["model"]
    n_iter = int(np.max(model.n_iter_))
    print(
        f"Fit {name}: rows={len(features)}, predictors={features.shape[1]}, "
        f"n_iter={n_iter}, max_iter={model.max_iter}, warnings={len(messages)}",
        flush=True,
    )
    for message in messages:
        print(message)
    require(
        not messages and n_iter < int(model.max_iter),
        f"{name} did not converge. Test predictions were not written.",
    )
    if "scaler" in pipeline.named_steps:
        seen = int(np.max(pipeline.named_steps["scaler"].n_samples_seen_))
        require(
            seen == len(features),
            f"{name} scaler saw {seen} rows, not {len(features)}.",
        )
    return messages


def require_prediction_series(probabilities, features, test_ids, name):
    require(
        probabilities.index.equals(features.index),
        f"{name} probabilities are not bound to the predicted RecordID index.",
    )
    identifiers = pd.Index(probabilities.index.astype(str))
    require(identifiers.is_unique, f"{name} prediction IDs are not unique.")
    require((identifiers.str.strip() != "").all(), f"{name} has an empty RecordID.")
    require(len(probabilities) == 800, f"{name} predicted {len(probabilities)} rows.")
    require(
        set(identifiers) == set(test_ids),
        f"{name} prediction IDs differ from the test split.",
    )
    values = probabilities.to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"{name} probability is not finite.")
    require(
        ((values >= 0) & (values <= 1)).all(),
        f"{name} probability is outside [0, 1].",
    )


def write_prediction(probabilities, path):
    frame = pd.DataFrame(
        {
            "RecordID": probabilities.index.astype(str),
            "predicted_probability": [
                format(float(value), ".17g") for value in probabilities.to_numpy()
            ],
        }
    )
    require(
        list(frame.columns) == ["RecordID", "predicted_probability"],
        "Prediction output columns are not RecordID and predicted_probability.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def verify_saved_prediction(path, probabilities, test_ids):
    saved = pd.read_csv(path, dtype={"RecordID": str})
    require(
        list(saved.columns) == ["RecordID", "predicted_probability"],
        f"{path.name} columns are {list(saved.columns)}.",
    )
    require(len(saved) == 800, f"{path.name} has {len(saved)} rows.")
    identifiers = saved["RecordID"].str.strip()
    require(identifiers.notna().all() and identifiers.ne("").all(), f"{path.name} has an empty RecordID.")
    require(identifiers.is_unique, f"{path.name} repeats a RecordID.")
    require(
        set(identifiers) == set(test_ids),
        f"{path.name} IDs differ from the test split.",
    )
    values = pd.to_numeric(saved["predicted_probability"], errors="raise")
    require(np.isfinite(values).all(), f"{path.name} has a non-finite probability.")
    require(
        values.ge(0).all() and values.le(1).all(),
        f"{path.name} has a probability outside [0, 1].",
    )
    reread = pd.Series(values.to_numpy(dtype=float), index=identifiers)
    aligned = reread.loc[probabilities.index.astype(str)]
    difference = np.abs(aligned.to_numpy(dtype=float) - probabilities.to_numpy(dtype=float))
    require(
        np.allclose(
            aligned.to_numpy(dtype=float),
            probabilities.to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ),
        f"{path.name} reread probabilities differ from the in-memory predictions.",
    )
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--", path.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    require(completed.returncode == 0, f"{path.relative_to(ROOT)} is not ignored by Git.")
    return float(np.max(difference))


def run_test_predictions():
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Inputs: {PROCESSED}")
    print(f"Tuning table: {LASSO_TUNING_TABLE}")
    print(f"Prediction output: {PREDICTION_DIR}")
    penalties = read_selected_penalties()
    training, test, train_features, test_features = load_train_and_test()
    outcome = training[OUTCOME].to_numpy(dtype=int)
    fitted = {}

    age_pipeline = make_age_baseline()
    fit_model(age_pipeline, train_features["age_only"], outcome, "age_only")
    fitted["age_only"] = age_pipeline

    for feature_set in LASSO_FEATURE_SETS:
        pipeline = make_lasso_pipeline(feature_set)
        pipeline.set_params(model__C=penalties[feature_set])
        require(
            pipeline.named_steps["model"].C == penalties[feature_set],
            f"{feature_set} model C was not set from the tuning table.",
        )
        require(
            pipeline.named_steps["model"].max_iter == LASSO_TUNE_MAX_ITER,
            f"{feature_set} max_iter is not {LASSO_TUNE_MAX_ITER}.",
        )
        fit_model(pipeline, train_features[feature_set], outcome, feature_set)
        fitted[feature_set] = pipeline

    predictions = {
        name: death_probabilities(pipeline, test_features[name])
        for name, pipeline in fitted.items()
    }
    for name, probabilities in predictions.items():
        require_prediction_series(probabilities, test_features[name], test.index, name)

    for name, probabilities in predictions.items():
        write_prediction(probabilities, PREDICTION_FILES[name])

    summaries = []
    for name, probabilities in predictions.items():
        max_difference = verify_saved_prediction(
            PREDICTION_FILES[name], probabilities, test.index
        )
        model = fitted[name].named_steps["model"]
        n_iter = int(np.max(model.n_iter_))
        if name == "age_only":
            encoded = filtered = int(model.n_features_in_)
        else:
            encoded = len(fitted[name].named_steps["preprocess"].get_feature_names_out())
            filtered = int(fitted[name].named_steps["constant_filter"].get_support().sum())
            require(
                filtered == int(model.n_features_in_),
                f"{name} model saw {model.n_features_in_} columns after filtering {filtered}.",
            )
        penalty = "not applicable" if name == "age_only" else penalties[name]
        values = probabilities.to_numpy(dtype=float)
        summaries.append(
            {
                "model": PREDICTION_FILES[name].stem,
                "n_fit": len(train_features[name]),
                "n_predictors": int(train_features[name].shape[1]),
                "encoded_columns": encoded,
                "filtered_columns": filtered,
                "C": penalty,
                "n_iter": n_iter,
                "n_predicted": len(probabilities),
                "min_probability": float(values.min()),
                "max_probability": float(values.max()),
                "warnings": 0,
                "max_reread_abs_diff": max_difference,
            }
        )
        print(
            f"{summaries[-1]['model']}: "
            f"fit={summaries[-1]['n_fit']}, predictors={summaries[-1]['n_predictors']}, "
            f"encoded={encoded}, filtered={filtered}, "
            f"C={summaries[-1]['C']}, n_iter={n_iter}, "
            f"predicted={summaries[-1]['n_predicted']}, "
            f"range=[{values.min():.10f}, {values.max():.10f}], "
            f"warnings=0, max_reread_abs_diff={max_difference:.3g}"
        )
    print("Git ignore check passed for all four prediction files.")
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit COMMON_COVARIATES_V2 age and LASSO mortality models "
            "on the shared training split."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/11_train_regularized_models.py --tune-lasso\n"
            "  python scripts/11_train_regularized_models.py --predict-test"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tune-lasso",
        action="store_true",
        help="Select C for the three LASSO feature sets on the shared training folds.",
    )
    mode.add_argument(
        "--predict-test",
        action="store_true",
        help="Refit selected models on all training stays and write test probabilities.",
    )
    args = parser.parse_args()
    if not args.tune_lasso and not args.predict_test:
        parser.print_help()
        return
    print(f"scikit-learn {sklearn.__version__}")
    if args.tune_lasso:
        run_lasso_tuning()
        return
    run_test_predictions()


if __name__ == "__main__":
    main()
