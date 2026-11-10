"""Train regularized in-hospital mortality models.

Run from the repository root:
    python scripts/11_train_regularized_models.py
    python scripts/11_train_regularized_models.py --check-values-lasso
    python scripts/11_train_regularized_models.py --tune-lasso
    python scripts/11_train_regularized_models.py --predict-test

The default entry point fits the age-only baseline on training folds
2-5 and checks one probability per stay in training fold 1.
The values check runs L1 logistic regression for C=0.1 and C=1
on the shared training folds. It does not select the final penalty.
The tuning entry selects one C for each LASSO feature set from the
fixed training-fold grid. It does not refit on all training rows.
The prediction entry refits the age baseline and three LASSO models on
all training stays and writes test-set probabilities.
"""

from pathlib import Path
import argparse
import json
import subprocess
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LASSO_TUNING_TABLE = (
    ROOT / "results" / "tables" / "regularized_models" / "lasso_cv_results.csv"
)
PREDICTION_DIR = ROOT / "data" / "predictions"
PREDICTION_FILES = {
    "age_only": PREDICTION_DIR / "age_only.csv",
    "values": PREDICTION_DIR / "lasso_values.csv",
    "patterns": PREDICTION_DIR / "lasso_patterns.csv",
    "combined": PREDICTION_DIR / "lasso_combined.csv",
}

AGE_PREDICTORS = ["Age"]
OUTCOME = "In-hospital_death"
POSITIVE_CLASS = 1
VALUES_LASSO_SOLVER = "saga"
VALUES_LASSO_RANDOM_STATE = 707
# scikit-learn 1.6.1 defaults to 100 iterations. Every values-fold fit
# reached that cap. A cap of 5000 covered the observed counts, at most 3740.
VALUES_LASSO_MAX_ITER = 5000
VALUES_LASSO_CHECK_C = [0.1, 1.0]
LASSO_TUNE_C = [0.001, 0.01, 0.1, 1.0, 10.0]
# max_iter=5000 left ConvergenceWarning on values C=10 (3 folds) and
# combined C=10 (4 folds). Those fits reached convergence by 6959 iterations.
LASSO_TUNE_MAX_ITER = 20000
LASSO_FEATURE_SETS = ("values", "patterns", "combined")
FEATURE_SET_FILES = {
    "values": "features_values.csv",
    "patterns": "features_patterns.csv",
    "combined": "features_combined.csv",
}
FEATURE_SET_WIDTHS = {"values": 111, "patterns": 74, "combined": 185}
# Mean validation AUROC from Python 3.13.5 and scikit-learn 1.6.1.
# These values are reported for cross-version comparison. An exact match
# is not required. Fold IDs, finite scores, and the five-fold mean are
# still checked.
VALUES_REFERENCE_MEAN_AUROC = {
    0.1: "0.8179274444",
    1.0: "0.8094112613",
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


def load_aligned_inputs():
    """Align baseline, outcome, and split rows by RecordID without a join."""
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


def run_age_baseline_fold_check(frame):
    training = training_rows(frame)
    fit_frame = training.loc[training["cv_fold"].isin([2, 3, 4, 5])]
    validation_frame = training.loc[training["cv_fold"].eq(1)]
    fit_ids = set(fit_frame.index)
    validation_ids = set(validation_frame.index)

    require(fit_ids.isdisjoint(validation_ids), "Fit and validation IDs overlap.")
    require(
        len(fit_frame) + len(validation_frame) == len(training),
        "Fit and validation folds do not cover the training set.",
    )
    fit_deaths = int(fit_frame[OUTCOME].sum())
    validation_deaths = int(validation_frame[OUTCOME].sum())
    require(len(fit_frame) == 2560, f"Expected 2560 fit rows, found {len(fit_frame)}.")
    require(fit_deaths == 355, f"Expected 355 fit deaths, found {fit_deaths}.")
    require(
        len(validation_frame) == 640,
        f"Expected 640 validation rows, found {len(validation_frame)}.",
    )
    require(
        validation_deaths == 88,
        f"Expected 88 validation deaths, found {validation_deaths}.",
    )

    fit_features = fit_frame.loc[:, AGE_PREDICTORS]
    validation_features = validation_frame.loc[:, AGE_PREDICTORS]
    require(list(fit_features.columns) == ["Age"], "Age baseline predictors are not ['Age'].")
    require(
        list(validation_features.columns) == ["Age"],
        "Validation predictors are not ['Age'].",
    )

    pipeline = make_age_baseline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(fit_features, fit_frame[OUTCOME])
        probabilities = death_probabilities(pipeline, validation_features)
        warning_messages = [
            f"{item.category.__name__}: {item.message}" for item in caught
        ]

    require(
        probabilities.index.equals(validation_frame.index),
        "Probability IDs differ from the validation IDs.",
    )
    values = probabilities.to_numpy(dtype=float)
    require(len(values) == 640, f"Expected 640 probabilities, found {len(values)}.")
    require(np.isfinite(values).all(), "Validation probability is not finite.")
    require(
        ((values >= 0) & (values <= 1)).all(),
        "Validation probability is outside [0, 1].",
    )

    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    require(np.isinf(model.C), "Age baseline C is not infinite.")
    require(model.l1_ratio == 0, "Age baseline l1_ratio is not 0.")
    require(len(imputer.statistics_) == 1, "Age imputer did not learn one median.")

    return {
        "n_fit": len(fit_frame),
        "fit_deaths": fit_deaths,
        "n_validation": len(validation_frame),
        "validation_deaths": validation_deaths,
        "n_intersection": len(fit_ids & validation_ids),
        "fit_age_missing": int(fit_features["Age"].isna().sum()),
        "validation_age_missing": int(validation_features["Age"].isna().sum()),
        "learned_age_median": float(imputer.statistics_[0]),
        "predictors": list(fit_features.columns),
        "l1_ratio": model.l1_ratio,
        "C": model.C,
        "solver": model.solver,
        "fit_intercept": bool(model.fit_intercept),
        "classes": [int(label) for label in model.classes_],
        "probability_min": float(values.min()),
        "probability_max": float(values.max()),
        "warnings": warning_messages,
    }


def feature_predictors(feature_set):
    feature_sets = json.loads((PROCESSED / "feature_sets.json").read_text())
    predictors = feature_sets[feature_set]
    expected = FEATURE_SET_WIDTHS[feature_set]
    require(
        isinstance(predictors, list) and len(predictors) == expected,
        f"feature_sets.json {feature_set} must contain {expected} columns, "
        f"found {len(predictors)}.",
    )
    require(
        len(predictors) == len(set(predictors)),
        f"feature_sets.json repeats a {feature_set} predictor.",
    )
    forbidden = {"RecordID", OUTCOME, "split", "cv_fold", *AGE_PREDICTORS}
    require(
        forbidden.isdisjoint(predictors),
        f"{feature_set} predictors include an identifier, outcome, split, or age column.",
    )
    return predictors


def value_predictors():
    return feature_predictors("values")


def load_values_training():
    """Align value features to outcomes and splits, then keep training stays."""
    predictors = value_predictors()
    tables = {
        "outcomes": read_indexed_table(PROCESSED / "outcomes.csv", [OUTCOME]),
        "splits": read_indexed_table(
            PROCESSED / "split_assignments.csv", ["split", "cv_fold"]
        ),
        "values": read_indexed_table(
            PROCESSED / "features_values.csv", predictors
        ),
    }
    require_same_record_ids(tables)
    require(
        list(tables["values"].columns) == predictors,
        "features_values.csv columns differ from feature_sets.json values.",
    )

    ordered_ids = tables["outcomes"].index
    outcome = pd.to_numeric(tables["outcomes"][OUTCOME], errors="raise")
    require(outcome.isin([0, 1]).all(), "In-hospital_death must be 0 or 1.")
    aligned = pd.DataFrame(
        {
            OUTCOME: outcome.astype(int),
            "split": tables["splits"].loc[ordered_ids, "split"],
            "cv_fold": tables["splits"].loc[ordered_ids, "cv_fold"],
        },
        index=ordered_ids,
    )
    aligned.index.name = "RecordID"
    training = training_rows(aligned).sort_index()
    require(len(training) == 3200, f"Expected 3200 training stays, found {len(training)}.")

    features = tables["values"].loc[training.index, predictors]
    require(
        features.index.equals(training.index),
        "Value-feature rows are not aligned with training outcomes and folds.",
    )
    require(features.shape == (3200, 111), f"Unexpected values matrix shape {features.shape}.")
    return features, training


def make_lasso_pipeline(max_iter=VALUES_LASSO_MAX_ITER):
    """Return the shared LASSO pipeline.

    Median imputation keeps all-missing columns, without missingness
    indicators. VarianceThreshold then removes those constant columns, and
    the same fitted support is used at prediction. Standardization and the
    L1 model are inside the pipeline so each CV fit learns them only on its
    own fitting rows. All three feature sets use this same sequence.

    scikit-learn 1.9 deprecates penalty='l1'. l1_ratio=1 is the pure L1
    penalty, with the penalty argument left unset.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("constant_filter", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver=VALUES_LASSO_SOLVER,
                    fit_intercept=True,
                    random_state=VALUES_LASSO_RANDOM_STATE,
                    max_iter=max_iter,
                ),
            ),
        ]
    )


def make_values_lasso():
    """Return the values LASSO pipeline at the validated iteration cap."""
    return make_lasso_pipeline()


def require_empty_columns_are_removed():
    """Confirm this sklearn version keeps, then drops, an all-missing column."""
    pipeline = make_values_lasso()
    imputer = pipeline.named_steps["imputer"]
    constant_filter = pipeline.named_steps["constant_filter"]
    require(imputer.add_indicator is False, "Missingness indicators are enabled.")
    require(
        imputer.keep_empty_features is True,
        "Imputer would drop all-missing columns before the constant filter.",
    )
    sample = pd.DataFrame(
        {
            "observed": [0.0, 1.0, 0.0, 1.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "empty": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    preprocessor = Pipeline(pipeline.steps[:-1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = preprocessor.fit_transform(sample)
        imputed = imputer.transform(sample)
        follow_up = preprocessor.transform(
            pd.DataFrame(
                {
                    "observed": [2.0, 3.0],
                    "constant": [5.0, 5.0],
                    "empty": [np.nan, np.nan],
                }
            )
        )
    require(not caught, f"Empty-column check emitted warnings: {caught}.")
    require(imputed.shape == (4, 3), "Imputer did not keep the all-missing column.")
    require(
        np.all(imputed[:, 2] == 0),
        "All-missing column was not replaced by one constant.",
    )
    require(
        np.array_equal(constant_filter.get_support(), [True, False, False]),
        "Constant filter did not remove the empty and constant columns.",
    )
    require(transformed.shape == (4, 1), "Scaler did not use the filtered columns.")
    require(follow_up.shape == (2, 1), "Prediction did not reuse the fitted columns.")
    require(np.isfinite(transformed).all() and np.isfinite(follow_up).all(), "Non-finite scaled values.")


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
        print(
            f"Project fold {project_fold}: validation IDs match "
            f"split_assignments.csv ({len(validation_ids)} IDs)"
        )
    require(
        len(seen) == len(set(seen)) == len(training),
        "Training records do not each appear in exactly one validation fold.",
    )
    print("Each training RecordID appears in exactly one validation fold.")


def run_values_lasso_check():
    features, training = load_values_training()
    print(f"Values predictors: {features.shape[1]}")
    print(f"Training matrix: {features.shape[0]} x {features.shape[1]}")
    print(
        "Training rows with all value features missing: "
        f"{int(features.isna().all(axis=1).sum())}"
    )
    for project_fold in range(1, 6):
        fitting_rows = features.loc[training["cv_fold"].ne(project_fold)]
        empty_columns = int(fitting_rows.isna().all(axis=0).sum())
        print(
            f"Fitting rows for validation fold {project_fold}: "
            f"all-missing columns={empty_columns}"
        )

    require_empty_columns_are_removed()
    print("Empty-column handling: passed on scikit-learn", sklearn.__version__)

    splitter = predefined_training_split(training)
    require_predefined_folds_match(features, training, splitter)

    estimator = make_values_lasso()
    model = estimator.named_steps["model"]
    require(model.l1_ratio == 1, "Values LASSO l1_ratio is not 1.")
    print(
        "L1 setting: "
        f"l1_ratio={model.l1_ratio}, solver={model.solver!r}, "
        f"fit_intercept={bool(model.fit_intercept)}, "
        f"random_state={model.random_state}, max_iter={model.max_iter}"
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"model__C": VALUES_LASSO_CHECK_C},
        scoring="roc_auc",
        cv=splitter,
        refit=False,
        error_score="raise",
        n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(features, training[OUTCOME].to_numpy(dtype=int))
        warning_messages = [
            f"{item.category.__name__}: {item.message}" for item in caught
        ]

    results = search.cv_results_
    require(len(results["params"]) == len(VALUES_LASSO_CHECK_C), "Unexpected candidate count.")
    compared = []
    for row, params in enumerate(results["params"]):
        scores = [float(results[f"split{fold}_test_score"][row]) for fold in range(5)]
        mean_score = float(results["mean_test_score"][row])
        require(np.isfinite(scores).all(), f"Non-finite AUROC for C={params['model__C']}.")
        require(np.isfinite(mean_score), f"Non-finite mean AUROC for C={params['model__C']}.")
        require(
            np.isclose(mean_score, float(np.mean(scores))),
            f"Mean AUROC does not match the five folds for C={params['model__C']}.",
        )
        formatted = ", ".join(f"{score:.10f}" for score in scores)
        print(f"C={params['model__C']}: fold AUROC [{formatted}]")
        print(f"C={params['model__C']}: mean AUROC {mean_score:.10f}")
        compared.append({"C": float(params["model__C"]), "mean_auroc": mean_score})

    print(f"Warnings: {len(warning_messages)}")
    for message in warning_messages:
        print(message)
    require(not warning_messages, "Values LASSO check emitted warnings.")
    report_values_cross_version(compared)


def training_labels():
    """Return the sorted training labels shared by every LASSO feature set."""
    tables = {
        "outcomes": read_indexed_table(PROCESSED / "outcomes.csv", [OUTCOME]),
        "splits": read_indexed_table(
            PROCESSED / "split_assignments.csv", ["split", "cv_fold"]
        ),
    }
    require_same_record_ids(tables)
    ordered_ids = tables["outcomes"].index
    outcome = pd.to_numeric(tables["outcomes"][OUTCOME], errors="raise")
    require(outcome.isin([0, 1]).all(), "In-hospital_death must be 0 or 1.")
    aligned = pd.DataFrame(
        {
            OUTCOME: outcome.astype(int),
            "split": tables["splits"].loc[ordered_ids, "split"],
            "cv_fold": tables["splits"].loc[ordered_ids, "cv_fold"],
        },
        index=ordered_ids,
    )
    aligned.index.name = "RecordID"
    training = training_rows(aligned).sort_index()
    require(len(training) == 3200, f"Expected 3200 training stays, found {len(training)}.")
    require(
        training.index.is_monotonic_increasing,
        "Training RecordID order is not sorted.",
    )
    return training


def load_lasso_training_matrices():
    """Align all three feature sets to the same sorted training rows."""
    training = training_labels()
    tables = {
        "outcomes": read_indexed_table(PROCESSED / "outcomes.csv", [OUTCOME]),
    }
    predictors = {}
    for feature_set in LASSO_FEATURE_SETS:
        predictors[feature_set] = feature_predictors(feature_set)
        path = PROCESSED / FEATURE_SET_FILES[feature_set]
        frame = read_indexed_table(path, predictors[feature_set])
        require(
            list(frame.columns) == predictors[feature_set],
            f"{path.name} columns differ from feature_sets.json {feature_set}.",
        )
        tables[feature_set] = frame
    require_same_record_ids(tables)

    matrices = {}
    for feature_set in LASSO_FEATURE_SETS:
        path = PROCESSED / FEATURE_SET_FILES[feature_set]
        features = tables[feature_set].loc[training.index, predictors[feature_set]]
        require(
            features.index.equals(training.index),
            f"{path.name} rows are not aligned with the training RecordID order.",
        )
        expected_shape = (3200, FEATURE_SET_WIDTHS[feature_set])
        require(
            features.shape == expected_shape,
            f"{path.name} training matrix is {features.shape}, expected {expected_shape}.",
        )
        matrices[feature_set] = features
    for feature_set, features in matrices.items():
        require(
            features.index.equals(matrices["values"].index),
            f"{feature_set} training row order differs from values.",
        )
    require_pattern_coding(matrices["patterns"])
    require_combined_matches(matrices)
    return matrices, training


def column_mismatches(left, right):
    """Count missingness and value disagreements, treating paired NA values as equal."""
    missing_disagreement = left.isna() ^ right.isna()
    value_disagreement = left.notna() & right.notna() & left.ne(right)
    return int(missing_disagreement.sum()), int(value_disagreement.sum())


def require_combined_matches(matrices):
    value_columns = list(matrices["values"].columns)
    pattern_columns = list(matrices["patterns"].columns)
    combined_columns = list(matrices["combined"].columns)
    require(
        set(value_columns).isdisjoint(pattern_columns),
        "Value and pattern predictor names overlap.",
    )
    require(
        combined_columns == value_columns + pattern_columns,
        "features_combined.csv columns are not values followed by patterns.",
    )
    comparisons = (
        ("values", "features_values.csv", value_columns),
        ("patterns", "features_patterns.csv", pattern_columns),
    )
    for feature_set, file_name, columns in comparisons:
        for column in columns:
            left = matrices["combined"][column]
            right = matrices[feature_set][column]
            missing_disagreement, value_disagreement = column_mismatches(left, right)
            require(
                missing_disagreement == 0 and value_disagreement == 0,
                "features_combined.csv does not match "
                f"{file_name} in column {column}: "
                f"{missing_disagreement} missingness differences, "
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
    """Keep zero counts, and require missing indicators to follow those counts."""
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
        estimator=make_lasso_pipeline(max_iter),
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


def report_values_cross_version(rows):
    """Report values AUROC against the scikit-learn 1.6.1 reference.

    A numerical difference does not stop the check or the tuning run.
    """
    for c_value, expected in VALUES_REFERENCE_MEAN_AUROC.items():
        matches = [row for row in rows if row["C"] == c_value]
        require(len(matches) == 1, f"Values grid is missing C={c_value}.")
        observed = float(matches[0]["mean_auroc"])
        reference = float(expected)
        rendered = f"{observed:.10f}"
        difference = observed - reference
        status = (
            "matches at 10 decimals"
            if rendered == expected
            else "differs at 10 decimals"
        )
        print(
            f"Values cross-version C={c_value}: mean AUROC {rendered} "
            f"(sklearn 1.6.1 reference {expected}, "
            f"difference {difference:+.3g}, {status})"
        )


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
    matrices, training = load_lasso_training_matrices()
    for feature_set in LASSO_FEATURE_SETS:
        print(
            f"{feature_set} training matrix: "
            f"{matrices[feature_set].shape[0]} x {matrices[feature_set].shape[1]}"
        )
    print(
        "Training rows with all value features missing: "
        f"{int(matrices['values'].isna().all(axis=1).sum())}"
    )
    splitter = predefined_training_split(training)
    require_predefined_folds_match(matrices["values"], training, splitter)
    model = make_lasso_pipeline(LASSO_TUNE_MAX_ITER).named_steps["model"]
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

    value_rows = [row for row in rows if row["feature_set"] == "values"]
    report_values_cross_version(value_rows)

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


def load_train_and_test():
    """Align age and LASSO inputs, then split them without dropping stays."""
    cohort = load_aligned_inputs().sort_index()
    training = training_rows(cohort)
    test = cohort.loc[cohort["split"].eq("test")].copy()
    require(len(training) == 3200, f"Expected 3200 training stays, found {len(training)}.")
    require(len(test) == 800, f"Expected 800 test stays, found {len(test)}.")
    require(
        set(training.index).isdisjoint(test.index),
        "Training and test RecordID sets overlap.",
    )
    require(
        len(training) + len(test) == len(cohort),
        "Training and test rows do not cover the aligned cohort.",
    )
    splits = read_indexed_table(
        PROCESSED / "split_assignments.csv", ["split", "cv_fold"]
    )
    expected_test_ids = set(splits.index[splits["split"].eq("test")])
    require(
        set(test.index) == expected_test_ids,
        "Test IDs differ from split_assignments.csv.",
    )

    train_features = {"age_only": training.loc[:, AGE_PREDICTORS]}
    test_features = {"age_only": test.loc[:, AGE_PREDICTORS]}
    require(
        list(train_features["age_only"].columns) == ["Age"],
        "Age baseline predictors are not ['Age'].",
    )
    for feature_set in LASSO_FEATURE_SETS:
        predictors = feature_predictors(feature_set)
        path = PROCESSED / FEATURE_SET_FILES[feature_set]
        frame = read_indexed_table(path, predictors)
        require(
            list(frame.columns) == predictors,
            f"{path.name} columns differ from feature_sets.json {feature_set}.",
        )
        missing = len(set(cohort.index) - set(frame.index))
        extra = len(set(frame.index) - set(cohort.index))
        require(
            missing == 0 and extra == 0,
            f"{path.name} RecordID set differs from the aligned cohort: "
            f"{missing} missing, {extra} extra.",
        )
        train_matrix = frame.loc[training.index, predictors]
        test_matrix = frame.loc[test.index, predictors]
        require(
            train_matrix.index.equals(training.index),
            f"{path.name} training rows are not aligned by RecordID.",
        )
        require(
            test_matrix.index.equals(test.index),
            f"{path.name} test rows are not aligned by RecordID.",
        )
        require(
            train_matrix.shape == (3200, FEATURE_SET_WIDTHS[feature_set]),
            f"{path.name} training matrix is {train_matrix.shape}.",
        )
        require(
            test_matrix.shape == (800, FEATURE_SET_WIDTHS[feature_set]),
            f"{path.name} test matrix is {test_matrix.shape}.",
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
    penalties = read_selected_penalties()
    training, test, train_features, test_features = load_train_and_test()
    outcome = training[OUTCOME].to_numpy(dtype=int)
    fitted = {}

    age_pipeline = make_age_baseline()
    fit_model(age_pipeline, train_features["age_only"], outcome, "age_only")
    fitted["age_only"] = age_pipeline

    for feature_set in LASSO_FEATURE_SETS:
        pipeline = make_lasso_pipeline(LASSO_TUNE_MAX_ITER)
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
        penalty = "not applicable" if name == "age_only" else penalties[name]
        values = probabilities.to_numpy(dtype=float)
        summaries.append(
            {
                "model": PREDICTION_FILES[name].stem,
                "n_fit": len(train_features[name]),
                "n_predictors": int(train_features[name].shape[1]),
                "C": penalty,
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
            f"C={summaries[-1]['C']}, predicted={summaries[-1]['n_predicted']}, "
            f"range=[{values.min():.10f}, {values.max():.10f}], "
            f"warnings=0, max_reread_abs_diff={max_difference:.3g}"
        )
    print("Git ignore check passed for all four prediction files.")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-values-lasso",
        action="store_true",
        help="Run the two-candidate values LASSO check on the shared training folds.",
    )
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
    print(f"scikit-learn {sklearn.__version__}")
    if args.tune_lasso:
        run_lasso_tuning()
        return
    if args.check_values_lasso:
        run_values_lasso_check()
        return
    if args.predict_test:
        run_test_predictions()
        return
    result = run_age_baseline_fold_check(load_aligned_inputs())
    print(f"Predictors: {result['predictors']}")
    print(f"Fit stays: {result['n_fit']}; deaths: {result['fit_deaths']}")
    print(
        "Validation stays: "
        f"{result['n_validation']}; deaths: {result['validation_deaths']}"
    )
    print(f"Fit/validation ID intersection: {result['n_intersection']}")
    print(f"Missing age in fit: {result['fit_age_missing']}")
    print(f"Missing age in validation: {result['validation_age_missing']}")
    print(f"Learned age median (years): {result['learned_age_median']}")
    print(
        "Penalty setting: "
        f"no penalty (C={result['C']}, l1_ratio={result['l1_ratio']}), "
        f"solver={result['solver']!r}, fit_intercept={result['fit_intercept']}"
    )
    print(f"classes_: {result['classes']}")
    print(
        "Validation probability range: "
        f"[{result['probability_min']:.10f}, {result['probability_max']:.10f}]"
    )
    print(f"Warnings: {len(result['warnings'])}")
    for message in result["warnings"]:
        print(message)


if __name__ == "__main__":
    main()
