# Modeling Handoff: COMMON_COVARIATES_V2

## Current Inputs

Use files under:

    data/processed/common_covariates_v2/

Do not use the measurement-only feature tables in the parent directory
for the revised main models.

| File | Predictor columns before encoding |
|------|----------------------------------:|
| features_values.csv | 114 |
| features_patterns.csv | 77 |
| features_combined.csv | 188 |

All three feature sets include Age, Gender, and ICUType.
RecordID must be used for alignment only.

## Common-Covariate Processing Convention

Use the same convention in the LASSO and random forest workflows:

| Variable | Interpretation | Processing |
|----------|----------------|------------|
| Age | Numeric age | Fold-fitted median imputation if needed |
| Gender | Categorical: 0=female, 1=male | Map missing values to Missing; one-hot encode |
| ICUType | Categorical: 1=coronary care, 2=cardiac surgery recovery, 3=medical ICU, 4=surgical ICU | Map missing values to Missing; one-hot encode |

Do not treat ICUType as a continuous or ordinal numeric predictor.

Fit categorical encoders within the modeling pipeline and handle
unseen categories without failing. Document the encoder settings,
including whether a reference category is dropped.

Any learned numerical imputation, scaling, or feature filtering must
be fitted using only the fitting subset of each cross-validation fold.

The prepared CSV files preserve missing values and original category
codes. The processing described above is performed by the model code.

Training-set missingness is:
- Age: 0.
- Gender: 3.
- ICUType: 0.

## Feature-Set Checks

The Patterns measurement-count and missingness columns contain no
missing values. Its common covariates may contain missing values.

Do not require the entire revised Patterns table to be complete.

Combined contains common covariates only once:

    Combined = common covariates + value features + pattern features

Do not concatenate the revised Values and Patterns tables directly,
because this duplicates Age, Gender, and ICUType.

The feature_sets.json schema is:

    feature_sets -> values/patterns/combined -> columns

Each feature set also provides numeric_columns and categorical_columns.

Validate the input columns against this metadata. Predictor counts
after one-hot encoding may differ from the raw CSV counts.

## Shared Split and Cross-Validation

Use the supplied split_assignments.csv.

- Training: 3,200 stays.
- Test: 800 stays.
- Training cross-validation: five folds of 640 validation stays each.
- Stratification variable for both steps: In-hospital_death.

The revision copied the original assignments byte-for-byte.
Do not generate new splits.

Align features, outcomes, and assignments by RecordID before modeling.

For fold k, fit on training stays with cv_fold != k and validate on
training stays with cv_fold == k. Never include test stays in tuning.

## Required Model Updates

Before rerunning models:

1. Change the input directory to common_covariates_v2.
2. Allow Age, Gender, and ICUType in all three main feature sets.
3. Update raw feature-count and metadata checks.
4. Add categorical preprocessing for Gender and ICUType.
5. Update Patterns missing-value checks.
6. Update Combined consistency checks to avoid duplicate common variables.
7. Repeat hyperparameter tuning using the existing five folds.
8. Refit on the full training set and export test probabilities.

Do not reuse the old best hyperparameters without retuning.

The age-only baseline definition is unchanged. Its predictions may be
reused only after verifying unchanged inputs, assignments, and the
baseline modeling procedure.

## Prediction Handoff

Provide one probability of in-hospital death per test RecordID.

Each prediction file should contain:

- RecordID.
- predicted_probability.

Check that each file has exactly the same 800 unique test RecordIDs
and finite probabilities between 0 and 1.

Record the feature version COMMON_COVARIATES_V2 in model documentation
and distinguish revised predictions from earlier model outputs.

## Interpretation

All three main feature sets contain the same common covariates.

The Values model has no explicit measurement-missingness indicators.
However, median imputation may leave recognizable patterns, and value
summaries may also reflect the recording process. Values should not
be described as completely free of missingness information.

The Missing category for a common covariate is available consistently
across all three feature sets.

## Reproduction

From the repository root:

```bash
python scripts/download_data.py
python scripts/06_build_features.py
python scripts/07_training_eda.py
python scripts/13_add_common_covariates.py
```

Script 06 must contain blood-pressure rule BP_RULE_B_V1.

Preserve the agreed split_assignments.csv when reproducing the data.

Script 13 refuses to overwrite an existing common_covariates_v2
directory. Archive that directory before intentionally rebuilding it.
After rebuilding the source features, rebuild the revised directory
before fitting models.

Script 12 is a backup and pre-update inspection utility, not a required
step for routine reproduction.

## Validation Files

The revised directory includes:

- validation_summary.json: revision checks.
- preparation_manifest.json: preparation settings and provenance.
- feature_dictionary.csv: definitions, including common covariates.
- feature_sets.json: explicit model-input columns.
- file_checksums.json: output-file SHA-256 checksums.
- training_common_covariate_missingness.csv: training missingness.
- training_cv_category_counts.csv: category counts in each fold's
  fitting and validation subsets.

Patient-level files remain under the Git-ignored data directory.
Share the agreed assignments and verify that all modeling members
are using the same version of the prepared inputs.