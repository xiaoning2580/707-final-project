# Data Reproduction and Modeling Handoff

This guide explains how to reproduce the prepared data and use the
shared modeling inputs.

For cleaning rules and methodological details, see
[data_preparation.md](data_preparation.md).

## 1. Run Commands from the Repository Root

The working directory should be the directory containing:

- scripts/
- data/
- README.md

Use the team's agreed Python environment.

The preparation and EDA scripts require:

- numpy
- pandas
- scikit-learn
- matplotlib

If these packages are not already installed:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

For exact reproducibility, the team should record and use the same
package versions. This installation command alone does not pin versions.

## 2. Download the Source Data

```bash
python scripts/download_data.py
```

Expected local inputs include:

```text
data/set-a/
data/Outcomes-a.txt
```

Do not manually edit the downloaded measurement files.

## 3. Confirm the Current Cleaning Version

The current feature-building script must contain:

```text
BP_RULE_B_V1
```

You can check this with:

```bash
python -c "from pathlib import Path; p = Path('scripts/06_build_features.py'); assert 'BP_RULE_B_V1' in p.read_text(), 'Blood-pressure rule B is missing'; print('BP_RULE_B_V1 found')"
```

This checks the version marker; it does not independently validate
the full implementation.

If the marker is missing, obtain the updated feature-building script
from the shared project branch before proceeding.

## 4. Build Features

```bash
python scripts/06_build_features.py
```

Expected headline results:

| Item | Expected value |
|------|---------------:|
| ICU stays | 4,000 |
| Deaths | 554 |
| Time-series variables | 37 |
| Value predictors | 111 |
| Pattern predictors | 74 |
| Combined predictors | 185 |
| Training stays | 3,200 |
| Test stays | 800 |
| Excluded -1 records | 2,223 |
| Excluded temperature records below 10 | 129 |
| Excluded zero blood-pressure records | 1,718 |
| Additional blood-pressure context exclusions | 384 |

Existing valid split assignments are reused.

Do not delete split_assignments.csv to generate a different split.
If the file is absent, the script creates assignments using seed 707;
these should be verified against the team's agreed assignments before
modeling.

## 5. Generate Training EDA

```bash
python scripts/07_training_eda.py
```

Outputs are written to:

```text
results/tables/training_eda/
results/figures/training_eda/
```

The script analyzes the training set only. It does not alter features
or split assignments.

Figures are exported in PNG and PDF formats.

Use the figures regenerated after blood-pressure rule B was applied.
Earlier blood-pressure figures describe an older feature version.

## 6. Prepared Files

All files below are under data/processed/.

| File | Purpose |
|------|---------|
| features_values.csv | 111 value predictors: mean, min, max |
| features_patterns.csv | 74 recording-count and missingness predictors |
| features_combined.csv | 185 combined predictors |
| baseline.csv | Age input for the baseline |
| static_descriptors.csv | Static descriptors for sample description |
| outcomes.csv | Outcome records indexed by RecordID |
| split_assignments.csv | Shared train/test and training validation-fold assignments |
| feature_dictionary.csv | Feature definitions |
| feature_sets.json | Feature-set metadata |
| preparation_manifest.json | Preparation settings and summary counts |

Read RecordID consistently across files and align tables by that key.

Never assume that row positions match between independently loaded
or transformed tables.

RecordID, split labels, fold labels, and outcome columns must not
be passed to an estimator as predictors.

## 7. Shared Split and Fold Rules

All models must use the supplied assignments.

For cross-validation fold k:

- Validation data: training-set stays assigned to fold k.
- Fitting data: training-set stays assigned to the other four folds.
- Test data: never included in either subset.

Within each fitting fold, fit:

- Missing-value imputation.
- Standardization, where needed.
- Constant-feature removal or other feature filtering.
- Any other learned preprocessing.

Use validation-fold performance to select hyperparameters.

After selection, refit the complete pipeline on all 3,200 training
stays and predict probabilities for the same 800 test stays.

Do not fit preprocessing on the full dataset before cross-validation.

## 8. Feature-Set Rules

### Values only

Use features_values.csv.

Missing values require the agreed fold-fitted handling strategy.
Do not add explicit missingness indicators to this comparison.

Avoid silently using native missing-value routing as a substitute
for the agreed imputation strategy: this may give a model an
additional way to use missingness.

### Patterns only

Use features_patterns.csv.

A count of zero is a meaningful observed feature value.
It must not be converted to missing.

A missingness indicator of one means that no valid record remains
for the corresponding variable after cleaning.

### Combined

Use features_combined.csv.

This contains the value features together with explicit count and
missingness features.

### Age-only baseline

Use Age from baseline.csv.

Do not add other static descriptors to the baseline or main feature
sets without a documented group decision.

## 9. Special Cases

### MechVent value features

MechVent_mean, MechVent_min, and MechVent_max are constant among
observed values in the full training set.

Depending on the imputation strategy, these columns may become
constant after imputation.

Handle constant and all-missing columns within each fitting fold.
Do not manually drop patients because these columns are missing.

### Stays without valid time-series records

Three training stays have no valid time-series records.

These stays remain in the analysis. Their value features are missing,
their recording counts are zero, and their missingness indicators
are one.

### Repeated timestamps

Counts include all retained valid rows, including multiple rows for
the same variable at the same timestamp.

Do not change to unique-timestamp counts in only one model.
Any alternative count definition requires a separate, consistently
implemented analysis.

## 10. Prediction Handoff

For evaluation, each model should supply a table with:

| Column | Meaning |
|--------|---------|
| RecordID | ICU stay identifier |
| predicted_probability | Predicted probability of in-hospital death |

This is the recommended output format for the modeling stage;
these prediction files are not created by the preparation scripts.

Each test prediction file should contain:

- Exactly 800 unique RecordIDs.
- Exactly the IDs assigned to the shared test set.
- One finite probability per stay.
- Probabilities between 0 and 1.

Use probabilities rather than thresholded class predictions for
AUROC, calibration, and Brier-score evaluation.

Include the model name, feature set, selected hyperparameters,
preprocessing strategy, and data preparation version in the
accompanying model documentation.

The evaluation workflow should join predictions to outcomes by
RecordID.

## 11. Audit and Migration Scripts

Scripts 01–05 document the earlier raw-data and temperature reviews.

Script 08 documents the MAP-tail review.

Script 09 documents the comparison of candidate blood-pressure rules.
It was designed for the pre-rule-B feature version and should not
be treated as a routine post-rule-B validation command.

Script 10 applies the rule-B migration, creates a backup, rebuilds
features, checks split preservation, and regenerates EDA.

Once the updated 06_build_features.py is present, routine reproduction
uses scripts 06 and 07. Repeatedly running script 10 is not necessary
for routine use.

Supplemental migration outputs include:

- data/audit/bp_rule_B_applied_summary.csv
- data/audit/bp_rule_B_applied_by_variable.csv

Some additional audit fields are written by script 10 rather than
by script 06 itself.

## 12. Version Control and Team Reproducibility

Commit the preparation code, documentation, and agreed aggregate
tables and figures.

Keep raw data, processed patient-level data, and backups out of Git.

Each member should download the source data locally and use the same
version of the preparation scripts.

Because split_assignments.csv is under the ignored data directory,
a code commit alone does not share the actual assignments.

The team should either:

1. Share the agreed assignment file through its approved project
   file-sharing location; or
2. Independently regenerate assignments and verify that the mapping
   from RecordID to split and fold is identical.

Matching the seed alone is not sufficient verification.

Before modeling, confirm that all members use:

- The same cleaning-rule version.
- The same feature definitions.
- The same RecordID-to-split mapping.
- The same training validation folds.
- Compatible, recorded package versions.

## 13. Troubleshooting

### Missing input files

Run the download script from the repository root and confirm that
data/set-a/ and data/Outcomes-a.txt exist.

### Unexpected feature dimensions

Check the preparation manifest and script version.
Do not include RecordID when counting predictor columns.

### Split validation failure

Do not delete the assignment file simply to bypass the error.
Check whether IDs, source data, or the assignment file differ from
the agreed project version.

### Different results between group members

Compare script versions, the preparation manifest, split assignments,
feature files, and package versions before changing the analysis.

### Older figures still showing extreme MAP means

Confirm that scripts 06 and 07 completed successfully after rule B
was installed, and open the regenerated figure files.

Rule B does not remove every extreme value. A remaining extreme
value does not by itself demonstrate that reproduction failed.