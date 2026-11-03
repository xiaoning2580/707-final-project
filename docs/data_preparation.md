# Data Preparation and Exploratory Analysis

Prepared by: Yixuan Liu

## 1. Objective

This project examines how much predictive information about in-hospital
mortality is contained in clinical measurement values versus recording
counts and missingness patterns.

We construct three feature sets from the first 48 hours of ICU data:

1. Values only.
2. Recording counts and missingness indicators only.
3. Values, recording counts, and missingness indicators combined.

These feature sets will be compared using LASSO logistic regression and
random forest models. An age-only logistic regression model will provide
a simple baseline.

## 2. Data Source and Study Population

We use Set A of the PhysioNet/Computing in Cardiology Challenge 2012,
version 1.0.0.

Source:
https://physionet.org/content/challenge-2012/1.0.0/

Required source files:

- set-a.tar.gz: individual ICU measurement records.
- Outcomes-a.txt: outcomes indexed by RecordID.

The dataset contains 4,000 adult ICU stays. The outcome is in-hospital
death:

- Survived: 3,446 stays.
- Died: 554 stays.
- Overall mortality: 13.85%.

All 4,000 stays were retained. Measurement records and outcomes were
matched using RecordID. No unmatched measurement or outcome IDs were
found.

The data are distributed under the Open Data Commons Attribution
License v1.0 (ODC-BY). Raw and processed patient-level data are excluded
from version control.

## 3. Observation Window and Structural Checks

The feature window includes measurements with elapsed times from
0 through 2,880 minutes, inclusive.

Times are interpreted as elapsed hours and minutes rather than
clock times, allowing hour values above 23.

Structural checks confirmed:

- All 4,000 measurement files could be parsed.
- Outcome IDs were unique and matched the measurement records.
- No measurement rows were outside the specified observation window.
- No exact duplicate rows were found.
- There were 2,223 values coded as -1.
- There were 24 other negative values, all for temperature.

RecordID is used only for matching and alignment and is not a predictor.

## 4. Cleaning Rules

Original source files are preserved. Cleaning is applied when
constructing derived features.

### 4.1 Missing-value codes

Values coded as -1 are excluded from valid measurement summaries
and counts.

An excluded record does not contribute to the mean, minimum, maximum,
or valid-record count.

### 4.2 Temperature

The temperature review identified negative values, zeros, and a cluster
of unusually low positive values.

The implemented rule excludes:

    Temp < 10 degrees Celsius

This is an operational data-quality rule, not a clinical reference range.
It does not establish that every retained temperature is accurate.

The rule removes 129 temperature records, leaving 86,276 valid
temperature records.

The recorded value of 24.2 degrees Celsius and high positive values,
including 42.1 degrees Celsius, are retained.

After cleaning:

- 3,936 stays have at least one valid temperature record.
- 64 stays have no valid temperature record.
- No stay becomes entirely missing for temperature because of this rule.

### 4.3 Blood pressure: Rule B

The main analysis uses blood-pressure rule B, recorded as:

    BP_RULE_B_V1

The rule applies to the following variables:

- MAP, SysABP, DiasABP.
- NIMAP, NISysABP, NIDiasABP.

First, zero-valued records for all six variables are excluded.

Second, a MAP or NIMAP record is excluded when its corresponding
systolic and diastolic records at the same elapsed minute are both zero:

- MAP uses SysABP and DiasABP.
- NIMAP uses NISysABP and NIDiasABP.

This context rule is applied only when there is exactly one
nonnegative systolic record and exactly one nonnegative diastolic
record for the relevant family at that time. Ambiguous contexts with
multiple such records are not used to trigger the rule.

The zero-valued context is identified before zero records are removed.
Exclusion categories are mutually exclusive: a zero-valued record is
counted under the zero rule rather than counted again under the
context rule.

| Population | Zero records excluded | Additional context exclusions | Total excluded | Affected stays |
|------------|----------------------:|------------------------------:|---------------:|---------------:|
| All stays  | 1,718 | 384 | 2,102 | 423 |
| Training   | 1,546 | 343 | 1,889 | 359 |
| Test       | 172 | 41 | 213 | 64 |

Rule B was selected following review of training-set blood-pressure
records and then applied consistently to the full dataset. The review
did not use mortality outcomes.

This rule is a data-quality assumption. It does not prove that every
excluded mean-pressure record is erroneous or that every retained
record is valid.

The main analysis does not apply a blanket MAP/NIMAP threshold of
less than 10 or greater than 200. Such thresholds were considered
as an alternative cleaning rule, not adopted as the main rule.

### 4.4 Records sharing a variable and timestamp

The raw-data audit identified:

- 3,730 repeated variable-time groups.
- 7,474 records belonging to those groups.
- 3,744 records beyond the first record in each group.
- 1,167 affected ICU stays.

These are repeated timestamps, not necessarily duplicate observations.
No exact duplicate rows were found.

All otherwise valid records are retained, including different values
recorded for the same variable at the same time.

Therefore:

- Means are calculated across valid records.
- Multiple records at one timestamp can each contribute to the mean.
- Counts represent valid recorded rows, not unique timestamps or
  verified independent clinical tests.

## 5. Feature Construction

Each row in a feature table represents one ICU stay.

For each of 37 time-series variables, we construct:

| Feature | Definition |
|---------|------------|
| mean | Arithmetic mean of all valid records in the window |
| min | Minimum valid recorded value |
| max | Maximum valid recorded value |
| count | Number of valid recorded rows |
| missing | 1 if the valid-record count is zero; otherwise 0 |

If a stay has no valid record for a variable:

- mean, min, and max remain missing.
- count is 0.
- missing is 1.

If only one valid record is available, mean, min, and max are equal.

The three feature sets are:

| Feature set | Contents | Predictor columns |
|-------------|----------|------------------:|
| Values | mean, min, max for 37 variables | 111 |
| Patterns | count and missing for 37 variables | 74 |
| Combined | All value and pattern features | 185 |

RecordID is not included in these predictor counts.

No imputation or standardization is performed during feature
construction.

### Static descriptors and baseline

Age, Gender, Height, and ICUType are stored separately from the three
main feature sets.

Age is supplied for the age-only baseline. There are no missing
age values in the prepared dataset.

Weight is treated as a time-series variable.

Outcome-related fields and severity scores, including in-hospital
death, length of stay, survival, SAPS-I, and SOFA, are not included
in the main predictor sets.

## 6. Shared Training/Test Split and Cross-Validation

A stratified 80%/20% training/test split is used.

| Split | Stays | Deaths | Mortality |
|-------|------:|-------:|----------:|
| Training | 3,200 | 443 | 13.84375% |
| Test | 800 | 111 | 13.87500% |

The training set is divided into five shared stratified
cross-validation folds.

| Validation fold | Stays | Deaths |
|-----------------|------:|-------:|
| 1 | 640 | 88 |
| 2 | 640 | 88 |
| 3 | 640 | 89 |
| 4 | 640 | 89 |
| 5 | 640 | 89 |

The random seed used when creating new assignments is 707.

Existing split assignments are reused if they pass validation.
Applying blood-pressure rule B did not change any training/test
or cross-validation assignments.

All group members must use the same saved assignments rather than
independently generating new splits.

## 7. Training-Set Exploratory Analysis

Outcome-stratified exploratory analysis uses only the 3,200
training stays.

Outputs include:

- A training sample description.
- Variable-level missingness.
- Valid-record count distributions.
- Value-feature distributions.
- Missingness comparisons by outcome.
- Empirical cumulative distribution plots by outcome.
- Checks for all-missing and constant-among-observed features.

Selected findings include:

- TroponinI and Cholesterol have high training-set missingness:
  approximately 94.9% and 92.5%, respectively.
- Three training stays have no valid time-series records; these stays
  are retained.
- No value-feature column is entirely missing in the full training set.
- MechVent_mean, MechVent_min, and MechVent_max each have only one
  distinct observed value, with 1,154 missing values.

A column that is not entirely missing in the full training set may
still be entirely missing in an individual cross-validation
training fold. Modeling pipelines must handle this possibility.

Missingness refers to absence of a valid record after cleaning.
It does not necessarily mean that a measurement was never attempted.

EDA results are descriptive. Differences between outcome groups do
not establish causality or demonstrate incremental predictive
performance; that requires the planned model comparisons.

## 8. Modeling Handoff

The prepared feature tables intentionally retain missing values.

Each modeling workflow should:

1. Align features, outcomes, and assignments by RecordID.
2. Use the supplied training/test split and validation folds.
3. Fit imputation, scaling, feature filtering, and tuning within
   cross-validation training folds.
4. Refit the selected pipeline using the full training set.
5. Evaluate the final model on the held-out test set.

For the values-only comparison, do not automatically add missingness
indicator columns or allow a separate missingness-aware modeling
path without documenting how this changes the comparison.

If imputation makes MechVent value features constant, any constant
feature filtering should be fitted within the modeling pipeline.

The pattern missingness indicator is deterministically related to
the count:

    missing = 1(count == 0)

The pattern set therefore represents the combined recording pattern;
it does not identify separate independent contributions of counts
and missingness indicators.

## 9. Interpretation and Limitations

- Valid-record counts reflect both recording practices and cleaning
  decisions. They are not direct counts of independent tests.
- Repeated timestamps may reflect multiple recordings or source-system
  behavior.
- Retained records may still contain artifacts.
- Initial structural and temperature audits used the available raw
  dataset. Subsequent blood-pressure review and outcome-stratified
  EDA used the training set.
- Values-only summaries are not guaranteed to contain no information
  about the observation process: sampling frequency can affect
  minima, maxima, and means, and imputation can leave recognizable
  patterns.
- Observation duration, treatment, illness severity, and documentation
  practices can all influence recording counts.
- Features covering the first 48 hours are not an admission-time
  prediction model. Prediction timing and stays with shorter
  observation periods should be discussed in the final report.
- A random split within Set A does not establish performance at a
  different hospital. Hospital-practice effects remain a limitation
  unless evaluated with an appropriate external or site-based design.

## 10. Reproducibility

The main feature and EDA workflow is:

    python scripts/download_data.py
    python scripts/06_build_features.py
    python scripts/07_training_eda.py

The feature-building script must contain the BP_RULE_B_V1 implementation.

The migration script, 10_apply_bp_rule_b.py, was used to install
the rule, back up previous outputs, rebuild features, verify unchanged
assignments, and regenerate EDA.

Some supplemental audit columns and verification flags are written by
the migration script. They are not necessarily regenerated by running
06_build_features.py alone.

## Reference

Silva I, Moody G, Scott DJ, Celi LA, Mark RG.
Predicting In-Hospital Mortality of Patients in ICU:
The PhysioNet/Computing in Cardiology Challenge 2012.
Computing in Cardiology. 2012;39:245–248.