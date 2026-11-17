# Regularized Logistic Models

## Methods

We predicted in-hospital death for the 4,000 ICU stays in Set A of the PhysioNet/Computing in Cardiology Challenge 2012. The outcome is `In-hospital_death`. The inputs use cleaning version `BP_RULE_B_V1` and feature version `COMMON_COVARIATES_V2`, read from `data/processed/common_covariates_v2/`. The cleaning rules are described in the [data preparation notes](data_preparation.md).

The shared split has 3,200 training stays and 800 test stays. Both this split and the five training folds are stratified by `In-hospital_death`. We reused the assignments in `split_assignments.csv` rather than drawing a new split. Tables are aligned by `RecordID`.

### Feature sets

All three feature sets include Age, Gender, and ICUType. Combined contains those common covariates only once. The counts below are raw predictor columns, before one-hot encoding.

| Feature set | Contents | Raw predictors |
|-------------|----------|---------------:|
| Values | Common covariates plus the mean, minimum, and maximum of each measurement | 114 |
| Patterns | Common covariates plus valid measurement counts and missingness indicators | 77 |
| Combined | Common covariates plus the value summaries and the measurement patterns | 188 |

Age is numeric and is kept in years. Gender and ICUType are categorical. Gender is coded 0 for female and 1 for male. ICUType is coded 1 for coronary care, 2 for cardiac surgery recovery, 3 for medical ICU, and 4 for surgical ICU. Missing values are mapped to the label `Missing`, then one-hot encoded with `drop=None` and `handle_unknown="ignore"`. A category that was not seen while fitting becomes an all-zero block for that variable, so it is not assigned to another level. The same missing-value rule is used in all three feature sets.

### Age-only baseline

The baseline is logistic regression with an intercept and no penalty. Its only predictor is Age. Missing ages would be filled with the training median, and Age is not standardized. In scikit-learn 1.9 the unpenalized model is fit with infinite `C`.

### LASSO models

Each LASSO model uses one pipeline: median imputation of the numeric columns, one-hot encoding of Gender and ICUType, removal of constant columns, standardization, and L1-penalized logistic regression with an intercept. The L1 penalty is `l1_ratio=1`. The imputer does not add extra missingness indicators. Zero measurement counts are kept as zeros, and stays with every measurement value missing remain in the analysis.

Inside cross-validation, the imputer, category encoder, constant-column filter, scaler, and LASSO coefficients are fit on the four fitting folds and then applied to the held-out fold. Standardization is applied to every column that remains after the constant filter, including the one-hot columns. The same pipeline is used for Values, Patterns, and Combined.

Median imputation can leave a recognizable pattern where a measurement was missing, so the Values model may still use measurement missingness indirectly. It does not contain explicit missingness indicators.

We searched `C = [0.001, 0.01, 0.1, 1, 10]`. Smaller values impose a stronger penalty. For each feature set, the selected C is the one with the highest mean validation AUROC across the five training folds. An exact tie uses the smaller C. The solver is `saga`, with `random_state=707` and `max_iter=20000`.

## Cross-validation Results

The scores in this section are training-set cross-validation results. They are not test-set performance. All three models selected `C=0.1`. The mean five-fold validation AUROCs are about 0.8281 for Values, 0.7574 for Patterns, and 0.8361 for Combined. Unrounded scores for every candidate are in [lasso_cv_results.csv](../results/tables/regularized_models/common_covariates_v2/lasso_cv_results.csv).

| Feature set | Selected C | Mean five-fold validation AUROC |
|-------------|-----------:|--------------------------------:|
| Values | 0.1 | 0.8281 |
| Patterns | 0.1 | 0.7574 |
| Combined | 0.1 | 0.8361 |

After choosing C, we refit each LASSO pipeline and the age baseline on all 3,200 training stays. Each fitted model then predicted the probability of in-hospital death for the same 800 test stays. Test outcomes were not used to fit the models or to choose C.

## Running the Models

First prepare the local data by following the [data usage guide](data_usage.md). From the repository root, run the tuning step followed by the final fitting and prediction step after activating the [shared environment](environment.md):

```bash
python scripts/11_train_regularized_models.py --tune-lasso
python scripts/11_train_regularized_models.py --predict-test
```

The prediction command reads the selected C values from the tuning table linked above. The random-forest models should use this same feature version, the same Gender and ICUType coding, and the same `RecordID` assignments in `data/processed/common_covariates_v2/split_assignments.csv`.

These results were produced with Python 3.11.18. The six main analysis packages and their supporting dependencies are pinned in the shared [requirements.txt](../requirements.txt). See the [environment guide](environment.md) for versions and installation.

## Prediction Files

The prediction command writes four files in `data/predictions/common_covariates_v2/`.

| File | Model |
|------|-------|
| `age_only.csv` | Age-only logistic regression |
| `lasso_values.csv` | LASSO using value summaries |
| `lasso_patterns.csv` | LASSO using measurement patterns |
| `lasso_combined.csv` | LASSO using both feature types |

Each file has two columns, `RecordID` and `predicted_probability`, and covers the same 800 test stays. The `data/` directory is ignored by Git, so these patient-level files should be sent privately to the group member responsible for evaluation. That evaluation, including test performance and paired bootstrap comparisons, is done separately. Differences between feature sets can reflect both patient status and hospital measurement habits; this model-fitting step does not establish that distinction as an external validation.
