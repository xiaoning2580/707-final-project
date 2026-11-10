# Regularized Logistic Models

## Methods

We used the 4,000 ICU stays in Set A of the PhysioNet/Computing in Cardiology Challenge 2012 to predict in-hospital death. The outcome was `In-hospital_death`, with 1 indicating death. We used the prepared features from cleaning version `BP_RULE_B_V1`. Details of the cleaning rules are in the [data preparation notes](data_preparation.md).

All four models used the same 3,200 training stays and 800 test stays. We matched the input tables by `RecordID` and used the five training folds already assigned in `split_assignments.csv`. Preprocessing and model selection used the training data only.

### Age-only baseline

The baseline was logistic regression with an intercept and no penalty, using `Age` as its only predictor. Age was kept in years. We included a median imputer for missing ages, although there were no missing ages in the prepared data.

### LASSO models

We fitted a separate LASSO logistic regression for each of the three feature sets defined in `feature_sets.json`.

| Feature set | Features | Number of input predictors |
|-------------|----------|---------------------------:|
| Values | Mean, minimum, and maximum of each measurement variable | 111 |
| Patterns | Valid measurement counts and missingness indicators | 74 |
| Combined | Both value summaries and measurement patterns | 185 |

For each model, we used a pipeline containing median imputation, removal of constant columns, standardization, and logistic regression with an L1 penalty and an intercept. The imputer added no extra missingness indicators. Zero measurement counts were retained, and stays with all value features missing remained in the analysis.

During cross-validation, each preprocessing step was fitted on the four fitting folds and applied to the remaining validation fold. This included the imputation medians, the choice of constant columns to remove, and the scaling parameters. We used the same procedure for all three feature sets.

We searched over `C = [0.001, 0.01, 0.1, 1, 10]`, where smaller values give stronger regularization. Each model's C was selected by its highest mean validation AUROC across the five shared folds. The selection rule used the smaller C in an exact tie. We used the `saga` solver with `random_state=707` and `max_iter=20000`. The final search completed without convergence warnings.

After selecting C, we refitted each full pipeline on all 3,200 training stays. We also fitted the age baseline on the full training set. Each model then produced a probability of in-hospital death for each test stay.

## Cross-validation Results

All three LASSO models selected `C=0.1`. The table shows mean five-fold validation AUROC within the training set. Full results for all candidate values are saved in [lasso_cv_results.csv](../results/tables/regularized_models/lasso_cv_results.csv).

| Feature set | Selected C | Mean five-fold validation AUROC |
|-------------|-----------:|-------------------------------:|
| Values | 0.1 | 0.8179 |
| Patterns | 0.1 | 0.7133 |
| Combined | 0.1 | 0.8288 |

## Running the Models

First prepare the local data by following the [data usage guide](data_usage.md). From the repository root, run the tuning step followed by the final fitting and prediction step.

```bash
python scripts/11_train_regularized_models.py --tune-lasso
python scripts/11_train_regularized_models.py --predict-test
```

The second command reads the selected C values from the tuning table. All models must use the same `split_assignments.csv`, including the random-forest models.

The saved results were produced with Python 3.13.5 and scikit-learn 1.6.1. The versions currently listed in `requirements.txt` differ from this environment, including scikit-learn 1.9.1, numpy 2.4.6, and pandas 3.1.0. Use the recorded environment when reproducing these results.

## Prediction Files

The prediction command saves four files in `data/predictions/`.

| File | Model |
|------|-------|
| `age_only.csv` | Age-only logistic regression |
| `lasso_values.csv` | LASSO using value summaries |
| `lasso_patterns.csv` | LASSO using measurement patterns |
| `lasso_combined.csv` | LASSO using both feature types |

Each file contains `RecordID` and `predicted_probability`. All four files were checked to cover the same 800 unique test IDs, with finite probabilities between 0 and 1. The evaluation code should join these predictions to `data/processed/outcomes.csv` by `RecordID` to calculate test performance.

The `data/` directory is ignored by Git, so the prediction files need to be shared separately with the group member responsible for evaluation.
