# 707-final-project

## Project Goals

Our project will examine how much information about in-hospital mortality comes from whether and how often a clinical variable was measured, compared with its recorded values.

Using the first 48 hours of ICU data, we will compare three feature sets:

1. Measurement values only.
2. Measurement counts and missingness indicators only.
3. Measurement values, counts, and missingness indicators together.

We will evaluate discrimination and calibration and discuss whether differences in hospital measurement practices could affect model performance.

## Group Members and Responsibilities

| Member | Email | Responsibilities |
| --- | --- | --- |
| Yixuan Liu | yixuan.liu3@duke.edu | Data preparation and exploratory analysis: load and match measurement and outcome records, handle missing-value codes, examine measurement distributions and missingness, and construct the three feature sets. Provide the data preparation code, exploratory figures, and corresponding explanations. |
| Kaiyan Chen | kaiyan.chen@duke.edu | Regularized models and baseline: build an age-only logistic regression baseline and LASSO logistic regression models for the three feature sets. Perform imputation, standardization, cross-validation, and tuning within the training data. Provide model code, test-set predicted probabilities, and method explanations. |
| Mai Li | mai.li@duke.edu | Tree-based models: build random forest models for the three feature sets. Perform missing-data handling, cross-validation, and tuning within the training data. Provide model code, test-set predicted probabilities, and method explanations. |
| Xiaoning Li | xiaoning.li@duke.edu | Model evaluation and report integration: write evaluation code to compare AUROC, calibration, and Brier scores; produce comparison tables and figures; discuss findings and limitations; and integrate contributions into the report and README, including running instructions, data source, license, and citations. |

All models will use the same training/test split and cross-validation folds. Each member will write explanations for their own work, commit their contributions using their own GitHub account, and submit an individual contribution statement. The group will review the final work together.

## Data Source and License

We will use Set A of the PhysioNet/Computing in Cardiology Challenge 2012 dataset, version 1.0.0. It contains 4,000 adult ICU stays. The outcome is in-hospital death.

- Original source: [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
- Data license: Open Data Commons Attribution License v1.0 (ODC-BY).
- Required files: `set-a.tar.gz` and `Outcomes-a.txt`, available from the original source.

## Data Access

From the repository root, run:

    python3 scripts/download_data.py

The script downloads the original files from PhysioNet and extracts the measurement records into `data/set-a/`. The outcome file is saved as `data/Outcomes-a.txt`.

The `data/` directory is excluded from version control through `.gitignore`. Each group member should run the script to obtain a local copy of the data.

## Planned Analysis

We will examine the data distributions and missingness, then compare the three feature sets using LASSO logistic regression and a random forest. An age-only logistic regression model will serve as a simple baseline.

We will use a stratified training/test split and cross-validation within the training set. Imputation and standardization will be fitted using training data only. Model evaluation will include AUROC, calibration plots, and Brier scores. We will discuss limitations, including whether differences in hospital measurement practices could affect model performance.

## Proposal

The project proposal is available in [707-proposal.md](707-proposal.md).

## Data Citations

Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting In-Hospital Mortality of Patients in ICU: The PhysioNet/Computing in Cardiology Challenge 2012. *Computing in Cardiology*. 2012;39:245–248.

Pollard T, Moody B, Lehman L, et al. PhysioNet as a global platform for biomedical research. *Nature Health*. 2026;1:792–795. https://doi.org/10.1038/s44360-026-00096-z
