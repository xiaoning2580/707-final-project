# 707-final-project

## Group Members

- Kaiyan Chen
- Mai Li
- Xiaoning Li
- Yixuan Liu

## Project Question

How much information about in-hospital mortality comes from whether and how often a clinical variable was measured, compared with its recorded values?

Using the first 48 hours of ICU data, we will compare three feature sets:

1. Measurement values only.
2. Measurement counts and missingness indicators only.
3. Measurement values, counts, and missingness indicators together.

## Data Source and License

We will use Set A of the PhysioNet/Computing in Cardiology Challenge 2012 dataset, version 1.0.0. It contains 4,000 adult ICU stays. The outcome is in-hospital death.

- Original source: [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
- Data license: Open Data Commons Attribution License v1.0 (ODC-BY).
- Required files: `set-a.tar.gz` and `Outcomes-a.txt`, available from the original source.

## Data Access

The repository will include a script to download the data from PhysioNet. Large data files will not be committed to this repository.

## Planned Analysis

We will examine the data distributions and missingness, then compare the three feature sets using LASSO logistic regression and a random forest. An age-only logistic regression model will serve as a simple baseline.

We will use a stratified training/test split and cross-validation within the training set. Imputation and standardization will be fitted using training data only. Model evaluation will include AUROC, calibration plots, and Brier scores. We will discuss limitations, including whether differences in hospital measurement practices could affect model performance.

## Proposal

The project proposal is available in [707-proposal.md](707-proposal.md).

## Data Citations

Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting In-Hospital Mortality of Patients in ICU: The PhysioNet/Computing in Cardiology Challenge 2012. *Computing in Cardiology*. 2012;39:245–248.

Pollard T, Moody B, Lehman L, et al. PhysioNet as a global platform for biomedical research. *Nature Health*. 2026;1:792–795. https://doi.org/10.1038/s44360-026-00096-z
