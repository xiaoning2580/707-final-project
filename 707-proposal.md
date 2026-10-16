# Do Measurement Patterns Help Predict In-Hospital Mortality in ICU Patients?

## 1. Clinical Question

Our project will ask how much information about in-hospital mortality comes from whether and how often a clinical variable was measured, compared with its recorded values. We will use measurements from the first 48 hours of an ICU stay.

We will compare three feature sets: measurement values only, measurement counts and missingness indicators only, and both together. This comparison will help us examine whether measurement patterns add useful information beyond the values themselves.

## 2. Cohort and Outcome

We will use Set A of the PhysioNet/Computing in Cardiology Challenge 2012 dataset, version 1.0.0. The data are publicly accessible under the Open Data Commons Attribution License v1.0. The source is https://physionet.org/content/challenge-2012/1.0.0/.

Our cohort will consist of the 4,000 adult ICU stays in Set A. The source dataset excludes ICU stays shorter than 48 hours. We have downloaded and loaded the data and confirmed that the outcome and measurement records are available.

The outcome will be In-hospital_death, coded as 1 for death during hospitalization and 0 for survival. We will make predictions using only information recorded during the first 48 hours. RecordID will be used for matching and splitting, not as a predictor. Length_of_stay, Survival, SAPS-I, and SOFA will not be included as predictors.

For each time-varying variable, we will summarize valid measurements within the observation window using the mean, minimum, and maximum. For the measurement-pattern feature set, we will count valid observations and create an indicator for whether the variable has no valid observations. The combined feature set will include both types of features. Values coded as −1 will be treated as missing.

## 3. Validation Design

We will first examine the outcome distribution, measurement distributions, missingness, and measurement counts. We will then create a stratified 80% training and 20% test split by RecordID, keeping each stay entirely within one split. The same split will be used for all three feature sets.

Within the training set, we will use stratified five-fold cross-validation to select model settings. Missing measurement summaries will be filled with training-fold medians, and predictors for the regularized model will be standardized using training-fold information. These steps will be repeated inside each fold to avoid leakage. The values-only models will not receive explicit measurement counts or missingness indicators.

We will fit LASSO logistic regression and a random forest for each feature set. This provides one regularized model and one tree-based model while keeping the model classes consistent across comparisons. As a simple baseline, we will use logistic regression with age alone.

After selecting model settings, we will fit the models on the full training set and evaluate them once on the held-out test set. We will report AUROC for discrimination and calibration plots and Brier scores for calibration and probability prediction accuracy. We will also report the differences in AUROC and Brier score between the combined and values-only models to assess the additional predictive value of measurement patterns.

## 4. Expected Findings and Limitations

We expect measurement values to predict mortality, but we also expect measurement counts and missingness indicators to contain useful information. One possible explanation is that patients who appear more seriously ill receive more frequent measurements. We therefore expect the combined feature set to perform better than values alone, although the size of the improvement is unknown.

If measurement patterns alone perform well, we will interpret this as evidence of predictive information in the recording process, not evidence that more testing causes death. We will also discuss how different testing practices could affect performance when a model is used at another hospital.

Our analysis will be limited to Set A and to patients with ICU stays of at least 48 hours. It will not directly establish performance in other hospitals or in shorter stays. Also, imputing missing values may leave some indirect information about missingness in the values-only models, so the three feature sets will not perfectly separate physiology from measurement practices.

References: Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting In-Hospital Mortality of Patients in ICU: The PhysioNet/Computing in Cardiology Challenge 2012. Computing in Cardiology. 2012;39:245–248.

Pollard T, et al. PhysioNet as a global platform for biomedical research. Nature Health. 2026;1:792–795.
