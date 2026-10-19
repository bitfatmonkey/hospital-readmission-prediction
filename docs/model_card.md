# Model card: XGBoost

## Intended use

Educational retrospective prediction of the UCI recorded `<30` readmission outcome at hospital discharge. Intended users are portfolio reviewers and researchers reproducing this analysis. This is not a clinically validated prediction system or a treatment recommendation.

## Population and design

UCI US hospital data from 1999–2008; living, non-hospice, completed discharges. See [cohort audit](../reports/cohort.json). Patient-group-aware training CV selects average precision. Validation determines an illustrative maximum-F2 threshold, frozen before test evaluation. No patient is shared across holdouts or CV boundaries.

## Inputs and outputs

Discharge-available encounter summaries; IDs, outcome, race, gender and payer are excluded as predictors. The risk output is a model score; external calibration is unproven. The decision threshold is **0.093570**. Feature engineering, imputation, scaling/encoding and missingness removal live inside the fitted pipeline. Source measurement/coding availability at discharge is an assumption, not a timestamp-validated operational fact.

## Measured test results

| Metric | Value |
| --- | ---: |
| Encounters | 19861 |
| Prevalence | 0.1138 |
| ROC-AUC | 0.6693 |
| Average precision | 0.2259 |
| Sensitivity | 0.7279 |
| Specificity | 0.5028 |
| Precision | 0.1582 |
| F1 | 0.2600 |
| Brier score | 0.0960 |
| Expected calibration error | 0.0024 |

See [patient-cluster confidence intervals](../reports/selected_model_confidence_intervals.json), [calibration](../figures/test_calibration.png), and [subgroup audit](../reports/test_subgroup_metrics.csv). Small strata are flagged; demographic exclusion does not guarantee fairness. The operating point has no established clinical utility/capacity justification.

## Limitations and safeguards

Historical US data have untested UK/NHS transportability. Follow-up, readmissions elsewhere, dates, coding availability and competing outcomes cannot be fully established. Transfers and frequent attenders complicate interpretation. SHAP captures model associations, not causal medication effects. No efficacy, regulatory approval, clinical safety or patient benefit is claimed.

Before any clinical use: contemporary external/temporal validation, development-only calibration, utility and equity analysis, clinical governance and prospective evaluation. Models and predictions are ignored by Git. Load joblib artifacts only when locally created/trusted; they are executable serialisations.
