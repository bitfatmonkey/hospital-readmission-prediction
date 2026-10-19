# Measured analysis outputs

These files were generated from the real repository ZIP. No patient-level records,
prediction lists, synthetic observations or patient identifier values are exported.

| Files | Scope / meaning |
| --- | --- |
| `raw_dataset_profile.json` | Full source schema, missingness, identifiers and target totals; before clinical filtering |
| `cohort.json` | Prespecified discharge exclusions and eligible population |
| `split_manifest.json` | Counts, prevalence, encounter-list hashes, patient separation and training CV audit |
| `training_data_quality.json`, `training_data_dictionary.csv` | Training-only quality and dtypes |
| `numeric_descriptive_statistics.csv`, `categorical_statistics.csv` | Training-only descriptive summaries |
| `missingness_patterns.csv`, `missingness_pattern_columns.json` | Aggregate training missingness bit patterns and their column order |
| `training_group_readmission_rates.csv` | Training group rates with descriptive, unadjusted Wilson intervals |
| `cv_*.csv`, `training_search.json` | Every actual candidate's grouped-CV scores and the selected training settings |
| `selection.json` | Model and validation thresholds frozen before test evaluation |
| `validation_metrics.json` | Threshold-development partition, separate from test |
| `test_metrics.json`, `model_comparison.csv` | Held-out test results at frozen validation thresholds |
| `selected_model_confidence_intervals.json` | 200 patient-cluster bootstrap intervals, conditional on the fitted model/split |
| `test_subgroup_metrics.csv` | Exploratory test audit; small strata flagged and discrimination withheld |
| `error_analysis.csv` | Aggregate clinical summaries by selected-model error group |
| `shap_summary.json`, `shap_feature_importance.csv` | Fixed random sample of 80 real test encounters; additivity and pipeline probability parity checked |
| `preprocessing.json` | Training-fitted missingness exclusions, encoded names and omitted predictors |
| `run_metadata.json` | Seed/configuration, package versions, source fingerprint and measured run duration |
| `test_execution.json` | Actual fresh-environment automated test execution summary |
| `artifact_verification.json` | Independent artifact-based recomputation of metrics and notebook/README consistency |

The README is generated from these reports, and `documentation --check` rejects
inconsistencies. Local fitted models live in ignored `artifacts/`; run
`python -m hospital_readmission.run` to recreate them. `verify` requires those
trusted local files and recomputes probabilities without any model refitting.

Test outcomes must not guide further feature engineering or hyperparameter search.
Subgroup/SHAP analyses are descriptive predictive associations, not causal effects
or a certification of fairness, clinical safety or NHS transportability.
