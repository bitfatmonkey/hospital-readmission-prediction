The repository previously contained the UCI archive without a complete modelling
workflow. This change adds a reproducible discharge-time, patient-aware analysis
using the unchanged real ZIP, while retaining the existing loader and tests.

The clinical cohort contains 99,319 encounters from 69,980 patients. Training,
validation, test and all CV folds have zero patient overlap. Preprocessing and
class-weight alternatives are fitted within training CV; the validation set alone
chooses F2 thresholds. XGBoost is selected by mean training CV average precision.

| Model | Test ROC-AUC | Test average precision | Brier |
| --- | ---: | ---: | ---: |
| Dummy | 0.5000 | 0.1138 | 0.1008 |
| Logistic Regression | 0.6603 | 0.2136 | 0.0968 |
| Random Forest | 0.6602 | 0.2114 | 0.0972 |
| XGBoost | 0.6693 | 0.2259 | 0.0960 |

At the validation-selected XGBoost threshold, test sensitivity is 72.79% and
precision 15.82%, with 8,751 false positives. This is an educational retrospective
model, not a clinically validated system. Historical US data, uncertain follow-up,
transfers, missing timestamps and limited UK transportability are documented.

The implementation includes 17 figures, three executed explanatory notebooks,
calibration and error/subgroup reports, 200 patient-cluster bootstrap intervals,
SHAP explanations checked against actual sparse-pipeline predictions, reusable
inference, a model card, a hashed dependency lock and a GitHub Actions workflow.
The README and model card are generated from measured reports. The original ZIP
is unchanged; extracted data, fitted models and patient-level predictions are not
committed.

Validation: all four models actually trained/tuned on CPU; 35 automated tests
passed in a separate clean Python 3.11.17 environment with no failures or skips;
all three notebooks executed. Hosted GitHub Actions status is separate from this
local validation.

Reproduce with the README installation commands, then:

```bash
python -m hospital_readmission.run
python -m hospital_readmission.notebooks --execute
python -m pytest -q
python -m hospital_readmission.documentation --check
python -m hospital_readmission.verify
```
