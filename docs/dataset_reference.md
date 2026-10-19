# Hospital Readmission Prediction

30-day hospital readmission risk prediction using the real UCI Diabetes 130-US
Hospitals dataset. The repository includes the original uploaded ZIP and a reusable
offline data loader for CPU machine learning experiments with Python 3.11.

## Environment and usage

The prepared Codex environment has Python 3.11.17, pandas, NumPy, scikit-learn,
XGBoost (`xgboost-cpu`), SHAP, Matplotlib, Seaborn, SciPy, ucimlrepo, and pytest.
All 32 direct/transitive dependencies are pinned with hashes in
`/workspace/hospital-readmission-environment/requirements.lock`; installation and
startup instructions are saved in the cloud environment configuration.

```bash
cd /workspace/hospital-readmission-prediction
source /workspace/hospital-readmission-environment/activate.sh
python -m hospital_readmission.data
python -m pytest tests -q
```

Activation selects the prepared environment, writable caches, Matplotlib's Agg
backend, and two CPU threads for BLAS/OpenMP. For XGBoost use `device="cpu"`,
`tree_method="hist"`, and `n_jobs=2`. No GPU or external data download is needed.
The loader itself needs Python 3.11+ and pandas; run imports from the repository root.

```python
from hospital_readmission.data import load_diabetes, summarize_dataset

dataset = load_diabetes()
encounters = dataset.data              # all 50 original columns
X = dataset.features                  # 47 source features, IDs/target excluded
y = dataset.target.eq("<30").astype("int8")  # 1: readmitted within 30 days
patient_groups = dataset.patient_ids   # retain for patient-level splitting
admission_types = dataset.mappings["admission_type_id"]
summary = summarize_dataset(dataset)
```

Use patient groups to keep a patient's encounters together when splitting training,
validation, and test sets. Repeated patients make a random row split prone to leakage.
Fit imputation, encoding, and other preprocessing on training data only. The three
mapping ID columns are categorical codes, even though their stored dtype is integer.
The loader preserves all encounters; cohort filtering and eligibility decisions
belong in the experiment pipeline.

## Source archive and dataset identity

The latest uploaded archive was fetched from repository commit
`bb20452d96432dd32428190c0d8f59b552fbc3fd`:

```text
data/raw/diabetes+130-us+hospitals+for+years+1999-2008.zip
SHA-256: f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a
```

It contains exactly `diabetic_data.csv` (19,159,383 bytes) and `IDS_mapping.csv`
(2,547 bytes). ZIP CRC checks pass. The archive's schema, 101,766 records,
71,518 patients, mapping tables, and three target class totals match the original
UCI Diabetes 130-US Hospitals for Years 1999–2008 dataset (ID 296).

The hash pins the uploaded archive and detects any changes to it. It was calculated
locally; no independent byte-for-byte comparison with a live official UCI download
was performed. The loader performs no network requests and makes no synthetic data.

Both CSVs are read directly through `zipfile.ZipFile.open`, without extraction.
The loader verifies the SHA-256, ZIP CRCs, members, exact column order, ID validity,
row counts, mapping coverage, and target counts before returning data. The raw ZIP
is unchanged, and extracted CSV duplicates are ignored by Git.

`IDS_mapping.csv` contains three separate tables, parsed as:

| Mapping | Shape (rows × columns) |
| --- | ---: |
| `admission_type_id` | 8 × 2 |
| `discharge_disposition_id` | 30 × 2 |
| `admission_source_id` | 25 × 2 |

All observed mapping codes are covered. Label whitespace is trimmed in memory;
literal `NULL` mapping descriptions are preserved.

## Verified real-data profile

`diabetic_data.csv` has **101,766 rows × 50 columns**: two identifiers, 47 source
features, and one `readmitted` target. The original column order is:

```text
encounter_id, patient_nbr, race, gender, age, weight,
admission_type_id, discharge_disposition_id, admission_source_id,
time_in_hospital, payer_code, medical_specialty, num_lab_procedures,
num_procedures, num_medications, number_outpatient, number_emergency,
number_inpatient, diag_1, diag_2, diag_3, number_diagnoses,
max_glu_serum, A1Cresult, metformin, repaglinide, nateglinide,
chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide,
tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol,
troglitazone, tolazamide, examide, citoglipton, insulin,
glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone,
metformin-rosiglitazone, metformin-pioglitazone, change, diabetesMed, readmitted
```

Identifier checks found **101,766 unique encounter IDs**, **71,518 unique patient
IDs**, zero missing/invalid IDs, zero duplicate encounters, and zero duplicate
complete rows. There are 16,773 patients with multiple encounters, 30,248 encounters
beyond each patient's first, and at most 40 encounters per patient. IDs remain
strings and are excluded from the feature matrix.

The original `?` missing markers are converted to pandas missing values:

| Column | Missing values |
| --- | ---: |
| `race` | 2,273 |
| `weight` | 98,569 |
| `payer_code` | 40,256 |
| `medical_specialty` | 49,949 |
| `diag_1` | 21 |
| `diag_2` | 358 |
| `diag_3` | 1,423 |
| **Total** | **192,849** |

There are no empty CSV fields or other `?` markers. Separately, the laboratory
columns contain the explicit **not measured** category `None`: 96,420 in
`max_glu_serum` and 84,748 in `A1Cresult`. These categories are preserved, rather
than silently converted to missing values by pandas' default NA rules.

The target has no missing values:

| `readmitted` | Meaning | Count | Percentage |
| --- | --- | ---: | ---: |
| `<30` | Readmitted within 30 days | 11,357 | 11.1599% |
| `>30` | Readmitted after 30 days | 35,545 | 34.9282% |
| `NO` | No recorded readmission | 54,864 | 53.9119% |

For the binary 30-day task, the positive class has 11,357 encounters and the
negative class (`>30` or `NO`) has 90,409. These are encounter-level class counts,
before cohort filtering or patient-level train/test splitting.

The project tests exercise this real archive, missing-value semantics, mapping
tables, identifiers, feature/target separation, and loading from another working
directory. They do not download data or generate synthetic records.

Validation completed with **7 tests passed**, including warnings treated as errors.
A separate CPU readiness check used all 101,766 real encounters and all 47 source
features: 81,278 training encounters and 20,488 test encounters with **zero patient
overlap**. Training-only preprocessing produced 2,454 encoded features; XGBoost
CPU fit/predict and SHAP additivity checks on eight real test encounters passed.
The data and environment are ready for the complete machine learning workflow;
this readiness check did not perform cohort selection or model tuning.

## UCI citation and attribution

Clore, J., Cios, K., DeShazo, J., & Strack, B.
(2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset].
UCI Machine Learning Repository. **DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J)**.

Dataset record: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>.

The dataset is distributed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
Credit belongs to the dataset creators listed above and the UCI Machine Learning
Repository. The original ZIP is redistributed unchanged. This project's loader
normalizes `?` missing markers and mapping-label whitespace in memory, preserves
the original records, and demonstrates a derived binary target in the usage example.
