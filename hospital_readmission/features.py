"""Stateless clinical derivations and training-fitted scikit-learn preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hospital_readmission.data import DATA_COLUMNS

MEDICATION_COLUMNS = DATA_COLUMNS[DATA_COLUMNS.index("metformin"):DATA_COLUMNS.index("change")]
CODE_COLUMNS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
EXCLUDED_PREDICTORS = ("encounter_id", "patient_nbr", "readmitted", "race", "gender", "payer_code")


def diagnosis_group(values: pd.Series) -> pd.Series:
    """Broad ICD-9 groups; keep missing, V and E codes distinguishable.

    These are educational groupings, not a validated clinical coding ontology.
    """
    text = values.astype("string")
    code = pd.to_numeric(text, errors="coerce")
    result = pd.Series("Other", index=values.index, dtype="string")
    rules = [
        (code.between(390, 460, inclusive="left") | code.between(785, 786, inclusive="left"), "Circulatory"),
        (code.between(460, 520, inclusive="left") | code.between(786, 787, inclusive="left"), "Respiratory"),
        (code.between(520, 580, inclusive="left") | code.between(787, 788, inclusive="left"), "Digestive"),
        (code.between(250, 251, inclusive="left"), "Diabetes"),
        (code.between(800, 1000, inclusive="left"), "Injury"),
        (code.between(710, 740, inclusive="left"), "Musculoskeletal"),
        (code.between(580, 630, inclusive="left") | code.between(788, 789, inclusive="left"), "Genitourinary"),
        (code.between(140, 240, inclusive="left"), "Neoplasm"),
        (text.str.startswith("V", na=False), "Supplementary V code"),
        (text.str.startswith("E", na=False), "External cause E code"),
        (text.isna(), "Missing"),
    ]
    for mask, label in rules:
        result.loc[mask.fillna(False)] = label
    return result


class ClinicalFeatureEngineer(TransformerMixin, BaseEstimator):
    """Discharge-available, prespecified rules; no learned population statistics."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = X.copy()
        frame["age_midpoint"] = pd.to_numeric(
            frame.age.astype("string").str.extract(r"\[(\d+)-", expand=False), errors="coerce"
        ).astype(float) + 5
        for column in ("diag_1", "diag_2", "diag_3"):
            frame[f"{column}_group"] = diagnosis_group(frame[column])
        prior = ["number_outpatient", "number_emergency", "number_inpatient"]
        frame["prior_visit_total"] = frame[prior].sum(axis=1)
        frame["active_diabetes_medications"] = frame[list(MEDICATION_COLUMNS)].isin(["Steady", "Up", "Down"]).sum(axis=1)
        frame["medication_dose_changes"] = frame[list(MEDICATION_COLUMNS)].isin(["Up", "Down"]).sum(axis=1)
        frame["a1c_measured"] = frame.A1Cresult.ne("None").astype(int)
        frame["glucose_measured"] = frame.max_glu_serum.ne("None").astype(int)
        frame = frame.drop(columns=[*EXCLUDED_PREDICTORS, "age", "diag_1", "diag_2", "diag_3"], errors="ignore")
        for column in CODE_COLUMNS:
            frame[column] = frame[column].astype("string")
        # sklearn's imputers consume np.nan rather than pandas scalar pd.NA.
        for column in frame.select_dtypes(include=["string", "object"]).columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame


class MissingnessDropper(TransformerMixin, BaseEstimator):
    """Learn which columns exceed the threshold on each training fold only."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = X.columns[X.isna().mean().le(self.threshold)].tolist()
        self.dropped_columns_ = [c for c in X if c not in self.columns_]
        return self

    def transform(self, X):
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def make_preprocessing(scale_numeric=False):
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                 max_categories=60, sparse_output=True, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=1.0)
