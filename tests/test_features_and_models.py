import numpy as np
import pandas as pd
import pytest

from hospital_readmission.evaluation import choose_threshold, probability_metrics
from hospital_readmission.features import ClinicalFeatureEngineer, diagnosis_group
from hospital_readmission.modelling import MODEL_NAMES, make_model


def test_clinical_features_are_stateless_and_exclude_identifiers(eligible_cohort):
    frame = eligible_cohort.iloc[:1000].copy()
    original = frame.copy(deep=True)
    engineered = ClinicalFeatureEngineer().fit_transform(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert not {"encounter_id", "patient_nbr", "readmitted", "race", "gender", "payer_code"} & set(engineered)
    np.testing.assert_array_equal(engineered.prior_visit_total,
                                  frame[["number_inpatient", "number_emergency", "number_outpatient"]].sum(axis=1))
    assert engineered.age_midpoint.between(5, 95).all()
    assert engineered.admission_type_id.map(type).eq(str).all()
    assert engineered.A1Cresult.eq("None").sum() == frame.A1Cresult.eq("None").sum()


@pytest.mark.parametrize("code,expected", [
    ("250.13", "Diabetes"), ("414", "Circulatory"), ("786", "Respiratory"),
    ("V45", "Supplementary V code"), ("E812", "External cause E code"),
    (pd.NA, "Missing"),
])
def test_icd9_category_rules(code, expected):
    assert diagnosis_group(pd.Series([code], dtype="string")).iloc[0] == expected


@pytest.fixture(scope="module")
def real_small_partitions(eligible_cohort, patient_splits):
    indices = patient_splits[0]
    train = eligible_cohort.iloc[indices["train"][:1800]]
    validation = eligible_cohort.iloc[indices["validation"][:600]]
    return train, validation


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_all_model_pipelines_fit_and_predict_real_records(name, real_small_partitions, experiment_config):
    train, validation = real_small_partitions
    model = make_model(name, experiment_config)
    if name in ["Random Forest", "XGBoost"]:
        model.set_params(model__n_estimators=8, model__max_depth=3)
    model.fit(train, train.readmitted.eq("<30").astype(int))
    p = model.predict_proba(validation)[:, 1]
    assert p.shape == (len(validation),)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert model.named_steps["missingness"].dropped_columns_ == ["weight"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    assert not any(any(token in value for token in ["patient_nbr", "encounter_id", "readmitted", "categorical__race_", "categorical__gender_"]) for value in names)
    threshold, f2 = choose_threshold(validation.readmitted.eq("<30"), p)
    metrics = probability_metrics(validation.readmitted.eq("<30"), p, threshold)
    assert sum(metrics["confusion_matrix"].values()) == len(validation)
    assert 0 <= f2 <= 1
    if name == "XGBoost":
        import json
        assert json.loads(model.named_steps["model"].get_booster().save_config())["learner"]["generic_param"]["device"] == "cpu"


def test_imputer_statistics_are_from_training_only(real_small_partitions, experiment_config):
    train, validation = real_small_partitions
    model = make_model("Logistic Regression", experiment_config)
    model.fit(train, train.readmitted.eq("<30").astype(int))
    clinical = model.named_steps["clinical"].transform(train)
    clinical = model.named_steps["missingness"].transform(clinical)
    numeric = clinical.select_dtypes(include="number")
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, numeric.median().to_numpy())
    before = imputer.statistics_.copy()
    model.predict_proba(validation)
    np.testing.assert_array_equal(imputer.statistics_, before)


def test_sparse_xgboost_shap_explains_actual_pipeline(real_small_partitions, experiment_config, tmp_path):
    import json
    from hospital_readmission.explain import explain_model

    train, validation = real_small_partitions
    pipeline = make_model("XGBoost", experiment_config)
    pipeline.set_params(model__n_estimators=8, model__max_depth=3)
    pipeline.fit(train, train.readmitted.eq("<30").astype(int))
    reports, figures = tmp_path / "reports", tmp_path / "figures"
    reports.mkdir()
    figures.mkdir()
    explain_model(pipeline, "XGBoost", train, validation, reports, figures, sample_size=8)
    result = json.loads((reports / "shap_summary.json").read_text())
    assert result["additivity_verified"] and result["pipeline_predictions_verified"]
    assert result["sample_size"] == 8
    assert (figures / "shap_summary.png").stat().st_size > 1000
