import numpy as np

from hospital_readmission.cohort import EXCLUDED_DISPOSITIONS
from hospital_readmission.splitting import split_patients, training_cv


def test_completed_living_non_hospice_discharge_cohort(eligible_cohort):
    assert len(eligible_cohort) == 99319
    assert eligible_cohort.patient_nbr.nunique() == 69980
    assert not eligible_cohort.discharge_disposition_id.isin(EXCLUDED_DISPOSITIONS).any()
    assert eligible_cohort.readmitted.eq("<30").sum() == 11303


def test_split_is_complete_disjoint_and_stratified(eligible_cohort, patient_splits):
    indices, audit = patient_splits
    assert len(np.unique(np.concatenate(list(indices.values())))) == len(eligible_cohort)
    groups = {name: set(eligible_cohort.iloc[index].patient_nbr) for name, index in indices.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["test"] & groups["validation"]
    for partition in ["train", "validation", "test"]:
        assert abs(audit[partition]["prevalence"] - eligible_cohort.readmitted.eq("<30").mean()) < .01


def test_split_is_reproducible(eligible_cohort, experiment_config, patient_splits):
    repeated, repeated_audit = split_patients(eligible_cohort, experiment_config)
    for name, index in patient_splits[0].items():
        np.testing.assert_array_equal(index, repeated[name])
    assert repeated_audit == patient_splits[1]


def test_every_training_cv_fold_has_disjoint_patients(eligible_cohort, patient_splits, experiment_config):
    train = eligible_cohort.iloc[patient_splits[0]["train"]]
    folds, audit = training_cv(eligible_cohort, patient_splits[0]["train"], experiment_config)
    assert len(folds) == 3
    for fit, validate in folds:
        assert not set(train.iloc[fit].patient_nbr) & set(train.iloc[validate].patient_nbr)
        assert train.iloc[validate].readmitted.eq("<30").nunique() == 2
    assert all(fold["patient_overlap"] == 0 for fold in audit)
