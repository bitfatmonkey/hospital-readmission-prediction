import pytest

from hospital_readmission.cohort import make_cohort
from hospital_readmission.config import ExperimentConfig
from hospital_readmission.data import load_diabetes
from hospital_readmission.splitting import split_patients


@pytest.fixture(scope="session")
def eligible_cohort():
    return make_cohort(load_diabetes())[0]


@pytest.fixture(scope="session")
def experiment_config():
    return ExperimentConfig()


@pytest.fixture(scope="session")
def patient_splits(eligible_cohort, experiment_config):
    return split_patients(eligible_cohort, experiment_config)
