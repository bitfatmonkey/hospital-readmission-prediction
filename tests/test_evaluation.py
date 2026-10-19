import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from hospital_readmission.evaluation import choose_threshold, cluster_bootstrap, probability_metrics
from hospital_readmission.modelling import make_model


@pytest.fixture(scope="module")
def real_validation_predictions(eligible_cohort, patient_splits, experiment_config):
    splits = patient_splits[0]
    train = eligible_cohort.iloc[splits["train"][:1800]]
    validation = eligible_cohort.iloc[splits["validation"][:600]]
    model = make_model("Logistic Regression", experiment_config)
    model.fit(train, train.readmitted.eq("<30").astype(int))
    p = model.predict_proba(validation)[:, 1]
    return validation, p


def test_validation_threshold_maximises_declared_f2(real_validation_predictions):
    frame, p = real_validation_predictions
    y = frame.readmitted.eq("<30").astype(int)
    threshold, selected_f2 = choose_threshold(y, p)
    precision, recall, candidates = precision_recall_curve(y, p)
    values = 5 * precision[:-1] * recall[:-1] / np.maximum(4*precision[:-1]+recall[:-1], 1e-15)
    assert selected_f2 == pytest.approx(values.max())
    assert threshold in candidates


def test_confusion_brier_and_specificity_match_real_predictions(real_validation_predictions):
    frame, p = real_validation_predictions
    y = frame.readmitted.eq("<30").to_numpy()
    threshold, _ = choose_threshold(y, p)
    result = probability_metrics(y, p, threshold)
    predicted = p >= threshold
    assert result["confusion_matrix"] == {
        "tp": int((y & predicted).sum()), "fp": int((~y & predicted).sum()),
        "tn": int((~y & ~predicted).sum()), "fn": int((y & ~predicted).sum()),
    }
    assert result["brier_score"] == pytest.approx(np.mean((y-p)**2))
    assert result["specificity"] == pytest.approx((~y & ~predicted).sum()/(~y).sum())
    assert len(result["calibration"]["mean_probability"]) <= 10


def test_patient_bootstrap_is_deterministic(real_validation_predictions):
    frame, p = real_validation_predictions
    y = frame.readmitted.eq("<30").astype(int)
    first = cluster_bootstrap(y, p, frame.patient_nbr, iterations=20, seed=42)
    assert first == cluster_bootstrap(y, p, frame.patient_nbr, iterations=20, seed=42)
    for interval in first.values():
        assert interval["resamples"] == 20
        assert 0 <= interval["lower"] <= interval["upper"] <= 1
