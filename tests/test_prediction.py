import joblib
import numpy as np

from hospital_readmission.modelling import make_model
from hospital_readmission.predict import predict_readmission


def test_reusable_inference_excludes_patient_identifiers(eligible_cohort, patient_splits, experiment_config, tmp_path):
    indices = patient_splits[0]
    train = eligible_cohort.iloc[indices["train"][:1000]]
    validation = eligible_cohort.iloc[indices["validation"][:200]]
    pipeline = make_model("Dummy", experiment_config)
    pipeline.fit(train, train.readmitted.eq("<30").astype(int))
    artifact = tmp_path / "trusted_local_model.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": .1}, artifact)
    output = predict_readmission(validation, artifact)
    assert list(output) == ["readmission_probability", "predicted_readmitted_30"]
    np.testing.assert_allclose(output.readmission_probability, pipeline.predict_proba(validation)[:, 1])
    np.testing.assert_array_equal(output.predicted_readmitted_30, output.readmission_probability >= .1)
    assert output.index.equals(validation.index)
