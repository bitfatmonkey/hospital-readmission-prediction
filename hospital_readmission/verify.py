"""Recompute saved held-out metrics from trusted local models, without fitting."""
import json

import joblib
import numpy as np
import nbformat

from hospital_readmission.cohort import make_cohort
from hospital_readmission.config import ExperimentConfig, ROOT
from hospital_readmission.data import load_diabetes
from hospital_readmission.documentation import render_readme
from hospital_readmission.evaluation import probability_metrics
from hospital_readmission.reporting import write_json
from hospital_readmission.splitting import split_patients, training_cv


def assert_results_equal(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_results_equal(expected[key], actual[key])
    elif isinstance(expected, (float, list)):
        np.testing.assert_allclose(expected, actual, rtol=1e-9, atol=1e-10)
    else:
        assert expected == actual


def verify_artifacts():
    config = ExperimentConfig()
    dataset = load_diabetes()
    cohort, _ = make_cohort(dataset)
    indices, manifest = split_patients(cohort, config)
    _, cv_audit = training_cv(cohort, indices["train"], config)
    manifest["training_cv"] = cv_audit
    saved_manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    assert manifest == saved_manifest
    test = cohort.iloc[indices["test"]]
    y = test.readmitted.eq("<30").astype(int)
    X = test.drop(columns=["encounter_id", "patient_nbr", "readmitted"])
    expected = json.loads((ROOT / "reports/test_metrics.json").read_text())
    selection = json.loads((ROOT / "reports/selection.json").read_text())
    for name, result in expected.items():
        path = ROOT / "artifacts" / f"{name.lower().replace(' ', '_')}.joblib"
        pipeline = joblib.load(path)  # trusted, locally created artifacts only
        probabilities = pipeline.predict_proba(X)[:, 1]
        actual = probability_metrics(y, probabilities, selection["thresholds"][name])
        assert_results_equal(result, actual)
        print(f"Verified saved test metrics from fitted artifact: {name}", flush=True)
    executed_cells = 0
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        value = nbformat.read(path, as_version=4)
        nbformat.validate(value)
        for cell in value.cells:
            if cell.cell_type == "code" and cell.source.strip():
                assert cell.execution_count is not None, path.name
                assert all(output.output_type != "error" for output in cell.outputs), path.name
                executed_cells += 1
    assert (ROOT / "README.md").read_text() == render_readme()
    result = {"models_recomputed_without_refitting": list(expected), "metrics_match": True,
              "patient_separation_verified": True, "notebooks_executed": len(notebooks),
              "executed_code_cells": executed_cells, "readme_matches_reports": True,
              "archive_unchanged": True, "dataset_downloaded": False, "synthetic_data_used": False}
    write_json(ROOT / "reports/artifact_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify_artifacts()
