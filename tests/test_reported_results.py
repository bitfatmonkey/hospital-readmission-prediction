import json

import nbformat
import pandas as pd
import pytest

from hospital_readmission.config import ROOT
from hospital_readmission.documentation import render_readme


def read(name):
    return json.loads((ROOT / "reports" / name).read_text())


def test_model_selection_uses_training_cv_not_test_metrics():
    search, selection = read("training_search.json"), read("selection.json")
    assert selection["model"] == max(search, key=lambda name: search[name]["cv_average_precision"])
    assert selection["test_used_for_selection"] is False
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        cv = pd.read_csv(ROOT / "reports" / f"cv_{name.lower().replace(' ', '_')}.csv")
        assert len(cv) == 4
        assert search[name]["cv_average_precision"] == pytest.approx(cv.mean_test_ap.max())


def test_metric_tables_and_confusion_counts_are_consistent():
    saved, manifest = read("test_metrics.json"), read("split_manifest.json")
    comparison = pd.read_csv(ROOT / "reports/model_comparison.csv").set_index("model")
    assert set(saved) == {"Dummy", "Logistic Regression", "Random Forest", "XGBoost"}
    for name, value in saved.items():
        assert sum(value["confusion_matrix"].values()) == manifest["test"]["encounters"]
        assert value["n"] == manifest["test"]["encounters"]
        assert value["positives"] == manifest["test"]["positives"]
        for metric in ["roc_auc", "average_precision", "precision", "recall", "specificity", "f1", "brier_score"]:
            assert comparison.loc[name, metric] == pytest.approx(value[metric])
        assert value["threshold"] == read("selection.json")["thresholds"][name]


def test_readme_exactly_matches_measured_results():
    assert (ROOT / "README.md").read_text() == render_readme()


def test_bootstrap_shap_and_preprocessing_evidence():
    for interval in read("selected_model_confidence_intervals.json").values():
        assert interval["resamples"] == 200
        assert 0 <= interval["lower"] <= interval["upper"] <= 1
    assert read("shap_summary.json")["additivity_verified"] is True
    assert read("shap_summary.json")["pipeline_predictions_verified"] is True
    assert read("preprocessing.json")["dropped_for_missingness"] == ["weight"]
    metadata = read("run_metadata.json")
    assert metadata["synthetic_data_used"] is False
    assert metadata["dataset_downloaded"] is False
    assert metadata["test_evaluation_completed"] is True


@pytest.mark.parametrize("name", ["01_exploratory_analysis.ipynb", "02_features_and_leakage.ipynb", "03_modelling_and_explainability.ipynb"])
def test_notebooks_are_valid_and_all_code_cells_executed(name):
    value = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(value)
    code = [cell for cell in value.cells if cell.cell_type == "code" and cell.source.strip()]
    assert code
    assert all(cell.execution_count is not None for cell in code)
    assert all(output.output_type != "error" for cell in code for output in cell.outputs)
