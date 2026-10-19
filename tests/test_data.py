"""Regression checks against the real, unchanged UCI ZIP in the repository."""
import json

import pandas as pd
import pytest

from hospital_readmission.data import load_diabetes, summarize_dataset


@pytest.fixture(scope="session")
def dataset():
    return load_diabetes()


def test_original_schema_and_patient_ids(dataset):
    assert dataset.data.shape == (101766, 50)
    assert dataset.data["encounter_id"].is_unique
    assert dataset.patient_ids.nunique() == 71518
    assert dataset.patient_ids.duplicated().sum() == 30248
    for name in ("encounter_id", "patient_nbr"):
        assert isinstance(dataset.data[name].dtype, pd.StringDtype)
        assert dataset.data[name].str.fullmatch(r"[0-9]+").all()


def test_question_marks_and_not_measured_categories(dataset):
    nonzero = dataset.data.isna().sum()
    assert nonzero[nonzero > 0].to_dict() == {
        "race": 2273, "weight": 98569, "payer_code": 40256,
        "medical_specialty": 49949, "diag_1": 21, "diag_2": 358, "diag_3": 1423,
    }
    assert dataset.data["max_glu_serum"].eq("None").sum() == 96420
    assert dataset.data["A1Cresult"].eq("None").sum() == 84748
    assert not dataset.data.isin(["?"]).any().any()


def test_original_and_thirty_day_targets(dataset):
    assert dataset.target.value_counts().to_dict() == {"NO": 54864, ">30": 35545, "<30": 11357}
    assert dataset.target.eq("<30").value_counts().to_dict() == {False: 90409, True: 11357}
    assert not dataset.target.isna().any()


def test_three_mapping_sections_preserve_null_descriptions(dataset):
    assert {name: frame.shape for name, frame in dataset.mappings.items()} == {
        "admission_type_id": (8, 2),
        "discharge_disposition_id": (30, 2),
        "admission_source_id": (25, 2),
    }
    for name, null_id in [("admission_type_id", 6), ("discharge_disposition_id", 18), ("admission_source_id", 17)]:
        mapping = dataset.mappings[name].set_index(name)["description"]
        assert mapping.loc[null_id] == "NULL"
        assert dataset.data[name].isin(mapping.index).all()


def test_feature_matrix_excludes_target_and_identifiers(dataset):
    assert dataset.features.shape == (101766, 47)
    assert not {"encounter_id", "patient_nbr", "readmitted"}.intersection(dataset.features.columns)


def test_default_archive_does_not_depend_on_working_directory(monkeypatch, tmp_path, dataset):
    monkeypatch.chdir(tmp_path)
    loaded = load_diabetes()
    assert loaded.archive_path == dataset.archive_path
    pd.testing.assert_frame_equal(loaded.data, dataset.data)


def test_summary_reports_actual_loaded_values(dataset):
    summary = summarize_dataset(dataset)
    assert json.loads(json.dumps(summary))["shape"] == [101766, 50]
    assert sum(entry["count"] for entry in summary["target_distribution"].values()) == len(dataset.data)
    assert summary["total_missing_values"] == int(dataset.data.isna().sum().sum())
    assert summary["duplicate_rows"] == 0
