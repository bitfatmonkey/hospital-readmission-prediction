"""Load the repository's original UCI dataset directly from its ZIP, offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "data/raw/diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
# Fingerprint of the unchanged archive uploaded in repository commit bb20452.
# This pins the local artifact; it is not an independently obtained UCI checksum.
ARCHIVE_SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
DATA_COLUMNS = (
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted",
)
MAPPING_ROWS = {
    "admission_type_id": 8,
    "discharge_disposition_id": 30,
    "admission_source_id": 25,
}
INTEGER_COLUMNS = {
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
}
TARGET_COUNTS = {"<30": 11357, ">30": 35545, "NO": 54864}


@dataclass(frozen=True)
class DiabetesDataset:
    """Unfiltered encounters, code mappings, and local archive provenance.

    '?' markers become missing values; literal 'None' and 'NULL' are preserved.
    Identifiers remain strings. No records are dropped or deduplicated.
    """

    data: pd.DataFrame
    mappings: dict[str, pd.DataFrame]
    archive_path: Path
    archive_sha256: str

    @property
    def features(self) -> pd.DataFrame:
        """The 47 source features, excluding both IDs and the target."""
        return self.data.drop(columns=["encounter_id", "patient_nbr", "readmitted"])

    @property
    def target(self) -> pd.Series:
        """Original three-class readmission labels: '<30', '>30', and 'NO'."""
        return self.data["readmitted"]

    @property
    def patient_ids(self) -> pd.Series:
        """Patient groups for splitting without patient overlap."""
        return self.data["patient_nbr"]


def _read_mappings(archive: ZipFile) -> dict[str, pd.DataFrame]:
    # IDS_mapping.csv contains three CSV tables separated by empty ',' rows.
    sections: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    with archive.open("IDS_mapping.csv") as member:
        with io.TextIOWrapper(member, encoding="utf-8-sig", newline="") as text:
            for row in csv.reader(text):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise ValueError(f"Unexpected mapping row: {row!r}")
                code, description = (value.strip() for value in row)
                if code in MAPPING_ROWS and description == "description":
                    if code in sections:
                        raise ValueError(f"Repeated mapping header: {code}")
                    current = code
                    sections[current] = []
                elif current is None:
                    raise ValueError("Mapping data appeared before its section header")
                else:
                    sections[current].append((int(code), description))
    if set(sections) != set(MAPPING_ROWS):
        raise ValueError("IDS_mapping.csv does not contain all three code tables")
    mappings = {}
    for name, rows in sections.items():
        frame = pd.DataFrame(rows, columns=[name, "description"])
        if len(frame) != MAPPING_ROWS[name] or frame[name].duplicated().any():
            raise ValueError(f"Unexpected or duplicate mapping codes in {name}")
        mappings[name] = frame
    return mappings


def load_diabetes(archive_path: str | Path | None = None) -> DiabetesDataset:
    """Read and validate the pinned UCI archive without extraction or networking.

    The default location is relative to this module, independent of the working
    directory. An explicit path must identify an identical copy of the archive.
    The SHA-256 pin and ZIP CRCs detect changes/corruption before parsing.
    """
    path = Path(archive_path) if archive_path is not None else DEFAULT_ARCHIVE
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original UCI ZIP not found: {path}")
    with path.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    if fingerprint != ARCHIVE_SHA256:
        raise ValueError(f"UCI archive SHA-256 mismatch: {fingerprint}")
    with ZipFile(path) as archive:
        members = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if sorted(members) != ["IDS_mapping.csv", "diabetic_data.csv"]:
            raise ValueError(f"Unexpected UCI ZIP members: {members}")
        failed_member = archive.testzip()
        if failed_member is not None:
            raise ValueError(f"ZIP CRC check failed for {failed_member}")
        with archive.open("diabetic_data.csv") as member:
            data = pd.read_csv(
                member,
                keep_default_na=False,
                na_values=["?"],
                dtype={
                    name: "int64" if name in INTEGER_COLUMNS else "string"
                    for name in DATA_COLUMNS
                },
            )
        mappings = _read_mappings(archive)
    if data.shape != (101766, 50) or tuple(data.columns) != DATA_COLUMNS:
        raise ValueError(f"Unexpected UCI data shape or schema: {data.shape}")
    for name in ("encounter_id", "patient_nbr"):
        if data[name].isna().any() or not data[name].str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"Missing or invalid {name}")
    if not data["encounter_id"].is_unique or data["patient_nbr"].nunique() != 71518:
        raise ValueError("Unexpected encounter or patient ID structure")
    if data["readmitted"].value_counts(dropna=False).to_dict() != TARGET_COUNTS:
        raise ValueError("Unexpected readmission labels or class counts")
    for name, mapping in mappings.items():
        if not data[name].isin(mapping[name]).all():
            raise ValueError(f"Unmapped IDs in {name}")
    return DiabetesDataset(data, mappings, path, fingerprint)


def summarize_dataset(dataset: DiabetesDataset) -> dict:
    """Return JSON-serializable statistics calculated from the loaded records."""
    data = dataset.data
    missing = data.isna().sum()
    patient_counts = dataset.patient_ids.value_counts()
    return {
        "archive": str(dataset.archive_path),
        "archive_sha256": dataset.archive_sha256,
        "zip_crc_verified": True,
        "shape": list(data.shape),
        "feature_shape": list(dataset.features.shape),
        "columns": data.columns.tolist(),
        "missing_values": {name: int(count) for name, count in missing.items() if count},
        "total_missing_values": int(missing.sum()),
        "not_measured_lab_results": {
            name: int(data[name].eq("None").sum())
            for name in ("max_glu_serum", "A1Cresult")
        },
        "patient_ids": {
            "unique_patients": int(data["patient_nbr"].nunique()),
            "patients_with_multiple_encounters": int(patient_counts.gt(1).sum()),
            "repeat_patient_rows": int(data["patient_nbr"].duplicated().sum()),
            "maximum_encounters_per_patient": int(patient_counts.max()),
            "unique_encounters": int(data["encounter_id"].nunique()),
            "duplicate_encounter_ids": int(data["encounter_id"].duplicated().sum()),
            "missing_patient_ids": int(data["patient_nbr"].isna().sum()),
            "missing_encounter_ids": int(data["encounter_id"].isna().sum()),
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "target_distribution": {
            label: {"count": int(count), "percent": round(100 * count / len(data), 4)}
            for label, count in dataset.target.value_counts().items()
        },
        "missing_targets": int(dataset.target.isna().sum()),
        "mapping_shapes": {name: list(frame.shape) for name, frame in dataset.mappings.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Path to an identical copy of the original ZIP")
    args = parser.parse_args()
    print(json.dumps(summarize_dataset(load_diabetes(args.archive)), indent=2))


if __name__ == "__main__":
    main()
