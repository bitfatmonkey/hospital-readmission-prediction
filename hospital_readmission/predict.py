"""Reusable inference from a locally trained, trusted model artifact."""
from pathlib import Path

import joblib
import pandas as pd

from hospital_readmission.config import ROOT


def predict_readmission(encounters: pd.DataFrame, artifact_path: str | Path | None = None) -> pd.DataFrame:
    """Return scores/decisions in input order, without identifier columns.

    The input index is retained for alignment; omit it when exporting to CSV.
    Only load artifacts created locally by this project: joblib/pickle loading
    executes code and is unsuitable for files from untrusted sources.
    """
    path = Path(artifact_path) if artifact_path is not None else ROOT / "artifacts/selected_model.joblib"
    artifact = joblib.load(path)
    probabilities = artifact["pipeline"].predict_proba(encounters)[:, 1]
    return pd.DataFrame({"readmission_probability": probabilities,
                         "predicted_readmitted_30": probabilities >= artifact["threshold"]}, index=encounters.index)
