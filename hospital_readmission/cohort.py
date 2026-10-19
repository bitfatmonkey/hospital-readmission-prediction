"""Clinically defined discharge-time cohort, chosen before model evaluation."""
import pandas as pd

from hospital_readmission.data import DiabetesDataset

EXCLUDED_DISPOSITIONS = {
    9: "Admitted as an inpatient to this hospital: not a completed discharge",
    11: "Expired",
    12: "Still a patient: not a completed discharge",
    13: "Hospice at home",
    14: "Hospice in a medical facility",
    19: "Expired at home",
    20: "Expired in a medical facility",
    21: "Expired, location unknown",
}


def make_cohort(dataset: DiabetesDataset) -> tuple[pd.DataFrame, dict]:
    """Retain living, non-hospice, completed discharges, without relabelling.

    Death precludes subsequent readmission. Hospice represents a different care
    objective. Ongoing admissions do not define the intended discharge time.
    Transfers remain eligible and are explicitly discussed as a label limitation.
    """
    data = dataset.data
    mask = data["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)
    cohort = data.loc[~mask].copy().reset_index(drop=True)
    audit = {
        "prediction_time": "hospital discharge",
        "positive_outcome": "recorded readmission <30 days",
        "raw_rows": len(data),
        "excluded_rows": int(mask.sum()),
        "eligible_rows": len(cohort),
        "eligible_patients": int(cohort.patient_nbr.nunique()),
        "eligible_target_counts": {str(k): int(v) for k, v in cohort.readmitted.value_counts().items()},
        "exclusions": [
            {"code": code, "reason": reason,
             "rows": int(data.discharge_disposition_id.eq(code).sum())}
            for code, reason in EXCLUDED_DISPOSITIONS.items()
        ],
        "transfers": "Retained; destination may represent continuation of care rather than a new admission",
    }
    return cohort, audit
