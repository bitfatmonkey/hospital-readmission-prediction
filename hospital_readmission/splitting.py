"""Deterministic stratified group splitting and auditable separation checks."""
import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def split_patients(cohort, config):
    y = cohort.readmitted.eq("<30").astype(int).to_numpy()
    groups = cohort.patient_nbr.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=config.test_folds, shuffle=True, random_state=config.seed)
    development, test = next(splitter.split(cohort, y, groups))
    validation_split = StratifiedGroupKFold(
        n_splits=config.validation_folds, shuffle=True, random_state=config.seed + 1
    )
    relative_train, relative_validation = next(validation_split.split(
        cohort.iloc[development], y[development], groups[development]
    ))
    indices = {"train": development[relative_train], "validation": development[relative_validation], "test": test}
    group_sets = {name: set(groups[index]) for name, index in indices.items()}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if group_sets[left] & group_sets[right]:
            raise AssertionError(f"Patient leakage between {left} and {right}")
    if len(np.unique(np.concatenate(list(indices.values())))) != len(cohort):
        raise AssertionError("Splits do not partition the cohort")
    manifest = {}
    for name, index in indices.items():
        encounter_ids = sorted(cohort.iloc[index].encounter_id.tolist())
        manifest[name] = {
            "encounters": len(index), "patients": len(group_sets[name]),
            "positives": int(y[index].sum()), "prevalence": float(y[index].mean()),
            "encounter_id_sha256": hashlib.sha256("\n".join(encounter_ids).encode()).hexdigest(),
        }
    manifest["pairwise_patient_overlap"] = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    return indices, manifest


def training_cv(cohort, train_indices, config):
    train = cohort.iloc[train_indices]
    y = train.readmitted.eq("<30").astype(int)
    groups = train.patient_nbr
    splitter = StratifiedGroupKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed + 2)
    folds = list(splitter.split(train, y, groups))
    audit = []
    for number, (fit, validate) in enumerate(folds, 1):
        overlap = set(groups.iloc[fit]) & set(groups.iloc[validate])
        if overlap:
            raise AssertionError("Patient leakage in training CV")
        audit.append({"fold": number, "fit_rows": len(fit), "validation_rows": len(validate),
                      "fit_patients": int(groups.iloc[fit].nunique()),
                      "validation_patients": int(groups.iloc[validate].nunique()), "patient_overlap": 0})
    return folds, audit
