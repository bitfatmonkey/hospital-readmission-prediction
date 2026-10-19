"""Validation-only operating points and held-out probability evaluation."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def choose_threshold(y_validation, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best]), float(f2[best])


def probability_metrics(y, probabilities, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in np.unique(bins))
    fraction, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": float(ece),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": {"mean_probability": mean.tolist(), "observed_frequency": fraction.tolist()},
    }


def cluster_bootstrap(y, probabilities, patient_ids, iterations=200, seed=42):
    """Patient-cluster percentile intervals for a frozen, selected model."""
    y, p = np.asarray(y), np.asarray(probabilities)
    codes, unique = pd.factorize(patient_ids)
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "average_precision": [], "brier_score": []}
    for _ in range(iterations):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[codes]
        if len(np.unique(y[weights > 0])) != 2:
            continue
        values["roc_auc"].append(roc_auc_score(y, p, sample_weight=weights))
        values["average_precision"].append(average_precision_score(y, p, sample_weight=weights))
        values["brier_score"].append(brier_score_loss(y, p, sample_weight=weights))
    return {name: {"lower": float(np.quantile(scores, .025)), "upper": float(np.quantile(scores, .975)),
                   "resamples": len(scores)} for name, scores in values.items()}


def subgroup_metrics(frame, y, probabilities, threshold):
    rows = []
    for attribute in ("age", "race", "gender", "discharge_disposition_id"):
        groups = frame[attribute].astype("string").fillna("Missing")
        for label in sorted(groups.unique()):
            mask = groups.eq(label).to_numpy(dtype=bool)
            truth, score = np.asarray(y)[mask], np.asarray(probabilities)[mask]
            n, positives = len(truth), int(truth.sum())
            reliable = n >= 100 and min(positives, n - positives) >= 20
            predicted = score >= threshold
            tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
            rows.append({"attribute": attribute, "group": label, "n": n, "positives": positives,
                         "prevalence": float(truth.mean()), "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
                         "specificity": float(tn/(tn+fp)) if tn+fp else None,
                         "precision": float(tp/(tp+fp)) if tp+fp else None,
                         "brier_score": float(brier_score_loss(truth, score)),
                         "roc_auc": float(roc_auc_score(truth, score)) if reliable else None,
                         "average_precision": float(average_precision_score(truth, score)) if reliable else None,
                         "small_sample_warning": not reliable})
    return pd.DataFrame(rows)
