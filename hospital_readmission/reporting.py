"""Aggregate-only EDA and evaluation figures; no patient-level exports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from hospital_readmission.features import ClinicalFeatureEngineer, diagnosis_group

PALETTE = sns.color_palette("colorblind")


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def group_summary(frame, column):
    y = frame.readmitted.eq("<30").astype(int)
    working = pd.DataFrame({"group": frame[column].astype("string").fillna("Missing"), "positive": y})
    result = working.groupby("group").positive.agg(n="size", positives="sum").reset_index()
    result["rate"] = result.positives / result.n
    z = norm.ppf(.975)
    denominator = 1 + z*z/result.n
    centre = (result.rate + z*z/(2*result.n))/denominator
    width = z*np.sqrt(result.rate*(1-result.rate)/result.n + z*z/(4*result.n**2))/denominator
    result["lower"], result["upper"] = centre-width, centre+width
    result.insert(0, "attribute", column)
    return result


def rate_plot(summary, ax, title, labels=None, limit=12):
    table = summary.nlargest(limit, "n").sort_values("rate")
    display = [str(labels.get(v, v) if labels else v)[:53] for v in table.group]
    ypos = np.arange(len(table))
    ax.errorbar(table.rate * 100, ypos,
                xerr=np.maximum(np.vstack([(table.rate-table.lower)*100, (table.upper-table.rate)*100]), 0),
                fmt="o", capsize=3, color=PALETTE[0])
    ax.set_yticks(ypos, display)
    ax.set_xlabel("30-day readmission rate (%) with 95% Wilson interval")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)


def run_eda(train, mappings, reports, figures):
    """Explore the training partition only; test outcomes remain untouched."""
    reports, figures = Path(reports), Path(figures)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    safe = train.drop(columns=["encounter_id", "patient_nbr"])
    clinical = ClinicalFeatureEngineer().fit_transform(train)
    numeric = clinical.select_dtypes(include="number")
    numeric.describe(percentiles=[.25, .5, .75, .95, .99]).T.to_csv(reports / "numeric_descriptive_statistics.csv")
    schema = pd.DataFrame({"column": train.columns, "dtype": train.dtypes.astype(str).values,
                           "unique_values": train.nunique(dropna=True).values,
                           "missing": train.isna().sum().values})
    schema.to_csv(reports / "training_data_dictionary.csv", index=False)
    missing = safe.isna().sum().sort_values(ascending=False)
    quality = {"scope": "training partition only", "rows": len(train),
               "patients": int(train.patient_nbr.nunique()),
               "missing_values": {str(k): int(v) for k, v in missing.items() if v},
               "lab_not_measured": {c: int(train[c].eq("None").sum()) for c in ["max_glu_serum", "A1Cresult"]},
               "target_counts": {str(k): int(v) for k,v in train.readmitted.value_counts().items()},
               "constant_predictors": [c for c in safe if safe[c].nunique(dropna=False) <= 1],
               "unknown_gender_rows": int(train.gender.eq("Unknown/Invalid").sum())}
    write_json(reports / "training_data_quality.json", quality)
    categorical = []
    for column in safe.select_dtypes(include=["string", "object"]):
        counts = safe[column].value_counts(dropna=False)
        categorical.append({"column": column, "unique": int(safe[column].nunique()),
                            "most_frequent_category": str(counts.index[0]), "most_frequent_count": int(counts.iloc[0])})
    pd.DataFrame(categorical).to_csv(reports / "categorical_statistics.csv", index=False)
    indicators = safe.loc[:, missing[missing > 0].index].isna().astype(int)
    for c in ["max_glu_serum", "A1Cresult"]:
        indicators[f"{c}: not measured"] = train[c].eq("None").astype(int)
    patterns = indicators.astype(str).agg("".join, axis=1).value_counts().rename_axis("pattern").reset_index(name="count")
    patterns.to_csv(reports / "missingness_patterns.csv", index=False)
    write_json(reports / "missingness_pattern_columns.json", indicators.columns.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    nonzero = missing[missing > 0].sort_values()
    axes[0].barh(nonzero.index, nonzero/len(train)*100, color=PALETTE[0])
    axes[0].set(xlabel="Missing (%)", title="Question-mark missingness (training only)")
    sns.heatmap(indicators.corr(), ax=axes[1], vmin=-1, vmax=1, cmap="vlag", annot=False)
    axes[1].set_title("Missingness / unmeasured-test association")
    save_figure(fig, figures / "eda_missingness.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    sample = indicators.sample(min(300, len(indicators)), random_state=42)
    sns.heatmap(sample.T, ax=ax, cmap=["#f0f4f8", "#0072B2"], cbar=False, xticklabels=False)
    ax.set(xlabel="300 sampled real training encounters (identifiers omitted)", title="Missingness patterns")
    save_figure(fig, figures / "eda_missingness_patterns.png")
    all_groups = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    counts = train.readmitted.value_counts().reindex(["<30", ">30", "NO"])
    axes[0, 0].bar(counts.index, counts, color=PALETTE[:3])
    axes[0, 0].set(title="Original readmission outcome (training only)", ylabel="Encounters")
    for column, ax in zip(["age", "race", "gender"], [axes[0,1], axes[1,0], axes[1,1]]):
        summary = group_summary(train, column)
        all_groups.append(summary)
        rate_plot(summary, ax, f"Readmission by {column}")
    save_figure(fig, figures / "eda_demographics.png")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for column, ax in zip(mappings, axes):
        summary = group_summary(train, column)
        all_groups.append(summary)
        labels = {str(row[column]): row["description"] for _, row in mappings[column].iterrows()}
        rate_plot(summary, ax, column.replace("_", " "), labels, limit=10)
    save_figure(fig, figures / "eda_admissions.png")
    visit_frame = train.copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for column, ax in zip(["number_inpatient", "number_outpatient", "number_emergency"], axes):
        visit_frame[f"{column}_band"] = visit_frame[column].clip(upper=3).astype(str).replace({"3":"3+"})
        summary = group_summary(visit_frame, f"{column}_band")
        all_groups.append(summary)
        rate_plot(summary, ax, column.replace("number_", "Prior ").replace("_", " "))
    save_figure(fig, figures / "eda_prior_visits.png")
    diagnosis_frame = train.copy()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for column, ax in zip(["diag_1", "diag_2", "diag_3"], axes):
        diagnosis_frame[f"{column}_group"] = diagnosis_group(train[column])
        summary = group_summary(diagnosis_frame, f"{column}_group")
        all_groups.append(summary)
        rate_plot(summary, ax, f"{column}: broad ICD-9 category", limit=11)
    save_figure(fig, figures / "eda_diagnoses.png")
    clinical_frame = train.copy()
    clinical_frame["medication_count_band"] = pd.cut(train.num_medications, [-1, 10, 20, 40, np.inf], labels=["0–10", "11–20", "21–40", "41+"])
    clinical_frame["length_of_stay_band"] = pd.cut(train.time_in_hospital, [0, 3, 7, 14], labels=["1–3", "4–7", "8–14"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for column, ax in zip(["length_of_stay_band", "medication_count_band", "A1Cresult", "insulin"], axes.flat):
        summary = group_summary(clinical_frame, column)
        all_groups.append(summary)
        rate_plot(summary, ax, column.replace("_", " "))
    save_figure(fig, figures / "eda_clinical_predictors.png")
    grouped = pd.concat(all_groups, ignore_index=True)
    grouped.to_csv(reports / "training_group_readmission_rates.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric.assign(readmitted_30=train.readmitted.eq("<30").astype(int)).corr(method="spearman")
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax, square=True, vmin=-1, vmax=1)
    ax.set_title("Spearman associations: training data; not causal effects")
    save_figure(fig, figures / "eda_numeric_associations.png")
    return quality


def evaluation_figures(y, predictions, metrics, selected, figures):
    figures = Path(figures)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name}: {metrics[name]['roc_auc']:.3f}")
    ax.plot([0,1], [0,1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="Untouched test set: ROC")
    ax.legend()
    save_figure(fig, figures / "test_roc.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in predictions.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{name}: AP {metrics[name]['average_precision']:.3f}")
    ax.axhline(np.mean(y), linestyle="--", color="grey", label=f"Prevalence {np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Untouched test set: precision–recall", ylim=(0, 1))
    ax.legend()
    save_figure(fig, figures / "test_precision_recall.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1], [0,1], "--", color="grey", label="Perfect calibration")
    for name, result in metrics.items():
        curve = result["calibration"]
        ax.plot(curve["mean_probability"], curve["observed_frequency"], "o-", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed readmission rate",
           title="Test calibration: up to 10 equal-frequency bins", xlim=(0,1), ylim=(0,1))
    ax.legend()
    save_figure(fig, figures / "test_calibration.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for (name, result), ax in zip(metrics.items(), axes.flat):
        cm = result["confusion_matrix"]
        ConfusionMatrixDisplay(np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]),
                               display_labels=["Not <30", "<30"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}; validation threshold {result['threshold']:.3f}")
    save_figure(fig, figures / "test_confusion_matrices.png")


def error_analysis(test, y, probabilities, threshold, reports, figures):
    truth, predicted = np.asarray(y, dtype=bool), np.asarray(probabilities) >= threshold
    labels = np.select([truth & predicted, truth & ~predicted, ~truth & predicted],
                       ["True positive", "False negative", "False positive"], default="True negative")
    working = test.copy()
    working["error_group"] = labels
    numeric = ["time_in_hospital", "number_inpatient", "number_emergency", "number_outpatient", "num_medications"]
    table = working.groupby("error_group")[numeric].agg(["mean", "median"])
    table.columns = ["_".join(column) for column in table.columns]
    table["n"] = working.groupby("error_group").size()
    table.to_csv(Path(reports) / "error_analysis.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, column in zip(axes, ["number_inpatient", "time_in_hospital", "num_medications"]):
        ax.bar(table.index, table[f"{column}_mean"], color=PALETTE[:len(table)])
        ax.tick_params(axis="x", rotation=30)
        ax.set(title=column.replace("_", " "), ylabel="Mean in test error group")
    save_figure(fig, Path(figures) / "test_error_analysis.png")


def subgroup_figure(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for attribute, ax in zip(["age", "race", "gender"], axes):
        sub = table.loc[(table.attribute == attribute) & ~table.small_sample_warning].sort_values("sensitivity")
        ax.barh(sub.group, sub.sensitivity, color=PALETTE[0])
        ax.set(xlabel="Sensitivity at frozen validation threshold", title=f"Test subgroup: {attribute}", xlim=(0,1))
    save_figure(fig, path)
