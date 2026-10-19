"""Run the complete real-data experiment: python -m hospital_readmission.run."""
import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hospital_readmission.cohort import make_cohort
from hospital_readmission.config import ExperimentConfig, ROOT
from hospital_readmission.data import load_diabetes, summarize_dataset
from hospital_readmission.evaluation import (choose_threshold, cluster_bootstrap, probability_metrics,
                                            subgroup_metrics)
from hospital_readmission.explain import explain_model
from hospital_readmission.modelling import MODEL_NAMES, tune_model
from hospital_readmission.reporting import (error_analysis, evaluation_figures, run_eda, subgroup_figure,
                                           write_json)
from hospital_readmission.splitting import split_patients, training_cv


def training_fingerprint(config, manifest):
    digest = hashlib.sha256(json.dumps({"config": config.to_dict(), "train": manifest["train"]}, sort_keys=True).encode())
    for filename in ["data.py", "features.py", "cohort.py", "config.py", "splitting.py", "modelling.py"]:
        digest.update((ROOT / "hospital_readmission" / filename).read_bytes())
    lock = ROOT / "requirements.lock"
    if lock.exists():
        digest.update(lock.read_bytes())
    return digest.hexdigest()


def run_experiment(config=None, resume=False):
    config = config or ExperimentConfig()
    started = time.monotonic()
    reports, figures, artifacts = ROOT / "reports", ROOT / "figures", ROOT / "artifacts"
    for path in [reports, figures, artifacts]:
        path.mkdir(exist_ok=True)
    dataset = load_diabetes()
    write_json(reports / "raw_dataset_profile.json", summarize_dataset(dataset))
    cohort, audit = make_cohort(dataset)
    write_json(reports / "cohort.json", audit)
    indices, manifest = split_patients(cohort, config)
    folds, cv_audit = training_cv(cohort, indices["train"], config)
    manifest["training_cv"] = cv_audit
    write_json(reports / "split_manifest.json", manifest)
    train, validation, test = [cohort.iloc[indices[name]] for name in ["train", "validation", "test"]]
    print(f"Cohort: {len(cohort):,}; train {len(train):,}; validation {len(validation):,}; test {len(test):,}", flush=True)
    run_eda(train, dataset.mappings, reports, figures)
    excluded = ["encounter_id", "patient_nbr", "readmitted"]
    X_train, X_validation = [frame.drop(columns=excluded) for frame in [train, validation]]
    y_train = train.readmitted.eq("<30").astype(int)
    y_validation = validation.readmitted.eq("<30").astype(int)
    fingerprint = training_fingerprint(config, manifest)
    fitted, training, thresholds, validation_results = {}, {}, {}, {}
    resumed_models = []
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        checkpoint, metadata_path = artifacts / f"{slug}.joblib", artifacts / f"{slug}.json"
        before = time.monotonic()
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if resume and checkpoint.exists() and metadata.get("fingerprint") == fingerprint:
            print(f"Reusing verified training checkpoint: {name}", flush=True)
            estimator = joblib.load(checkpoint)
            info = metadata["training"]
            resumed_models.append(name)
        else:
            print(f"Training/tuning {name}: training folds only", flush=True)
            estimator, info, cv_results = tune_model(name, X_train, y_train, folds, config)
            info["training_seconds"] = time.monotonic() - before
            if cv_results is not None:
                pd.DataFrame(cv_results).to_csv(reports / f"cv_{slug}.csv", index=False)
            joblib.dump(estimator, checkpoint, compress=3)
            write_json(metadata_path, {"fingerprint": fingerprint, "training": info})
        p_validation = estimator.predict_proba(X_validation)[:, 1]
        threshold, f2 = choose_threshold(y_validation, p_validation)
        thresholds[name] = threshold
        validation_results[name] = probability_metrics(y_validation, p_validation, threshold)
        validation_results[name]["f2"] = f2
        fitted[name], training[name] = estimator, info
        write_json(reports / "training_search.json", training)
        print(f"{name}: CV AP {info['cv_average_precision']:.4f}; validation AP {average_precision_score(y_validation, p_validation):.4f}; threshold {threshold:.4f}", flush=True)
    # Selection and all operating points are frozen BEFORE any test prediction.
    selected = max(MODEL_NAMES, key=lambda name: training[name]["cv_average_precision"])
    selection = {"model": selected, "criterion": config.selection_metric,
                 "threshold_rule": config.threshold_rule, "thresholds": thresholds,
                 "test_used_for_selection": False, "training_fingerprint": fingerprint}
    write_json(reports / "selection.json", selection)
    write_json(reports / "validation_metrics.json", validation_results)
    print(f"Frozen selected model: {selected}. Starting untouched test evaluation.", flush=True)
    X_test, y_test = test.drop(columns=excluded), test.readmitted.eq("<30").astype(int)
    predictions = {name: model.predict_proba(X_test)[:, 1] for name, model in fitted.items()}
    results = {name: probability_metrics(y_test, p, thresholds[name]) for name, p in predictions.items()}
    write_json(reports / "test_metrics.json", results)
    rows = []
    for name, result in results.items():
        rows.append({"model": name, "selected_by_training_cv": name == selected,
                     "cv_average_precision": training[name]["cv_average_precision"],
                     **{key: value for key, value in result.items() if key not in ["calibration", "confusion_matrix"]}})
    pd.DataFrame(rows).to_csv(reports / "model_comparison.csv", index=False)
    intervals = cluster_bootstrap(y_test, predictions[selected], test.patient_nbr,
                                  config.bootstrap_iterations, config.seed)
    write_json(reports / "selected_model_confidence_intervals.json", intervals)
    evaluation_figures(y_test, predictions, results, selected, figures)
    subgroups = subgroup_metrics(test, y_test, predictions[selected], thresholds[selected])
    subgroups.to_csv(reports / "test_subgroup_metrics.csv", index=False)
    subgroup_figure(subgroups, figures / "test_subgroup_sensitivity.png")
    error_analysis(test, y_test, predictions[selected], thresholds[selected], reports, figures)
    print("Computing SHAP on a fixed real test sample", flush=True)
    explain_model(fitted[selected], selected, X_train, X_test, reports, figures,
                  sample_size=config.shap_rows, seed=config.seed)
    joblib.dump({"pipeline": fitted[selected], "threshold": thresholds[selected], "selection": selection},
                artifacts / "selected_model.joblib", compress=3)
    write_json(reports / "preprocessing.json", {
        "dropped_for_missingness": fitted[selected].named_steps["missingness"].dropped_columns_,
        "encoded_feature_names": fitted[selected].named_steps["preprocess"].get_feature_names_out().tolist(),
        "preprocessing_fit_partition": "training only, independently within CV folds",
        "excluded_predictors": ["encounter_id", "patient_nbr", "readmitted", "race", "gender", "payer_code"],
    })
    metadata = {"config": config.to_dict(), "python": platform.python_version(),
                "packages": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "xgboost-cpu", "shap", "scipy"]},
                "training_fingerprint": fingerprint, "archive_sha256": dataset.archive_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "resume_requested": resume, "resumed_models": resumed_models,
                "training_seconds_sum": sum(info["training_seconds"] for info in training.values()),
                "synthetic_data_used": False, "dataset_downloaded": False,
                "test_evaluation_completed": True, "shap_completed": True}
    write_json(reports / "run_metadata.json", metadata)
    print(json.dumps({"selected": selected, "test": results[selected], "elapsed_seconds": metadata['elapsed_seconds']}, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse fitted models only if code/config/lock/splits match")
    args = parser.parse_args()
    run_experiment(resume=args.resume)


if __name__ == "__main__":
    main()
