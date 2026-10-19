"""Create/exercise explanatory notebooks using the real measured run outputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

from hospital_readmission.config import ROOT

SETUP = '''from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "hospital_readmission").is_dir())
def report(name):
    return json.loads((ROOT / "reports" / name).read_text())
def figure(name):
    display(Image(filename=str(ROOT / "figures" / name)))
from hospital_readmission.data import load_diabetes
from hospital_readmission.cohort import make_cohort
from hospital_readmission.config import ExperimentConfig
from hospital_readmission.splitting import split_patients
dataset = load_diabetes()
cohort, cohort_audit = make_cohort(dataset)
splits, split_audit = split_patients(cohort, ExperimentConfig())
training = cohort.iloc[splits["train"]]
print("Real ZIP loaded offline:", dataset.data.shape, "eligible:", cohort.shape)
'''


class LocalKernelManager(KernelManager):
    """Keep notebook communication on local IPC rather than TCP sockets."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("transport", "ipc")
        super().__init__(*args, **kwargs)


def notebook(cells):
    result = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in cells
    ])
    result.metadata.kernelspec = {"display_name": "Python 3.11 (Hospital Readmission)",
                                  "language": "python", "name": "hospital-readmission"}
    result.metadata.language_info = {"name": "python", "version": "3.11.17"}
    return result


def create_notebooks():
    common = [("code", SETUP)]
    eda = notebook([
        ("markdown", "# 01 · Real-data exploratory analysis\n\n**Question:** which discharge-available characteristics are associated with recorded 30-day readmission?\n\nThe source is the unchanged UCI ZIP. The prediction time is discharge. Model-guiding EDA uses **training patients only**; the validation set sets operating points and the test set remains untouched until selection is frozen. These are retrospective associations, not causal treatment effects."),
        *common,
        ("code", 'display(pd.DataFrame(cohort_audit["exclusions"]))\ndisplay(pd.DataFrame({k:split_audit[k] for k in ["train","validation","test"]}).T)'),
        ("markdown", "## Data quality and missingness\nQuestion marks become missing values. Literal `None` in the two laboratory columns means **not measured** and is retained. Missingness may encode clinical decision-making; imputation does not make it missing at random. No patient-level records or identifiers are displayed."),
        ("code", 'display(pd.read_csv(ROOT / "reports/training_data_dictionary.csv"))\ndisplay(pd.read_csv(ROOT / "reports/numeric_descriptive_statistics.csv", index_col=0))\ndisplay(pd.read_csv(ROOT / "reports/categorical_statistics.csv"))'),
        ("code", 'display(pd.Series(report("training_data_quality.json")))\nfigure("eda_missingness.png")\nfigure("eda_missingness_patterns.png")'),
        ("markdown", "## Class imbalance, demographics and admission characteristics\nReadmission rates use encounter denominators. Wilson intervals in these descriptive plots are unadjusted for repeated patients. Small groups have considerable uncertainty; they do not establish group-specific causal effects."),
        ("code", 'figure("eda_demographics.png")\nfigure("eda_admissions.png")'),
        ("markdown", "## Previous utilisation, diagnoses and clinical predictors\nPrior visit counts are supplied by the source; no future encounter history is constructed. ICD-9 codes are grouped using prespecified rules. Diagnosis categories, insulin changes and test ordering may reflect severity and care processes rather than treatment effects."),
        ("code", 'figure("eda_prior_visits.png")\nfigure("eda_diagnoses.png")\nfigure("eda_clinical_predictors.png")\nfigure("eda_numeric_associations.png")'),
        ("code", 'rates = pd.read_csv(ROOT / "reports/training_group_readmission_rates.csv")\ndisplay(rates.loc[rates.n >= 100].sort_values("rate", ascending=False).head(15))'),
        ("markdown", "## Consequences for modelling\nUse training-fitted missingness removal, imputation and infrequent-category encoding; scale numbers for logistic regression. Compare against a prior-only baseline. Keep every patient's encounters in one split. Exclude identifiers and the outcome; omit race, gender and payer from the primary feature set while retaining demographics for audit. Exclusion alone does not ensure fairness. The EDA informs interpretation, not a post hoc test-driven model search."),
    ])
    features = notebook([
        ("markdown", "# 02 · Cohort, clinical features and leakage controls\n\nPrediction is at **discharge**, not admission. Death/hospice and ongoing admissions are excluded before splitting. No rows are resampled and no synthetic observations are generated."),
        *common,
        ("code", 'display(pd.DataFrame(cohort_audit["exclusions"]))\nprint("Eligible encounters:", len(cohort), "patients:", cohort.patient_nbr.nunique())'),
        ("markdown", "## Prespecified feature engineering\nDerive age midpoint, broad diagnosis categories, prior-visit totals, medication indicators and test-measurement flags. These rules are stateless. Discharge code IDs are categorical. Drop identifiers, the target, race, gender and payer. Weight is removed only if its missingness exceeds 95% on the current training fold."),
        ("code", 'from hospital_readmission.features import ClinicalFeatureEngineer\nengineered = ClinicalFeatureEngineer().fit_transform(training)\ndisplay(pd.DataFrame({"dtype":engineered.dtypes.astype(str), "missing_fraction":engineered.isna().mean()}))\nprint("Engineered columns:", engineered.columns.tolist())'),
        ("markdown", "## Patient-level holdouts and cross-validation\nFive-fold stratified group splits reserve approximately 20% for test and a further 20% of development for threshold validation. Three group-aware CV folds within training tune models. The same patients never appear on both sides of any split."),
        ("code", 'manifest = report("split_manifest.json")\ndisplay(pd.DataFrame({k:manifest[k] for k in ["train","validation","test"]}).T)\ndisplay(pd.DataFrame(manifest["training_cv"]))\nassert all(v == 0 for v in manifest["pairwise_patient_overlap"].values())'),
        ("markdown", "## Training-fitted pipeline\nEach CV candidate owns its feature engineering, missingness rule, median imputation and one-hot category dictionary. Logistic regression additionally scales numeric features. Class weights are candidates inside training CV, rather than oversampling before splitting. Validation and test transformations never refit these steps."),
        ("code", 'import joblib\nartifact = joblib.load(ROOT / "artifacts/selected_model.joblib")  # locally trained/trusted only\ndisplay(artifact["pipeline"])\nprep = report("preprocessing.json")\nprint("Removed for training missingness:", prep["dropped_for_missingness"])\nprint("Encoded features:", len(prep["encoded_feature_names"]))\nprint("Excluded predictors:", prep["excluded_predictors"])'),
        ("markdown", "## Limitations of the prediction snapshot\nThe encounter summaries are assumed available at discharge. Exact coding/test-result availability is not timestamped, so operational real-time availability is unproven. Admission/discharge dates are absent, preventing a true temporal validation or an overlap-in-time audit across distinct patients. Transfers and incomplete readmission capture remain label limitations."),
    ])
    modelling = notebook([
        ("markdown", "# 03 · Model comparison, uncertainty and explainability\n\nRun `python -m hospital_readmission.run` first to execute the actual CPU experiment. This notebook reads its measured outputs and locally fitted model; it does not fabricate metrics or rerun tuning when opened. Model selection uses **mean training CV average precision**, followed by a validation-only maximum-F2 threshold. Test metrics are reported after those choices are frozen."),
        *common,
        ("code", 'selection = report("selection.json")\ncomparison = pd.read_csv(ROOT / "reports/model_comparison.csv")\ndisplay(comparison.sort_values("cv_average_precision", ascending=False))\nprint("Selected before test evaluation:", selection["model"])\nprint("Thresholds:", selection["thresholds"])\nassert selection["test_used_for_selection"] is False'),
        ("code", 'for path in sorted((ROOT / "reports").glob("cv_*.csv")):\n    print(path.stem)\n    search = pd.read_csv(path)\n    display(search[["params","mean_test_ap","std_test_ap","mean_test_roc_auc"]])'),
        ("markdown", "## Untouched test performance\nAverage precision is the step-weighted area under the precision–recall relationship, not trapezoidal PR area. ROC-AUC describes discrimination; precision/recall/specificity describe the frozen operating point. Brier score and reliability plots assess probability quality. Weighting can improve ranking while distorting absolute probabilities. No calibration was fitted on the test set."),
        ("code", 'display(pd.DataFrame(report("selected_model_confidence_intervals.json")).T)\nfigure("test_roc.png")\nfigure("test_precision_recall.png")\nfigure("test_calibration.png")\nfigure("test_confusion_matrices.png")'),
        ("markdown", "## Subgroup and error analysis\nThese are exploratory audits after model selection. Discrimination metrics are withheld for subgroups with fewer than 100 encounters or 20 examples of either outcome. Repeated encounters, uncertain labels and small strata limit fairness conclusions. The confidence intervals above bootstrap patients, not individual rows."),
        ("code", 'groups = pd.read_csv(ROOT / "reports/test_subgroup_metrics.csv")\ndisplay(groups.loc[groups.attribute.isin(["age","race","gender"])])\nfigure("test_subgroup_sensitivity.png")\ndisplay(pd.read_csv(ROOT / "reports/error_analysis.csv"))\nfigure("test_error_analysis.png")'),
        ("markdown", "## SHAP: predictive associations\nA fixed random sample of real test encounters is explained after selection. Additivity is verified against the model output. For tree/logistic margins, values are in log odds; for Random Forest they are positive-class probability contributions. Encoded-category features are not independently manipulable causes. No treatment benefit can be inferred from medication SHAP values."),
        ("code", 'display(pd.Series(report("shap_summary.json")))\nfigure("shap_global_importance.png")\nfigure("shap_summary.png")\nfigure("shap_dependence.png")'),
        ("markdown", "## Clinical interpretation and future validation\nThis educational retrospective model is not a clinically validated prediction system. Assess precision, recall, false-positive workload and calibration together. The F2 threshold expresses an illustrative preference for recall, not an established clinical utility function. Before UK healthcare use, obtain contemporary NHS data, undertake temporal/external validation, assess coding and demographic compatibility, calibrate using development data, evaluate net benefit/capacity constraints, and conduct governance and prospective evaluation."),
    ])
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    for name, value in [("01_exploratory_analysis.ipynb", eda), ("02_features_and_leakage.ipynb", features),
                        ("03_modelling_and_explainability.ipynb", modelling)]:
        path = directory / name
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing notebook: {path}")
        nbformat.write(value, path)


def execute_notebooks(filename=None):
    if not (ROOT / "reports/run_metadata.json").exists():
        raise RuntimeError("Run python -m hospital_readmission.run before executing notebooks")
    with tempfile.TemporaryDirectory(prefix="readmission-jupyter-") as directory:
        directory = Path(directory)
        overrides = {"JUPYTER_PATH": str(directory / "share/jupyter"),
                     "IPYTHONDIR": str(directory / "ipython"),
                     "JUPYTER_RUNTIME_DIR": str(directory / "runtime"),
                     "JUPYTER_CONFIG_DIR": str(directory / "config")}
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(directory),
                            "--name", "hospital-readmission"], check=True, capture_output=True, text=True)
            paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
            if filename is not None:
                paths = [path for path in paths if path.name == filename]
            for path in paths:
                value = nbformat.read(path, as_version=4)
                client = NotebookClient(value, timeout=300, kernel_name="hospital-readmission",
                                        kernel_manager_class=LocalKernelManager,
                                        resources={"metadata": {"path": str(ROOT)}})
                client.execute()
                nbformat.write(value, path)
                print(f"Executed {path.name}", flush=True)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create the three notebooks only if absent")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--notebook", choices=[path.name for path in (ROOT / "notebooks").glob("*.ipynb")])
    args = parser.parse_args()
    if args.create:
        create_notebooks()
    if args.execute:
        execute_notebooks(args.notebook)
    if not args.create and not args.execute:
        parser.error("Choose --create and/or --execute")


if __name__ == "__main__":
    main()
