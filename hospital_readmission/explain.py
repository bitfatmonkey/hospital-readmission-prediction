"""SHAP associations for the model selected using training CV only."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.special import expit

from hospital_readmission.reporting import save_figure, write_json


def explain_model(pipeline, name, X_train, X_test, reports, figures, sample_size=80, seed=42):
    # A fixed random test sample; no selection of particularly flattering cases.
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=seed)
    transformed = pipeline[:-1].transform(sample)
    dense = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    estimator = pipeline.named_steps["model"]
    if name == "Logistic Regression":
        background = pipeline[:-1].transform(X_train.sample(min(100, len(X_train)), random_state=seed))
        background = background.toarray() if sparse.issparse(background) else background
        explanation = shap.LinearExplainer(estimator, background)(dense)
        expected_output = estimator.decision_function(transformed)
        scale = "log odds"
    elif name in ("Random Forest", "XGBoost"):
        if name == "XGBoost":
            # XGBoost distinguishes absent CSR entries from explicit dense zeros.
            # Explain the exact sparse representation used by the fitted pipeline.
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed)
            explanation = shap.Explanation(
                values=values, base_values=np.full(len(sample), explainer.expected_value),
                data=dense, feature_names=names,
            )
            expected_output = estimator.predict(transformed, output_margin=True)
            scale = "log odds"
        else:
            explanation = shap.TreeExplainer(estimator)(dense)
            explanation = explanation[:, :, 1]
            expected_output = estimator.predict_proba(transformed)[:, 1]
            scale = "positive-class probability"
    else:
        write_json(Path(reports) / "shap_summary.json", {"status": "not applicable to prior-only Dummy baseline"})
        return
    explanation.feature_names = names
    np.testing.assert_allclose(explanation.values.sum(axis=1) + explanation.base_values,
                               expected_output, atol=2e-4, rtol=1e-4)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    probabilities = expit(reconstructed) if scale == "log odds" else reconstructed
    np.testing.assert_allclose(probabilities, pipeline.predict_proba(sample)[:, 1], atol=2e-5, rtol=1e-4)
    importance = pd.DataFrame({"feature": names, "mean_absolute_shap": np.abs(explanation.values).mean(axis=0)})
    importance = importance.sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(Path(reports) / "shap_feature_importance.csv", index=False)
    top = importance.head(20).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top.feature, top.mean_absolute_shap, color="#0072B2")
    ax.set(xlabel=f"Mean absolute SHAP value ({scale})", title=f"{name}: global predictive associations, {len(sample)} real test encounters")
    save_figure(fig, Path(figures) / "shap_global_importance.png")
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    save_figure(plt.gcf(), Path(figures) / "shap_summary.png")
    strongest = names.index(importance.iloc[0].feature)
    shap.plots.scatter(explanation[:, strongest], show=False)
    save_figure(plt.gcf(), Path(figures) / "shap_dependence.png")
    write_json(Path(reports) / "shap_summary.json", {
        "model": name, "sample_size": len(sample), "feature_count": len(names), "output_scale": scale,
        "additivity_verified": True, "pipeline_predictions_verified": True,
        "input_representation": "original sparse model input; dense data used only for plotting",
        "top_features": importance.head(10).to_dict(orient="records"),
        "interpretation": "Predictive associations, not treatment effects or causal explanations",
    })
