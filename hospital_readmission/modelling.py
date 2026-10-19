"""Small, explicit CPU search spaces with training-only imbalance alternatives."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from hospital_readmission.features import ClinicalFeatureEngineer, MissingnessDropper, make_preprocessing

MODEL_NAMES = ("Dummy", "Logistic Regression", "Random Forest", "XGBoost")


def make_model(name, config):
    estimators = {
        "Dummy": DummyClassifier(strategy="prior", random_state=config.seed),
        "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1500, random_state=config.seed),
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=20,
                                                max_features="sqrt", n_jobs=config.threads, random_state=config.seed),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=.07, min_child_weight=10,
                                  subsample=.8, colsample_bytree=.8, reg_lambda=5,
                                  tree_method="hist", device="cpu", n_jobs=config.threads,
                                  random_state=config.seed, eval_metric="logloss"),
    }
    return Pipeline([
        ("clinical", ClinicalFeatureEngineer()),
        ("missingness", MissingnessDropper(config.missingness_threshold)),
        ("preprocess", make_preprocessing(scale_numeric=name == "Logistic Regression")),
        ("model", estimators[name]),
    ])


def search_space(name, y):
    if name == "Logistic Regression":
        return {"model__C": [.1, 1.0], "model__class_weight": [None, "balanced"]}
    if name == "Random Forest":
        return {"model__max_depth": [12, 18], "model__class_weight": [None, "balanced_subsample"]}
    if name == "XGBoost":
        ratio = float((len(y) - y.sum()) / y.sum())
        return {"model__max_depth": [3, 5], "model__scale_pos_weight": [1.0, ratio]}
    return None


def tune_model(name, X, y, folds, config):
    pipeline = make_model(name, config)
    if name == "Dummy":
        scores = cross_val_score(pipeline, X, y, cv=folds, scoring="average_precision", n_jobs=1, error_score="raise")
        pipeline.fit(X, y)
        return pipeline, {"best_params": {}, "cv_average_precision": float(scores.mean()),
                          "cv_ap_std": float(scores.std())}, None
    search = RandomizedSearchCV(
        pipeline, search_space(name, y), n_iter=config.search_iterations,
        scoring={"ap": "average_precision", "roc_auc": "roc_auc"}, refit="ap",
        cv=folds, n_jobs=1, random_state=config.seed, error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)
    metadata = {"best_params": search.best_params_, "cv_average_precision": float(search.best_score_),
                "cv_ap_std": float(search.cv_results_["std_test_ap"][search.best_index_])}
    return search.best_estimator_, metadata, search.cv_results_
