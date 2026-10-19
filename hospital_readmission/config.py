"""Prespecified experiment settings; no test-set-driven configuration."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    test_folds: int = 5
    validation_folds: int = 5
    cv_folds: int = 3
    search_iterations: int = 4
    threads: int = 2
    bootstrap_iterations: int = 200
    shap_rows: int = 80
    missingness_threshold: float = 0.95
    selection_metric: str = "mean training cross-validation average precision"
    threshold_rule: str = "maximum validation F2; highest threshold breaks ties"

    def to_dict(self):
        return asdict(self)
